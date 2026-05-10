#define GGML_BACKEND_SHARED
#include "ggml-impl.h"
#include "ggml-xdna.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#define __restrict__ __restrict
#define aligned_alloc(align, size) _aligned_malloc(size, align)
#define free(ptr) _aligned_free(ptr)
#define GGML_XDNA_PATH_SEP "\\"
#else
#define GGML_XDNA_PATH_SEP "/"
#endif

// Return the Python interpreter command for invoking compile.py.
// Honours GGML_XDNA_PYTHON_CMD env var; defaults to "python" (works on
// Windows and most Linux venvs).  Prefer this over hardcoding "python3".
static const char * xdna_python_cmd() {
    const char * env = getenv("GGML_XDNA_PYTHON_CMD");
    return env ? env : "python";
}

// Platform-specific null redirect suffix for system() calls.
// Suppresses compile.py stdout/stderr to avoid polluting llama-cli output.
static const char * xdna_null_redirect() {
#ifdef _WIN32
    return " > NUL 2>&1";
#else
    return " > /dev/null 2>&1";
#endif
}

// Session-wide buffer-traffic counters (behind XDNA_DEBUG). These measure
// how much data the scheduler moves through our buffer interface — a proxy
// for inter-backend copy overhead. Logged on backend_free.
static std::atomic<size_t> g_set_tensor_bytes{0};
static std::atomic<size_t> g_set_tensor_calls{0};
static std::atomic<size_t> g_get_tensor_bytes{0};
static std::atomic<size_t> g_get_tensor_calls{0};
static std::atomic<size_t> g_cpy_tensor_bytes{0};
static std::atomic<size_t> g_cpy_tensor_calls{0};

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_kernel.h"  // xrt::runlist (production-linked in libxrt_coreutil)
#include "xrt/experimental/xrt_elf.h"     // xrt::elf — fused tblock single-ELF loader
#include "xrt/experimental/xrt_ext.h"     // xrt::ext::kernel(ctx, "main:sequence")

// ============================================================================
// Cached kernel entry — one per unique (op, shape, dtype) tuple
// ============================================================================

enum xdna_op_kind : int {
    XDNA_OP_GEMM               = 0,  // M>=32 prefill MUL_MAT via IRON GEMM
    XDNA_OP_GEMV               = 1,  // M==1  decode  MUL_MAT via IRON GEMV
    XDNA_OP_SWIGLU_DECODE      = 2,  // M==1  fused SwiGLU FFN (gate/up/down + SiLU + mul)
    XDNA_OP_SWIGLU_PREFILL     = 3,  // M>=32 fused SwiGLU FFN (gate/up/down + SiLU + mul)
    XDNA_OP_SWIGLU_DECODE_INT8 = 4,  // M==1  W8A16 fused SwiGLU FFN (int8 weights, bf16 acts)
    XDNA_OP_SWIGLU_FUSED_INT8  = 5,  // M==1  fused gate+up+silu+mul INT8 + standalone down GEMV
    XDNA_OP_SWIGLU_PREFILL_INT8 = 6,  // M>=32 W8A8 INT8 SwiGLU FFN (both weights and activations int8)
    XDNA_OP_QKV = 7,  // M==1 fused Q/K/V (single GEMV with concatenated weights)
    XDNA_OP_RMS_NORM            = 8,  // standalone RMSNorm (bf16, eps=1e-5 baked-in)
    XDNA_OP_ATTENTION_PREFILL   = 9,  // chained attention block (RMSNorm+QKV+RoPE+MHA+O+residual)
};

struct xdna_kernel_entry {
    xdna_op_kind  op_kind;
    xrt::xclbin   xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel   kernel;
    std::vector<char> insts;
    xrt::bo       insts_bo;
    int64_t       M, K, N;
    // Persistent per-call BOs. Shape is fixed by (M,K,N) for this kernel,
    // so we allocate once and memcpy new data in per call.
    // unique_ptr defers construction until first use (needs kernel group_id).
    // GEMM: a_bo = activation (M*K bf16), c_bo = output (M*N bf16)
    // GEMV: a_bo = vector activation (K bf16), c_bo = output (N bf16)
    std::unique_ptr<xrt::bo> a_bo;
    std::unique_ptr<xrt::bo> c_bo;
    // Cache of per-weight-pointer bf16 weight buffers keyed by src0->data.
    // Weights are immutable after load, so we convert+DMA once and reuse.
    // GEMM: transposed to [K,N] row-major. GEMV: native [N,K] row-major.
    std::unordered_map<const void *, xrt::bo> b_bo_cache;
    std::unique_ptr<std::mutex> b_bo_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// Fused SwiGLU FFN kernel entry — one chained xclbin, 4 sub-kernels
// ============================================================================

// Sub-kernel slot indices into the chained SwiGLU xclbin.
//
// IRON-windows uses different kernel counts per path:
//   - Decode bf16: 2 kernels (fused gate+up+silu+mul, down GEMV)
//   - Prefill bf16: 3 kernels (gemm_1, silu_mul, gemm_2)
//   - Decode INT8: NOT AVAILABLE in IRON-windows
//   - Prefill INT8: NOT AVAILABLE in IRON-windows
//
// We use a fixed array size of 3 (the max across active paths) and
// per-path kernel counts to loop only over valid slots.

// Slot indices — reused across paths where semantics overlap.
// Decode: FUSED=0 (gate+up+silu+mul), DOWN=1
// Prefill: GATE_UP=0, SILU_MUL=1, DOWN=2
enum xdna_swiglu_slot : int {
    XDNA_SWIGLU_SLOT_0      = 0,  // decode: fused gate+up+silu+mul | prefill: gemm_1 (gate/up)
    XDNA_SWIGLU_SLOT_1      = 1,  // decode: down GEMV              | prefill: silu_mul
    XDNA_SWIGLU_SLOT_2      = 2,  // decode: unused                 | prefill: gemm_2 (down)
    XDNA_SWIGLU_MAX_SLOTS   = 3,
};

// Per-path kernel counts.
static constexpr int XDNA_SWIGLU_DECODE_NUM_KERNELS = 2;
static constexpr int XDNA_SWIGLU_PREFILL_NUM_KERNELS = 3;

// Shorthand aliases for readability in dispatch code.
#define XDNA_SWIGLU_FUSED     XDNA_SWIGLU_SLOT_0  // decode fused kernel
#define XDNA_SWIGLU_DOWN      XDNA_SWIGLU_SLOT_1  // decode down GEMV
#define XDNA_SWIGLU_GATE_UP   XDNA_SWIGLU_SLOT_0  // prefill gate/up GEMM
#define XDNA_SWIGLU_SILU_MUL  XDNA_SWIGLU_SLOT_1  // prefill fused SiLU+Mul
#define XDNA_SWIGLU_DOWN_P    XDNA_SWIGLU_SLOT_2  // prefill down GEMM

// Legacy slot names — only used by INT8 dead code below.
// Wrapped in #if 0 since IRON-windows does not compile INT8 kernels.
#if 0
#define XDNA_SWIGLU_MATMUL_1  XDNA_SWIGLU_SLOT_0
#define XDNA_SWIGLU_MATMUL_2  XDNA_SWIGLU_SLOT_2
#define XDNA_SWIGLU_SILU      XDNA_SWIGLU_SLOT_1
#define XDNA_SWIGLU_ELTWISE   XDNA_SWIGLU_SLOT_1
#endif

// Kernel attribute names inside IRON-windows SwiGLU xclbin.
// Indexed by xdna_swiglu_slot; only [0..num_kernels-1] are valid per path.
static constexpr const char * XDNA_SWIGLU_DECODE_KERNEL_NAMES[XDNA_SWIGLU_MAX_SLOTS] = {
    "swiglu_fused",     // slot 0: fused gate+up dual-GEMV + SiLU + Mul
    "swiglu_gemv_2",    // slot 1: down projection GEMV
    nullptr,            // slot 2: unused for decode
};
static constexpr const char * XDNA_SWIGLU_PREFILL_KERNEL_NAMES[XDNA_SWIGLU_MAX_SLOTS] = {
    "swiglu_gemm_1",    // slot 0: gate/up GEMM
    "swiglu_silu_mul",  // slot 1: fused SiLU + element-wise Mul
    "swiglu_gemm_2",    // slot 2: down GEMM
};

// Per-slot insts filename tags (no "swiglu_" prefix). The bridge stages insts
// files as "<cache_dir>/swiglu_<tag>.insts".
static constexpr const char * XDNA_SWIGLU_DECODE_INSTS_TAGS[XDNA_SWIGLU_MAX_SLOTS] = {
    "fused", "gemv_2", nullptr,
};
static constexpr const char * XDNA_SWIGLU_PREFILL_INSTS_TAGS[XDNA_SWIGLU_MAX_SLOTS] = {
    "gemm_1", "silu_mul", "gemm_2",
};

struct xdna_swiglu_kernel_entry {
    xdna_op_kind    op_kind;  // XDNA_OP_SWIGLU_{DECODE,PREFILL}
    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernels[XDNA_SWIGLU_MAX_SLOTS];
    int             num_kernels;  // actual kernel count for this path
    std::vector<char> insts_data[XDNA_SWIGLU_MAX_SLOTS];
    xrt::bo         insts_bo  [XDNA_SWIGLU_MAX_SLOTS];

    int64_t embedding_dim;
    int64_t hidden_dim;
    int64_t seq_len;  // 1 for decode, >=32 for prefill
    int     num_cols;
    // INT8-only: group_size baked into the compiled kernel (-DGROUP_SIZE) and
    // the clamped tile_size_input for each inner gemv stage. Unused for bf16
    // paths (left at their default-initialised zero values).
    int     group_size    = 0;
    int     mm1_m_input   = 0;  // gemv_int8_1 tile_size_input after L1-budget clamp
    int     mm2_m_input   = 0;  // gemv_int8_2 tile_size_input after L1-budget clamp

    // I/O BOs (lazy — allocated on first dispatch).
    // input_bo : embedding_dim * M bf16 (M=1 for decode)
    // output_bo: embedding_dim * M bf16
    std::unique_ptr<xrt::bo> input_bo;
    std::unique_ptr<xrt::bo> output_bo;

    // Intermediate BOs (lazy). All are hidden_dim * M bf16.
    std::unique_ptr<xrt::bo> left_bo;          // matmul_1(w_gate, input)
    std::unique_ptr<xrt::bo> right_bo;         // matmul_1(w_up,   input)
    std::unique_ptr<xrt::bo> left_swished_bo;  // silu(left)
    std::unique_ptr<xrt::bo> intermediate_bo;  // eltwise_mul(left_swished, right)

    // Weight BO caches — one map per slot (gate, up, down), keyed by the
    // ggml weight tensor's data pointer. Multi-layer models share a kernel
    // entry across layers (same shape ⇒ same cache key), so we need separate
    // BOs for each layer's weights. Mirrors xdna_kernel_entry::b_bo_cache.
    // For bf16 paths these hold bf16 transposed/native weights; for INT8
    // they hold packed uint8 byte buffers (interleaved int8 + bf16 scales).
    std::unordered_map<const void *, xrt::bo> w1_bo_cache;  // gate
    std::unordered_map<const void *, xrt::bo> w2_bo_cache;  // up
    std::unordered_map<const void *, xrt::bo> w3_bo_cache;  // down

    // Fused gate+up weight BO cache for decode DualGEMVSiLUMul kernel.
    // Keyed by a combined 64-bit hash of both source data pointers.
    std::unordered_map<uint64_t, xrt::bo> w_fused_bo_cache;

    std::unique_ptr<std::mutex> weights_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// Fused INT8 SwiGLU entry — single combined xclbin with 2 kernel entries
// (fused gate+up+silu+mul + down GEMV), dispatched via xrt::runlist.
// ============================================================================

struct xdna_swiglu_fused_entry {
    xdna_op_kind  op_kind;  // always XDNA_OP_SWIGLU_FUSED_INT8

    // Single combined xclbin with 2 kernel entries.
    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     fused_kernel;   // gate+up+silu+mul
    xrt::kernel     down_kernel;    // standalone down GEMV

    // Per-kernel insts data + BOs.
    std::vector<char> fused_insts_data;
    xrt::bo         fused_insts_bo;
    std::vector<char> down_insts_data;
    xrt::bo         down_insts_bo;

    int64_t embedding_dim;
    int64_t hidden_dim;
    int     num_cols;
    int     group_size;
    int     fused_m_input;  // clamped tile_size_input for gate/up (fused kernel)
    int     down_m_input;   // clamped tile_size_input for down GEMV

    // I/O BOs (lazy — allocated on first dispatch).
    std::unique_ptr<xrt::bo> input_bo;         // embedding_dim bf16
    std::unique_ptr<xrt::bo> intermediate_bo;  // hidden_dim bf16
    std::unique_ptr<xrt::bo> output_bo;        // embedding_dim bf16

    // Weight BO caches — one map per slot, keyed by ggml weight tensor data ptr.
    std::unordered_map<const void *, xrt::bo> w_gate_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_up_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_down_bo_cache;

    std::unique_ptr<std::mutex> weights_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// Fused QKV projection entry — single combined xclbin with 1 fused kernel
// (fused_qkv GEMV), dispatched via a single xrt::run.
// Weights Wq, Wk, Wv are concatenated row-wise into one matrix on host.
// ============================================================================

struct xdna_qkv_entry {
    xdna_op_kind  op_kind;  // always XDNA_OP_QKV

    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     fused_kernel;     // single fused_qkv GEMV

    std::vector<char> insts_data;     // qkv_main.insts
    xrt::bo           insts_bo;

    int64_t embedding_dim;
    int64_t q_dim, k_dim, v_dim;
    int64_t total_out;   // q_dim + k_dim + v_dim
    int     num_cols;

    // I/O BOs (lazy)
    std::unique_ptr<xrt::bo> input_bo;      // embedding_dim bf16
    std::unique_ptr<xrt::bo> output_bo;     // total_out bf16

    // Weight BO cache — keyed by Q weight ptr (assumes q/k/v change together).
    std::unordered_map<const void *, xrt::bo> w_concat_bo_cache;

    std::unique_ptr<std::mutex> weights_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// Attention-prefill chained entry — 11 sub-kernels in one combined.xclbin.
// Mirrors iron/operators/attention_block_prefill/op.py AttentionBlockPrefill.
// Kernel names inside the xclbin are "attn_prefill_<slot>"; insts files are
// staged at "<bundle_dir>/attn_prefill_<slot>.insts".
// ============================================================================

enum xdna_attn_prefill_slot : int {
    XDNA_ATTN_RMS_NORM = 0,
    XDNA_ATTN_GEMM_Q   = 1,
    XDNA_ATTN_GEMM_KV  = 2,  // shared by K and V projections
    XDNA_ATTN_ROPE_Q   = 3,
    XDNA_ATTN_ROPE_K   = 4,
    XDNA_ATTN_PERM_Q   = 5,
    XDNA_ATTN_PERM_KV  = 6,  // shared by K and V permutations
    XDNA_ATTN_MHA      = 7,
    XDNA_ATTN_PERM_O   = 8,
    XDNA_ATTN_GEMM_O   = 9,
    XDNA_ATTN_ADD      = 10,
    XDNA_ATTN_NUM_SLOTS = 11,
};

static constexpr const char * XDNA_ATTN_PREFILL_KERNEL_NAMES[XDNA_ATTN_NUM_SLOTS] = {
    "attn_prefill_rms_norm",
    "attn_prefill_gemm_q",
    "attn_prefill_gemm_kv",
    "attn_prefill_rope_q",
    "attn_prefill_rope_k",
    "attn_prefill_perm_q",
    "attn_prefill_perm_kv",
    "attn_prefill_mha",
    "attn_prefill_perm_o",
    "attn_prefill_gemm_o",
    "attn_prefill_add",
};

static constexpr const char * XDNA_ATTN_PREFILL_INSTS_TAGS[XDNA_ATTN_NUM_SLOTS] = {
    "rms_norm", "gemm_q", "gemm_kv", "rope_q", "rope_k",
    "perm_q",   "perm_kv","mha",    "perm_o", "gemm_o", "add",
};

struct xdna_attention_prefill_entry {
    xdna_op_kind    op_kind;  // always XDNA_OP_ATTENTION_PREFILL

    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernels[XDNA_ATTN_NUM_SLOTS];
    std::vector<char> insts_data[XDNA_ATTN_NUM_SLOTS];
    xrt::bo         insts_bo  [XDNA_ATTN_NUM_SLOTS];

    // Shape params (baked into the xclbin).
    int64_t seq_len_padded;
    int64_t embed_dim;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;

    std::string cache_key;

    // Per-call persistent BOs (lazy, allocated on first dispatch).
    //   bo_x_in    : (seq*embed) bf16 — residual input, also fed to RMSNorm
    //   bo_x_norm  : (seq*embed) bf16 — RMSNorm output
    //   bo_q_proj  : (seq*H*d)   bf16 — GEMM_Q output (pre-RoPE)
    //   bo_k_proj  : (seq*KV*d)  bf16 — GEMM_KV(K) output
    //   bo_v_proj  : (seq*KV*d)  bf16 — GEMM_KV(V) output
    //   bo_q_rope  : (seq*H*d)   bf16 — RoPE(Q)
    //   bo_k_rope  : (seq*KV*d)  bf16 — RoPE(K)
    //   bo_q_perm  : (H*seq_pad*d) bf16  (MHA's Q/K/V all share this buffer size)
    //   bo_k_perm  : (H*seq_pad*d) bf16  (oversized to match MHA's arg_spec)
    //   bo_v_perm  : (H*seq_pad*d) bf16  (ditto)
    //   bo_attn_out: (H*seq_pad*d) bf16 — MHA output
    //   bo_o_perm  : (seq*H*d)   bf16 — perm_o output (seq,H,d)
    //   bo_o_proj  : (seq*embed) bf16 — GEMM_O output
    //   bo_output  : (seq*embed) bf16 — residual ADD output (final)
    //   bo_gain    : (embed)     bf16 — RMSNorm gain (per-layer, repacked-cached)
    //   bo_angles  : (seq*d)     bf16 — RoPE cos/sin LUT (per-dispatch, positions vary)
    std::unique_ptr<xrt::bo> bo_x_in;
    std::unique_ptr<xrt::bo> bo_x_norm;
    std::unique_ptr<xrt::bo> bo_q_proj;
    std::unique_ptr<xrt::bo> bo_k_proj;
    std::unique_ptr<xrt::bo> bo_v_proj;
    std::unique_ptr<xrt::bo> bo_q_rope;
    std::unique_ptr<xrt::bo> bo_k_rope;
    std::unique_ptr<xrt::bo> bo_q_perm;
    std::unique_ptr<xrt::bo> bo_k_perm;
    std::unique_ptr<xrt::bo> bo_v_perm;
    std::unique_ptr<xrt::bo> bo_attn_out;
    std::unique_ptr<xrt::bo> bo_o_perm;
    std::unique_ptr<xrt::bo> bo_o_proj;
    std::unique_ptr<xrt::bo> bo_output;
    std::unique_ptr<xrt::bo> bo_angles;

    // Per-layer weight / gain BOs, cached by source ggml tensor data pointer.
    // w_q/w_k/w_v/w_o hold bf16 transposed [K,N] layouts (GEMM B-buffer);
    // gain holds bf16 row of size embed_dim.
    std::unordered_map<const void *, xrt::bo> w_q_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_k_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_v_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_o_bo_cache;
    std::unordered_map<const void *, xrt::bo> gain_bo_cache;

    std::unique_ptr<std::mutex> weights_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// TransformerBlockPrefill chained entry — 17 sub-kernels (attention + SwiGLU
// FFN + two residuals) in one combined.xclbin. Mirrors
// iron/operators/transformer_block_prefill/op.py. Kernel names inside the
// xclbin are "tblock_prefill_<slot>"; insts files are staged at
// "<bundle_dir>/tblock_prefill_<slot>.insts".
// ============================================================================

enum xdna_tblock_prefill_slot : int {
    XDNA_TBLOCK_RMS_NORM_ATTN = 0,
    XDNA_TBLOCK_GEMM_Q        = 1,
    XDNA_TBLOCK_GEMM_KV       = 2,  // shared K/V
    XDNA_TBLOCK_ROPE_Q        = 3,
    XDNA_TBLOCK_ROPE_K        = 4,
    XDNA_TBLOCK_PERM_Q        = 5,
    XDNA_TBLOCK_PERM_KV       = 6,  // shared K/V
    XDNA_TBLOCK_MHA           = 7,
    XDNA_TBLOCK_PERM_O        = 8,
    XDNA_TBLOCK_GEMM_O        = 9,
    XDNA_TBLOCK_ADD_ATTN      = 10,
    XDNA_TBLOCK_RMS_NORM_FFN  = 11,
    XDNA_TBLOCK_GEMM_GATE_UP  = 12,  // shared gate/up
    XDNA_TBLOCK_SILU          = 13,
    XDNA_TBLOCK_ELTWISE_MUL   = 14,
    XDNA_TBLOCK_GEMM_DOWN     = 15,
    XDNA_TBLOCK_ADD_FFN       = 16,
    XDNA_TBLOCK_NUM_SLOTS     = 17,
};

static constexpr const char * XDNA_TBLOCK_PREFILL_KERNEL_NAMES[XDNA_TBLOCK_NUM_SLOTS] = {
    "tblock_prefill_rms_norm_attn",
    "tblock_prefill_gemm_q",
    "tblock_prefill_gemm_kv",
    "tblock_prefill_rope_q",
    "tblock_prefill_rope_k",
    "tblock_prefill_perm_q",
    "tblock_prefill_perm_kv",
    "tblock_prefill_mha",
    "tblock_prefill_perm_o",
    "tblock_prefill_gemm_o",
    "tblock_prefill_add_attn",
    "tblock_prefill_rms_norm_ffn",
    "tblock_prefill_gemm_gate_up",
    "tblock_prefill_silu",
    "tblock_prefill_eltwise_mul",
    "tblock_prefill_gemm_down",
    "tblock_prefill_add_ffn",
};

static constexpr const char * XDNA_TBLOCK_PREFILL_INSTS_TAGS[XDNA_TBLOCK_NUM_SLOTS] = {
    "rms_norm_attn", "gemm_q", "gemm_kv", "rope_q", "rope_k",
    "perm_q", "perm_kv", "mha", "perm_o", "gemm_o", "add_attn",
    "rms_norm_ffn", "gemm_gate_up", "silu", "eltwise_mul",
    "gemm_down", "add_ffn",
};

struct xdna_transformer_block_prefill_entry {
    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernels[XDNA_TBLOCK_NUM_SLOTS];
    std::vector<char> insts_data[XDNA_TBLOCK_NUM_SLOTS];
    xrt::bo         insts_bo  [XDNA_TBLOCK_NUM_SLOTS];

    int64_t seq_len_padded;
    int64_t embed_dim;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t ffn_hidden_dim;

    std::string cache_key;

    // Attention-side persistent BOs (same layout as attn_prefill_entry).
    std::unique_ptr<xrt::bo> bo_x_in;
    std::unique_ptr<xrt::bo> bo_x_norm;
    std::unique_ptr<xrt::bo> bo_q_proj;
    std::unique_ptr<xrt::bo> bo_k_proj;
    std::unique_ptr<xrt::bo> bo_v_proj;
    std::unique_ptr<xrt::bo> bo_q_rope;
    std::unique_ptr<xrt::bo> bo_k_rope;
    std::unique_ptr<xrt::bo> bo_q_perm;
    std::unique_ptr<xrt::bo> bo_k_perm;
    std::unique_ptr<xrt::bo> bo_v_perm;
    std::unique_ptr<xrt::bo> bo_attn_out;
    std::unique_ptr<xrt::bo> bo_o_perm;
    std::unique_ptr<xrt::bo> bo_o_proj;
    std::unique_ptr<xrt::bo> bo_attn_residual;
    std::unique_ptr<xrt::bo> bo_angles;
    // FFN-side persistent BOs.
    std::unique_ptr<xrt::bo> bo_ffn_norm;
    std::unique_ptr<xrt::bo> bo_gate_out;
    std::unique_ptr<xrt::bo> bo_up_out;
    std::unique_ptr<xrt::bo> bo_silu_out;
    std::unique_ptr<xrt::bo> bo_mul_out;
    std::unique_ptr<xrt::bo> bo_down_out;
    std::unique_ptr<xrt::bo> bo_output;

    std::unordered_map<const void *, xrt::bo> w_q_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_k_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_v_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_o_bo_cache;
    std::unordered_map<const void *, xrt::bo> gain_attn_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_gate_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_up_bo_cache;
    std::unordered_map<const void *, xrt::bo> w_down_bo_cache;
    std::unordered_map<const void *, xrt::bo> gain_ffn_bo_cache;

    std::unique_ptr<std::mutex> weights_mutex = std::make_unique<std::mutex>();
};

// ============================================================================
// Standalone RMSNorm entry — single-kernel xclbin, one kernel, one insts file.
// Staged under a bundle dir (combined.xclbin + rms_norm_main.insts) so the
// on-disk layout matches chained composites while the in-memory entry stays
// minimal (no runlist, no chained sub-kernels).
// ============================================================================

struct xdna_rms_norm_entry {
    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernel;
    std::vector<char> insts;
    xrt::bo         insts_bo;

    std::string cache_key;
    int64_t size;
    int num_cols;
    int num_channels;
    int tile_size;

    // Persistent per-call BOs. RMSNorm's shape is fixed by (size); one call
    // streams new activations in and results out, both bf16.
    //   in_bo : size * sizeof(uint16_t) bf16 input
    //   out_bo: size * sizeof(uint16_t) bf16 output
    std::unique_ptr<xrt::bo> in_bo;
    std::unique_ptr<xrt::bo> out_bo;
};

// ============================================================================
// Backend context — holds XRT device and kernel cache
// ============================================================================

// ============================================================================
// FlowKV Decode Attention entry — single xclbin with 2-tile streaming pipeline
// (score + value) per KV head group. Implements online softmax attention with
// fused RoPE on Q. Mirrors iron/operators/flowkv_decode/op.py AIEFlowKVDecode.
// ============================================================================

struct xdna_flowkv_entry {
    xrt::xclbin     xclbin;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernel;   // "flowkv_decode"

    std::vector<char> insts_data;
    xrt::bo           insts_bo;

    // Shape params (baked into xclbin).
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;
    int64_t seq_len;
    int64_t chunk_size;
    int     num_cols;

    std::string cache_key;

    // Persistent BOs (lazy, allocated on first dispatch).
    //   bo_kv:   interleaved KV cache (num_kv_heads * seq_len * 2 * head_dim bf16)
    //   bo_q:    packed Q + RoPE angles (num_kv_heads * (group_size*head_dim + head_dim) bf16)
    //   bo_out:  attention output (num_heads * head_dim bf16)
    std::unique_ptr<xrt::bo> bo_kv;
    std::unique_ptr<xrt::bo> bo_q;
    std::unique_ptr<xrt::bo> bo_out;

    std::unique_ptr<std::mutex> mu = std::make_unique<std::mutex>();
};

// Forward declarations used by the context (definitions later in the file).
struct xdna_tblock_fused_group;

struct ggml_backend_xdna_context {
    xrt::device device;
    std::string cache_dir;
    std::string compile_script;
    std::unordered_map<std::string, xdna_kernel_entry> kernel_cache;
    std::unordered_set<std::string> kernel_compile_failed;
    std::unordered_map<std::string, xdna_swiglu_kernel_entry> swiglu_cache;
    std::unordered_map<std::string, xdna_swiglu_fused_entry> swiglu_fused_cache;
    std::unordered_set<std::string> swiglu_compile_failed;
    std::unordered_map<std::string, xdna_qkv_entry> qkv_cache;
    std::unordered_set<std::string> qkv_compile_failed;
    std::unordered_map<std::string, xdna_rms_norm_entry> rms_norm_cache;
    std::unordered_map<std::string, xdna_attention_prefill_entry> attention_prefill_cache;
    std::unordered_map<std::string, xdna_transformer_block_prefill_entry> transformer_block_prefill_cache;
    std::unordered_map<std::string, struct xdna_tblock_fused_entry> tblock_fused_cache;
    std::unordered_map<std::string, struct xdna_flowkv_entry> flowkv_cache;
    std::unordered_set<std::string> flowkv_compile_failed;
    int num_cols;
    // Set of cgraph pointers for which attn_prefill_bulk_prewarm() has already
    // run. Purely an optimization guard — the dispatch path's per-weight cache
    // check covers correctness if we re-enter.
    std::unordered_set<const void *> attn_prewarmed_cgraphs;
    std::unordered_set<const void *> tblock_prewarmed_cgraphs;
    std::unordered_set<const void *> tblock_fused_prewarmed_cgraphs;
    // Layer 4B multi-layer packing groups, keyed by cgraph pointer →
    // map from the FIRST block's rms_norm_idx to a group descriptor. Built
    // at bulk_prewarm time when XDNA_ENABLE_TBLOCK_FUSED_N={2,4} is set;
    // consulted by the dispatch branch to emit one xrt::run per N blocks.
    // Definition is after xdna_transformer_block_match (forward-declared).
    std::unordered_map<const void *, std::unordered_map<int, struct xdna_tblock_fused_group>>
        tblock_fused_groups_per_cgraph;
    std::mutex cache_mutex;
    bool device_valid;
    // CPU backend for delegating ops we don't run on NPU.
    // Our buffers are plain host RAM so CPU can compute on them directly.
    ggml_backend_t cpu_backend;

    ggml_backend_xdna_context() : device_valid(false), cpu_backend(nullptr) {
        // Cache directory
        const char * cache_env = getenv("GGML_XDNA_CACHE_DIR");
        if (cache_env) {
            cache_dir = cache_env;
        } else {
#ifdef _WIN32
            const char * local_appdata = getenv("LOCALAPPDATA");
            if (local_appdata) {
                cache_dir = std::string(local_appdata) + "\\ggml-xdna\\xclbin";
            } else {
                const char * userprofile = getenv("USERPROFILE");
                if (userprofile) {
                    cache_dir = std::string(userprofile) + "\\AppData\\Local\\ggml-xdna\\xclbin";
                } else {
                    cache_dir = "C:\\ggml-xdna\\xclbin";
                }
            }
#else
            const char * home = getenv("HOME");
            if (home) {
                cache_dir = std::string(home) + "/.cache/ggml-xdna/xclbin";
            } else {
                cache_dir = "/tmp/ggml-xdna/xclbin";
            }
#endif
        }

        // Find compile.py: GGML_XDNA_COMPILE_SCRIPT env var overrides,
        // otherwise look in common locations relative to CWD.
        const char * script_env = getenv("GGML_XDNA_COMPILE_SCRIPT");
        if (script_env) {
            compile_script = script_env;
        } else {
#ifdef _WIN32
            compile_script = "ggml\\src\\ggml-xdna\\compile.py";
#else
            compile_script = "ggml/src/ggml-xdna/compile.py";
#endif
        }

        // Initialize XRT device
        try {
            device = xrt::device(0);
            device_valid = true;
            fprintf(stderr, "ggml-xdna: XRT device initialized\n");
        } catch (const std::exception & e) {
            fprintf(stderr, "ggml-xdna: failed to initialize XRT device: %s\n", e.what());
        }

        // CPU backend for fallback. Our buffers are plain host RAM so CPU
        // can operate on them directly — no copies needed.
        cpu_backend = ggml_backend_cpu_init();

        // Number of AIE columns to use
        num_cols = 4;
        const char * cols_env = getenv("GGML_XDNA_NUM_COLS");
        if (cols_env) {
            num_cols = atoi(cols_env);
        }
        fprintf(stderr, "ggml-xdna: using %d AIE columns\n", num_cols);
    }

    ~ggml_backend_xdna_context() {
        if (cpu_backend) {
            ggml_backend_free(cpu_backend);
        }
    }
};

// ============================================================================
// Helpers
// ============================================================================

static std::vector<char> read_binary_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        GGML_LOG_ERROR("ggml-xdna: cannot open file: %s\n", path.c_str());
        return {};
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> data(size);
    f.read(data.data(), size);
    return data;
}

static std::string make_cache_key(xdna_op_kind op_kind,
                                   int64_t M, int64_t K, int64_t N,
                                   const char * dtype_in, int num_cols) {
    char buf[256];
    if (op_kind == XDNA_OP_GEMV) {
        // GEMV: M is implicitly 1, key omits it.
        snprintf(buf, sizeof(buf), "gemv_K%lld_N%lld_%s_%dcol",
                 (long long)K, (long long)N, dtype_in, num_cols);
    } else {
        snprintf(buf, sizeof(buf), "gemm_%lldx%lldx%lld_%s_%dcol",
                 (long long)M, (long long)K, (long long)N, dtype_in, num_cols);
    }
    return std::string(buf);
}

// Convert f32 array to bf16 (truncate lower 16 bits of each float)
static void f32_to_bf16(const float * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(bits));
        // Round to nearest even (add 0x7FFF + bit 16)
        bits += (0x7FFF + ((bits >> 16) & 1));
        dst[i] = (uint16_t)(bits >> 16);
    }
}

// Convert bf16 array to f32 (pad lower 16 bits with zeros)
static void bf16_to_f32(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &bits, sizeof(bits));
    }
}

// Convert an IEEE fp16 (half) value to bf16. Q8_0 block scales are stored as
// ggml_half (fp16); the IRON gemv_int8 kernel expects bf16 per-group scales.
// Implementation: expand fp16 → fp32 (exact), then truncate to bf16 with
// round-to-nearest-even. Handles subnormals, infinity, and NaN per IEEE 754.
static inline uint16_t fp16_to_bf16(uint16_t h) {
    const uint32_t sign    = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp_f16 = (h >> 10) & 0x1F;
    const uint32_t mant_f16 = h & 0x3FF;

    uint32_t f32_bits;
    if (exp_f16 == 0) {
        if (mant_f16 == 0) {
            // Signed zero
            f32_bits = sign;
        } else {
            // Subnormal → normalise.
            uint32_t m = mant_f16;
            int e = -1;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3FF;
            const uint32_t exp_f32 = (uint32_t)(127 - 15 + e + 1);
            f32_bits = sign | (exp_f32 << 23) | (m << 13);
        }
    } else if (exp_f16 == 0x1F) {
        // Inf / NaN: preserve mantissa (keep NaN payload high bit so it stays NaN).
        f32_bits = sign | (0xFFu << 23) | (mant_f16 << 13);
    } else {
        const uint32_t exp_f32 = (uint32_t)((int)exp_f16 - 15 + 127);
        f32_bits = sign | (exp_f32 << 23) | (mant_f16 << 13);
    }

    // Round-to-nearest-even down to bf16 (matches f32_to_bf16 above).
    f32_bits += (0x7FFF + ((f32_bits >> 16) & 1));
    return (uint16_t)(f32_bits >> 16);
}

// Mirror IRON's iron/operators/gemv_int8/op.py::GEMVInt8.__post_init__ L1-
// budget clamp so the C++ side computes a matching tile_size_input without
// re-invoking Python. Must stay in sync with that function; tested via the
// gemv_int8 test suite on the Python side.
//   L1_TOTAL = 64 KB per compute tile
//   STACK_AND_SCRATCH = 2 KB
//   B (activation) buffer: K * 2 bytes (bf16)
//   C (output) buffer, double-buffered: 2 * tso * 2 bytes
//   A (packed weights+scales) buffer, per-tile, double-buffered.
// Returns the clamped tile_size_input (>= 1).
// Auto-select the largest tile_size_input that fits in L1 and divides tso.
// Fewer, larger tiles means fewer DMA transactions — per-tile DMA overhead
// (BD setup + sync) dominates GEMV wall time at M=1.
static int xdna_best_gemv_int8_tile_in(int tso, int64_t K, int group_size) {
    const int L1_TOTAL = 64 * 1024;
    const int STACK_AND_SCRATCH = 2 * 1024;
    const int B_bytes = (int)(K * 2);
    const int C_bytes_x2 = 2 * tso * 2;
    const int A_budget = (L1_TOTAL - STACK_AND_SCRATCH - B_bytes - C_bytes_x2) / 2;
    const int num_groups_per_row = (int)(K / group_size);

    static const int candidates[] = {64, 32, 16, 8, 4, 2, 1};
    for (int tsi : candidates) {
        if (tso % tsi != 0) continue;
        int tile_bytes = tsi * (int)K + tsi * num_groups_per_row * 2;
        if (tile_bytes <= A_budget) return tsi;
    }
    return 1;
}

// Legacy clamp-down function (kept for callers that pass a specific requested_tsi).
static int xdna_clamp_gemv_int8_tile_in(int requested_tsi, int tso,
                                        int64_t K, int group_size) {
    const int L1_TOTAL = 64 * 1024;
    const int STACK_AND_SCRATCH = 2 * 1024;
    const int B_bytes = (int)(K * 2);
    const int C_bytes_x2 = 2 * tso * 2;
    const int A_budget = (L1_TOTAL - STACK_AND_SCRATCH - B_bytes - C_bytes_x2) / 2;
    const int num_groups_per_row = (int)(K / group_size);

    int tsi = requested_tsi;
    while (tsi > 1) {
        int tile_bytes = tsi * (int)K + tsi * num_groups_per_row * 2;
        if (tile_bytes <= A_budget) break;
        tsi /= 2;
    }
    return tsi < 1 ? 1 : tsi;
}

// Repack a Q8_0-quantised ggml tensor into IRON's gemv_int8 per-tile DDR
// layout. Mirrors iron/operators/gemv_int8/reference.py::quantize_and_pack
// byte-for-byte but reads from ggml's Q8_0 block stream instead of generating
// random quantised data.
//
// Inputs (caller guarantees Q8_0 invariants):
//   q8_0  — Q8_0 blob: M rows × (K/32) × 34 bytes
//   M, K  — weight dims (M = output rows for GEMV; K = reduction dim)
//   m_input   — clamped tile_size_input from IRON op (packed tile height)
//   cols      — num_aie_columns (AIE split factor across rows)
//   group_size — must equal Q8_0's 32 for now; we assert that
//
// Output:
//   packed_out — uint8 buffer, total size = cols * tiles_per_col *
//                (m_input*K + m_input*(K/group_size)*2). Caller owns it.
//
// Per tile: [m_input × K signed int8 bytes][m_input × (K/G) × 2 bf16 scale bytes].
// Tile iteration order: col 0 tiles 0..T-1, then col 1, etc.
static void xdna_repack_q8_0_to_gemv_int8(
        const uint8_t * q8_0,
        int64_t M, int64_t K,
        int m_input, int cols, int group_size,
        uint8_t * packed_out) {
    GGML_ASSERT(group_size == 32);  // Q8_0 blocks are fixed at 32 elements.
    GGML_ASSERT(K % group_size == 0);
    GGML_ASSERT(M % cols == 0);

    const int64_t rows_per_col = M / cols;
    GGML_ASSERT(rows_per_col % m_input == 0);

    const int64_t num_groups_per_row = K / group_size;
    const size_t Q8_0_BLOCK_BYTES = 2 + (size_t)group_size;  // fp16 d + 32 int8
    const size_t row_stride_q80 = (size_t)num_groups_per_row * Q8_0_BLOCK_BYTES;
    const size_t packed_bytes_per_tile =
        (size_t)m_input * (size_t)K + (size_t)m_input * (size_t)num_groups_per_row * 2;
    const int64_t tiles_per_col = rows_per_col / m_input;

    for (int col = 0; col < cols; col++) {
        for (int64_t tile_idx = 0; tile_idx < tiles_per_col; tile_idx++) {
            const int64_t row_start = (int64_t)col * rows_per_col + tile_idx * m_input;
            const int64_t flat_tile = (int64_t)col * tiles_per_col + tile_idx;
            const size_t tile_offset = (size_t)flat_tile * packed_bytes_per_tile;

            // 1. INT8 weight bytes for m_input consecutive rows.
            for (int r = 0; r < m_input; r++) {
                const int64_t global_row = row_start + r;
                const uint8_t * row_src =
                    q8_0 + (size_t)global_row * row_stride_q80;
                uint8_t * row_dst = packed_out + tile_offset + (size_t)r * K;
                // Each block: [fp16 scale (2B)][int8 quants (32B)]. Concatenate
                // the int8 payloads across the row's num_groups_per_row blocks.
                for (int64_t g = 0; g < num_groups_per_row; g++) {
                    const uint8_t * blk = row_src + (size_t)g * Q8_0_BLOCK_BYTES;
                    memcpy(row_dst + (size_t)g * group_size,
                           blk + 2,  // skip fp16 scale
                           (size_t)group_size);
                }
            }

            // 2. bf16 scale bytes (converted from fp16) for m_input rows.
            const size_t scale_region_start = tile_offset + (size_t)m_input * (size_t)K;
            for (int r = 0; r < m_input; r++) {
                const int64_t global_row = row_start + r;
                const uint8_t * row_src =
                    q8_0 + (size_t)global_row * row_stride_q80;
                uint16_t * scale_dst = (uint16_t *)(packed_out + scale_region_start
                    + (size_t)r * (size_t)num_groups_per_row * 2);
                for (int64_t g = 0; g < num_groups_per_row; g++) {
                    const uint8_t * blk = row_src + (size_t)g * Q8_0_BLOCK_BYTES;
                    uint16_t fp16_val;
                    memcpy(&fp16_val, blk, 2);  // little-endian fp16 scale
                    scale_dst[g] = fp16_to_bf16(fp16_val);
                }
            }
        }
    }
}

// Repack Q8_0 weights to per-tensor int8 in (K, N) row-major layout for GEMM.
// Returns the per-tensor scale factor.
// Q8_0: each block = [fp16 scale][32 int8 values]. Per-block dequant: x_f = qs * d.
// We compute a global scale = max(|qs[i] * d_block|) / 127 across ALL blocks,
// then re-quantize: out[i] = round(qs[i] * d_block / global_scale).
// Result is (K, N) row-major int8 — transposed from ggml's (N, K) native layout.
static float xdna_repack_q8_0_to_gemm_int8(
        const uint8_t * q8_0,
        int64_t N, int64_t K,  // N=rows in ggml layout, K=cols
        int8_t * out_int8) {    // (K, N) transposed output
    const int group_size = 32;
    const size_t Q8_0_BLOCK_BYTES = 2 + group_size;
    const int64_t groups_per_row = K / group_size;
    const size_t row_stride = (size_t)groups_per_row * Q8_0_BLOCK_BYTES;

    // Pass 1: find global max |value| across all dequantized elements
    float global_max = 0.0f;
    for (int64_t n = 0; n < N; n++) {
        const uint8_t * row = q8_0 + (size_t)n * row_stride;
        for (int64_t g = 0; g < groups_per_row; g++) {
            const uint8_t * blk = row + (size_t)g * Q8_0_BLOCK_BYTES;
            float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *)blk);
            float abs_d = fabsf(d);
            for (int j = 0; j < group_size; j++) {
                float val = abs_d * fabsf((float)(int8_t)blk[2 + j]);
                if (val > global_max) global_max = val;
            }
        }
    }

    float per_tensor_scale = (global_max > 1e-10f) ? (global_max / 127.0f) : 1e-10f;

    // Pass 2: re-quantize and transpose (N, K) → (K, N)
    for (int64_t n = 0; n < N; n++) {
        const uint8_t * row = q8_0 + (size_t)n * row_stride;
        for (int64_t g = 0; g < groups_per_row; g++) {
            const uint8_t * blk = row + (size_t)g * Q8_0_BLOCK_BYTES;
            float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *)blk);
            for (int j = 0; j < group_size; j++) {
                int64_t k = g * group_size + j;
                float val = (float)(int8_t)blk[2 + j] * d;
                int32_t q = (int32_t)roundf(val / per_tensor_scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                out_int8[k * N + n] = (int8_t)q;  // transposed: (K, N) layout
            }
        }
    }

    return per_tensor_scale;
}

// Magic sentinel that GEMM bf16s design.py embeds at RTP write locations.
static constexpr uint32_t XDNA_BF16S_SCALE_MAGIC = 0x34123412;

// Find magic sentinel offsets in insts data (called once at load time).
static std::vector<size_t> xdna_find_bf16s_scale_offsets(const std::vector<char> & insts_data) {
    std::vector<size_t> offsets;
    const size_t n_words = insts_data.size() / sizeof(uint32_t);
    const uint32_t * words = (const uint32_t *)insts_data.data();
    for (size_t i = 0; i < n_words; i++) {
        if (words[i] == XDNA_BF16S_SCALE_MAGIC) {
            offsets.push_back(i);
        }
    }
    return offsets;
}

// Patch bf16s scale at pre-computed offsets.
static void xdna_patch_bf16s_scale(std::vector<char> & insts_data,
                                    const std::vector<size_t> & offsets,
                                    float scale) {
    uint32_t scale_bits;
    memcpy(&scale_bits, &scale, sizeof(scale_bits));
    uint32_t * words = (uint32_t *)insts_data.data();
    for (size_t off : offsets) {
        words[off] = scale_bits;
    }
}

// Per-tensor symmetric quantization: bf16 → int8 + float scale.
static float xdna_quantize_bf16_to_int8(const uint16_t * bf16_data, int8_t * out_int8, size_t n_elems) {
    // Find max abs value
    float max_abs = 0.0f;
    for (size_t i = 0; i < n_elems; i++) {
        // Convert bf16 to f32: shift left by 16 bits
        uint32_t bits = (uint32_t)bf16_data[i] << 16;
        float val;
        memcpy(&val, &bits, sizeof(val));
        float abs_val = fabsf(val);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    float scale = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 1e-10f;
    float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < n_elems; i++) {
        uint32_t bits = (uint32_t)bf16_data[i] << 16;
        float val;
        memcpy(&val, &bits, sizeof(val));
        int32_t q = (int32_t)roundf(val * inv_scale);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out_int8[i] = (int8_t)q;
    }
    return scale;
}

// ============================================================================
// Compilation — call compile.py to generate xclbin + insts
// ============================================================================

static bool ensure_compiled(ggml_backend_xdna_context * ctx,
                            const std::string & cache_key,
                            xdna_op_kind op_kind,
                            int64_t M, int64_t K, int64_t N,
                            const char * dtype_in, int num_cols,
                            const char * dtype_out = nullptr) {
    // Skip recompilation if we already failed for this key
    if (ctx->kernel_compile_failed.count(cache_key)) {
        return false;
    }

    std::string xclbin_path = ctx->cache_dir + GGML_XDNA_PATH_SEP + cache_key + ".xclbin";
    std::string insts_path  = ctx->cache_dir + GGML_XDNA_PATH_SEP + cache_key + ".insts";

    // Check if already cached on disk
    {
        std::ifstream xf(xclbin_path);
        std::ifstream inf(insts_path);
        if (xf.good() && inf.good()) {
            return true;
        }
    }

    // Compile via Python subprocess
    char cmd[1024];
    if (op_kind == XDNA_OP_GEMV) {
        snprintf(cmd, sizeof(cmd),
                 "%s \"%s\" --quiet gemv --N %lld --K %lld --dtype-in %s --dtype-out %s "
                 "--num-aie-columns %d --out \"%s\"%s",
                 xdna_python_cmd(), ctx->compile_script.c_str(),
                 (long long)N, (long long)K,
                 dtype_in, dtype_in,
                 num_cols,
                 xclbin_path.c_str(), xdna_null_redirect());
        fprintf(stderr, "ggml-xdna: compiling GEMV K=%lld N=%lld (first run, will be cached)...\n",
                      (long long)K, (long long)N);
    } else {
        // [INT8 GEMM] Use separate dtype_out when provided (e.g. "i32" for i8 input).
        const char * out_dtype = dtype_out ? dtype_out : dtype_in;
        snprintf(cmd, sizeof(cmd),
                 "%s %s --quiet gemm --M %lld --K %lld --N %lld --dtype-in %s --dtype-out %s "
                 "--num-aie-columns %d --out %s%s",
                 xdna_python_cmd(), ctx->compile_script.c_str(),
                 (long long)M, (long long)K, (long long)N,
                 dtype_in, out_dtype,
                 num_cols,
                 xclbin_path.c_str(), xdna_null_redirect());
        fprintf(stderr, "ggml-xdna: compiling GEMM %lldx%lldx%lld (first run, will be cached)...\n",
                      (long long)M, (long long)K, (long long)N);
    }

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: compilation failed (exit code %d)\n", ret);
        ctx->kernel_compile_failed.insert(cache_key);
        return false;
    }

    // Verify files were created
    std::ifstream xf(xclbin_path);
    std::ifstream inf(insts_path);
    if (!xf.good() || !inf.good()) {
        GGML_LOG_ERROR("ggml-xdna: compilation succeeded but output files missing\n");
        ctx->kernel_compile_failed.insert(cache_key);
        return false;
    }

    fprintf(stderr, "ggml-xdna: compilation complete, cached at %s\n", xclbin_path.c_str());
    return true;
}

// ============================================================================
// XRT kernel loading — load xclbin + insts, create kernel handle
// ============================================================================

static xdna_kernel_entry * get_or_load_kernel(ggml_backend_xdna_context * ctx,
                                               const std::string & cache_key,
                                               xdna_op_kind op_kind,
                                               int64_t M, int64_t K, int64_t N) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);

    // Check in-memory cache
    auto it = ctx->kernel_cache.find(cache_key);
    if (it != ctx->kernel_cache.end()) {
        return &it->second;
    }

    // Load from disk — try flat layout first (cache_key.xclbin + cache_key.insts
    // in cache_dir), fall back to bundle layout (combined.xclbin in subdir).
    // Flat layout is preferred because compile.py generates flat files for GEMV/GEMM.
    std::string xclbin_path = ctx->cache_dir + "\\" + cache_key + ".xclbin";
    std::string insts_path  = ctx->cache_dir + "\\" + cache_key + ".insts";

    // If flat layout doesn't exist, try bundle layout
    // (combined.xclbin in a subdirectory, used by fused kernels like SwiGLU/QKV).
    if (!std::ifstream(xclbin_path).good()) {
        const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
        xclbin_path = bundle_dir + "\\combined.xclbin";
        insts_path  = bundle_dir + "\\" + cache_key + ".insts";
    }

    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    if (dbg) {
        fprintf(stderr, "ggml-xdna: kernel '%s'\n", cache_key.c_str());
        fprintf(stderr, "ggml-xdna:   xclbin: %s (exists: %d)\n", xclbin_path.c_str(), std::ifstream(xclbin_path).good());
        fprintf(stderr, "ggml-xdna:   insts:  %s (exists: %d)\n", insts_path.c_str(), std::ifstream(insts_path).good());
    }

    try {
        xdna_kernel_entry entry;
        entry.op_kind = op_kind;
        entry.M = M;
        entry.K = K;
        entry.N = N;

        // Load xclbin
        entry.xclbin = xrt::xclbin(xclbin_path);
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);

        // Get kernel (IRON kernels are always named "MLIR_AIE")
        entry.kernel = xrt::kernel(entry.hw_ctx, "MLIR_AIE");

        // Load instructions
        entry.insts = read_binary_file(insts_path);
        if (entry.insts.empty()) {
            GGML_LOG_ERROR("ggml-xdna: failed to read insts file: %s\n", insts_path.c_str());
            return nullptr;
        }

        // Create instruction buffer object
        entry.insts_bo = xrt::bo(ctx->device, entry.insts.size(),
                                  xrt::bo::flags::cacheable,
                                  entry.kernel.group_id(1));
        entry.insts_bo.write(entry.insts.data());
        entry.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto [inserted_it, _] = ctx->kernel_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded kernel for %s\n", cache_key.c_str());
        return &inserted_it->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load kernel %s: %s\n",
                       cache_key.c_str(), e.what());
        return nullptr;
    }
}

// ============================================================================
// Op dispatch — MUL_MAT via IRON GEMM
// ============================================================================

static bool xdna_shape_dispatchable(int64_t M, int64_t K, int64_t N);
static bool xdna_shape_dispatchable_gemv(int64_t K, int64_t N);

// GEMM (prefill, M>=32). Caller (graph_compute) must have already verified
// xdna_shape_dispatchable() before invoking this — no CPU fallback here.
// On unexpected XRT failure we log and return without writing dst; the
// resulting garbage is a correctness bug caller-side, not silent degradation.
static void ggml_backend_xdna_mul_mat_gemm(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: mul_mat called but XRT device invalid\n");
        return;
    }

    int num_cols = ctx->num_cols;

    // [INT8 GEMM] Detect Q8_0 weights: use int8 GEMM kernel (dtype_in="i8",
    // dtype_out="i32"). The per-tensor scales from weights and activations
    // are applied on CPU after the i32 accumulator is read back.
    // For bf16/f32 weights, keep the existing bf16 path unchanged.
    const bool use_int8 = (src0->type == GGML_TYPE_Q8_0);
    const char * dtype_in  = use_int8 ? "i8"   : "bf16";
    const char * dtype_out = use_int8 ? "i32"  : nullptr;  // nullptr = same as dtype_in (bf16)

    std::string cache_key = make_cache_key(XDNA_OP_GEMM, M, K, N, dtype_in, num_cols);

    // [INT8 GEMM] Pass explicit dtype_out="i32" so compile.py builds an
    // int8×int8→i32 kernel instead of bf16×bf16→bf16.
    if (!ensure_compiled(ctx, cache_key, XDNA_OP_GEMM, M, K, N, dtype_in, num_cols, dtype_out)) {
        GGML_LOG_ERROR("ggml-xdna: compile failed for %lldx%lldx%lld (dtype=%s)\n",
                       (long long)M, (long long)K, (long long)N, dtype_in);
        return;
    }

    xdna_kernel_entry * entry = get_or_load_kernel(ctx, cache_key, XDNA_OP_GEMM, M, K, N);
    if (!entry) {
        return;
    }

    // [INT8 GEMM] Persistent weight scale cache: src0->data → per-tensor scale.
    // Needed because the scale is computed once during weight BO creation but
    // must be applied on every dispatch for i32→f32 dequantization.
    // Keyed identically to b_bo_cache (by weight data pointer).
    // Safe as static (not thread_local): ggml graph_compute dispatches
    // MUL_MAT nodes sequentially on a single thread.
    static std::unordered_map<const void *, float> s_wt_scale_cache;

    try {
        size_t a_elems = M * K;
        size_t b_elems = N * K;
        size_t c_elems = M * N;

        // [INT8 GEMM] A and B are 1 byte/elem (int8); C is 4 bytes/elem (i32).
        // Bf16 path: A and B are 2 bytes/elem (bf16); C is 2 bytes/elem (bf16).
        size_t a_bytes = use_int8 ? a_elems * 1 : a_elems * sizeof(uint16_t);
        size_t b_bytes = use_int8 ? b_elems * 1 : b_elems * sizeof(uint16_t);
        size_t c_bytes = use_int8 ? c_elems * 4 : c_elems * sizeof(uint16_t);

        // Lazily allocate persistent A and C BOs on first dispatch for this kernel.
        if (!entry->a_bo) {
            entry->a_bo = std::make_unique<xrt::bo>(ctx->device, a_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(3));
        }
        if (!entry->c_bo) {
            entry->c_bo = std::make_unique<xrt::bo>(ctx->device, c_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(5));
        }

        // [INT8 GEMM] Quantize bf16/f32 activation to per-tensor int8.
        // bf16 path: convert f32→bf16 or memcpy as before.
        float act_scale = 1.0f;
        if (use_int8) {
            // First convert activation to bf16, then quantize bf16→int8.
            std::vector<uint16_t> act_bf16(a_elems);
            if (src1->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)src1->data, act_bf16.data(), a_elems);
            } else {
                memcpy(act_bf16.data(), src1->data, a_elems * sizeof(uint16_t));
            }
            act_scale = xdna_quantize_bf16_to_int8(
                act_bf16.data(), (int8_t *)entry->a_bo->map<void*>(), a_elems);
        } else {
            if (src1->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)src1->data, (uint16_t *)entry->a_bo->map<void*>(), a_elems);
            } else {
                memcpy(entry->a_bo->map<void*>(), src1->data, a_bytes);
            }
        }
        entry->a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Get or build the cached B (weight). Weight data is immutable after
        // model load, so the conversion + DMA sync happens once per (kernel, weight_ptr) pair.
        // [INT8 GEMM] Q8_0 weights → repack to per-tensor int8 in [K,N] layout.
        // Bf16 path: transpose + bf16 convert as before.
        xrt::bo * b_bo_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->b_bo_mutex);
            auto it = entry->b_bo_cache.find(src0->data);
            if (it == entry->b_bo_cache.end()) {
                xrt::bo new_b(ctx->device, b_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(4));

                if (use_int8) {
                    // [INT8 GEMM] Repack Q8_0 → per-tensor int8 [K, N] row-major.
                    // xdna_repack_q8_0_to_gemm_int8 transposes ggml's (N, K) to (K, N)
                    // and returns the per-tensor scale factor.
                    float wt_scale = xdna_repack_q8_0_to_gemm_int8(
                        (const uint8_t *)src0->data, N, K,
                        (int8_t *)new_b.map<void*>());
                    // Persist weight scale for future dequantization.
                    s_wt_scale_cache[src0->data] = wt_scale;
                    fprintf(stderr, "ggml-xdna: INT8 GEMM weight scale=%.6f K=%lld N=%lld\n",
                            wt_scale, (long long)K, (long long)N);
                } else if (src0->type == GGML_TYPE_F32) {
                    const float * src0_f32 = (const float *)src0->data;
                    uint16_t * b_ptr = (uint16_t *)new_b.map<void*>();
                    for (int64_t k = 0; k < K; k++) {
                        for (int64_t n = 0; n < N; n++) {
                            float val = src0_f32[n * K + k];
                            uint32_t bits;
                            memcpy(&bits, &val, sizeof(bits));
                            bits += (0x7FFF + ((bits >> 16) & 1));
                            b_ptr[k * N + n] = (uint16_t)(bits >> 16);
                        }
                    }
                } else {
                    const uint16_t * src0_bf16 = (const uint16_t *)src0->data;
                    uint16_t * b_ptr = (uint16_t *)new_b.map<void*>();
                    for (int64_t k = 0; k < K; k++) {
                        for (int64_t n = 0; n < N; n++) {
                            b_ptr[k * N + n] = src0_bf16[n * K + k];
                        }
                    }
                }
                new_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto [ins, _] = entry->b_bo_cache.emplace(src0->data, std::move(new_b));
                b_bo_ptr = &ins->second;
                fprintf(stderr, "ggml-xdna: warm b_bo K=%lld N=%lld dtype=%s weight=%s (%zu cached for this kernel)\n",
                    (long long)K, (long long)N, dtype_in, src0->name, entry->b_bo_cache.size());
                fflush(stderr);
            } else {
                b_bo_ptr = &it->second;
            }
        }

        auto run = entry->kernel(3, entry->insts_bo, (uint32_t)entry->insts.size(),
                                  *entry->a_bo, *b_bo_ptr, *entry->c_bo);
        run.wait();

        entry->c_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        if (use_int8) {
            // [INT8 GEMM] Dequantize i32 accumulators → f32.
            // INT8 GEMM produces: C[i,j] = sum_k(A_int8[i,k] * B_int8[k,j])
            // The true float value is: C_f32[i,j] = C_i32[i,j] * act_scale * wt_scale
            // Convert to f32 for ggml's downstream consumers.
            float wt_scale = 1.0f;
            auto sc_it = s_wt_scale_cache.find(src0->data);
            if (sc_it != s_wt_scale_cache.end()) wt_scale = sc_it->second;
            const int32_t * c_i32 = (const int32_t *)entry->c_bo->map<void*>();
            float * dst_f32 = (float *)dst->data;
            float combined_scale = act_scale * wt_scale;
            for (size_t i = 0; i < c_elems; i++) {
                dst_f32[i] = (float)c_i32[i] * combined_scale;
            }
        } else {
            bf16_to_f32((const uint16_t *)entry->c_bo->map<void*>(), (float *)dst->data, c_elems);
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: XRT dispatch failed (%s)\n", e.what());
    }
}

// GEMV (decode, M==1). Caller must have verified xdna_shape_dispatchable_gemv().
// IRON GEMV arg order is (matrix, vector, output) — different from GEMM's
// (A_activation, B_weight, C_output). Kernel group_ids: matrix=3, vector=4, output=5.
// Matrix layout matches ggml src0 natively ([N,K] row-major), so no transpose.
static void ggml_backend_xdna_mul_mat_gemv(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // weight (matrix), [N,K]
    const struct ggml_tensor * src1 = dst->src[1];  // activation (vector), [K]

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];

    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: gemv called but XRT device invalid\n");
        return;
    }

    int num_cols = ctx->num_cols;
    std::string cache_key = make_cache_key(XDNA_OP_GEMV, 1, K, N, "bf16", num_cols);

    if (!ensure_compiled(ctx, cache_key, XDNA_OP_GEMV, 1, K, N, "bf16", num_cols)) {
        GGML_LOG_ERROR("ggml-xdna: GEMV compile failed for K=%lld N=%lld\n", (long long)K, (long long)N);
        return;
    }

    xdna_kernel_entry * entry = get_or_load_kernel(ctx, cache_key, XDNA_OP_GEMV, 1, K, N);
    if (!entry) return;

    try {
        size_t vec_elems = K;
        size_t mat_elems = N * K;
        size_t out_elems = N;
        size_t vec_bytes = vec_elems * sizeof(uint16_t);
        size_t mat_bytes = mat_elems * sizeof(uint16_t);
        size_t out_bytes = out_elems * sizeof(uint16_t);

        // Lazily allocate persistent vector and output BOs.
        // group_ids: matrix=3, vector=4, output=5 (per IRON arg_spec).
        if (!entry->a_bo) {
            entry->a_bo = std::make_unique<xrt::bo>(
                ctx->device, vec_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(4));
        }
        if (!entry->c_bo) {
            entry->c_bo = std::make_unique<xrt::bo>(
                ctx->device, out_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(5));
        }

        // Load vector activation fresh each call (changes per token).
        if (src1->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1->data, (uint16_t *)entry->a_bo->map<void*>(), vec_elems);
        } else {
            memcpy(entry->a_bo->map<void*>(), src1->data, vec_bytes);
        }
        entry->a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Cached bf16 weight matrix keyed by src0->data. No transpose —
        // IRON GEMV expects [N,K] row-major, which is ggml src0's native layout.
        xrt::bo * mat_bo_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->b_bo_mutex);
            auto it = entry->b_bo_cache.find(src0->data);
            if (it == entry->b_bo_cache.end()) {
                xrt::bo new_mat(ctx->device, mat_bytes, xrt::bo::flags::host_only,
                                entry->kernel.group_id(3));
                if (src0->type == GGML_TYPE_F32) {
                    f32_to_bf16((const float *)src0->data,
                                (uint16_t *)new_mat.map<void*>(), mat_elems);
                } else {
                    memcpy(new_mat.map<void*>(), src0->data, mat_bytes);
                }
                new_mat.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                auto [ins, _] = entry->b_bo_cache.emplace(src0->data, std::move(new_mat));
                mat_bo_ptr = &ins->second;
                fprintf(stderr, "ggml-xdna: warm gemv matrix K=%lld N=%lld weight=%s (%zu cached)\n",
                        (long long)K, (long long)N, src0->name, entry->b_bo_cache.size());
                fflush(stderr);
            } else {
                mat_bo_ptr = &it->second;
            }
        }

        // Arg order: (opcode, insts, insts_size, matrix, vector, output)
        auto run = entry->kernel(3, entry->insts_bo, (uint32_t)entry->insts.size(),
                                  *mat_bo_ptr, *entry->a_bo, *entry->c_bo);
        run.wait();

        entry->c_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->c_bo->map<void*>(), (float *)dst->data, out_elems);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: GEMV XRT dispatch failed (%s)\n", e.what());
    }
}

// Dispatch MUL_MAT to GEMM (M>=32 prefill) or GEMV (M==1 decode) kernel.
static void ggml_backend_xdna_mul_mat(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst) {
    const int64_t M = dst->src[1]->ne[1];
    if (M == 1) {
        ggml_backend_xdna_mul_mat_gemv(ctx, dst);
    } else {
        ggml_backend_xdna_mul_mat_gemm(ctx, dst);
    }
}

// ============================================================================
// Fused SwiGLU FFN — chained xclbin with 4 sub-kernels
// ============================================================================

// Pick the largest tile_m such that IRON GEMM's M % (tile_m * 4) == 0 holds.
// Returns 0 if seq_len < 32 or has no valid tile in {64, 32, 16, 8}.
static int xdna_pick_swiglu_prefill_tile_m(int64_t seq_len) {
    static const int tms[] = {64, 32, 16, 8};
    for (int i = 0; i < 4; i++) {
        if (seq_len % ((int64_t)tms[i] * 4) == 0) return tms[i];
    }
    return 0;
}

// Pick the largest tile_n such that both inner GEMMs tile cleanly:
//   GEMM1 (gate/up): hidden_dim  % (tile_n * cols) == 0
//   GEMM2 (down):    embedding_dim % (tile_n * cols) == 0
// Returns 0 if no candidate in {64, 32, 16, 8} works.
static int xdna_pick_swiglu_prefill_tile_n(int64_t embedding_dim, int64_t hidden_dim,
                                           int num_cols) {
    static const int tns[] = {64, 32, 16, 8};
    for (int i = 0; i < 4; i++) {
        int64_t step = (int64_t)tns[i] * num_cols;
        if (hidden_dim % step == 0 && embedding_dim % step == 0)
            return tns[i];
    }
    return 0;
}

static std::string make_swiglu_cache_key(xdna_op_kind op_kind,
                                         int64_t embedding_dim, int64_t hidden_dim,
                                         int64_t seq_len, int num_cols,
                                         int tile_m, int group_size = 0,
                                         int tile_n = 64) {
    char buf[256];
    if (op_kind == XDNA_OP_SWIGLU_DECODE) {
        snprintf(buf, sizeof(buf), "swiglu_decode_K%lld_N%lld_bf16_%dcol",
                 (long long)embedding_dim, (long long)hidden_dim, num_cols);
    } else if (op_kind == XDNA_OP_SWIGLU_DECODE_INT8) {
        // group_size is baked into the inner gemv_int8 kernel at compile time
        // via -DGROUP_SIZE, so it MUST participate in the key to avoid
        // re-using a stale binary built for a different group layout.
        snprintf(buf, sizeof(buf),
                 "swiglu_decode_int8_K%lld_N%lld_%dcol_g%d",
                 (long long)embedding_dim, (long long)hidden_dim, num_cols, group_size);
    } else if (op_kind == XDNA_OP_SWIGLU_PREFILL_INT8) {
        if (tile_n != 64) {
            snprintf(buf, sizeof(buf), "swiglu_prefill_int8_M%lld_K%lld_N%lld_tm%d_tn%d_%dcol",
                     (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim,
                     tile_m, tile_n, num_cols);
        } else {
            snprintf(buf, sizeof(buf), "swiglu_prefill_int8_M%lld_K%lld_N%lld_tm%d_%dcol",
                     (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim,
                     tile_m, num_cols);
        }
    } else {
        // Include tile_m (and tile_n when non-default) in the key so different
        // prefill configurations get distinct xclbins.
        if (tile_n != 64) {
            snprintf(buf, sizeof(buf), "swiglu_prefill_M%lld_K%lld_N%lld_tm%d_tn%d_bf16_%dcol",
                     (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim,
                     tile_m, tile_n, num_cols);
        } else {
            snprintf(buf, sizeof(buf), "swiglu_prefill_M%lld_K%lld_N%lld_tm%d_bf16_%dcol",
                     (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim,
                     tile_m, num_cols);
        }
    }
    return std::string(buf);
}

// Check that the full SwiGLU bundle (combined.xclbin + insts) is present on disk.
static bool swiglu_bundle_present(const std::string & bundle_dir,
                                  const char * const insts_tags[XDNA_SWIGLU_MAX_SLOTS],
                                  int num_kernels) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    std::string xclbin_path = bundle_dir + "\\combined.xclbin";
    if (dbg) fprintf(stderr, "ggml-xdna: checking swiglu xclbin at %s\n", xclbin_path.c_str());
    std::ifstream xf(xclbin_path);
    if (!xf.good()) return false;
    for (int s = 0; s < num_kernels; s++) {
        std::string inst_path = bundle_dir + "\\swiglu_" + insts_tags[s] + ".insts";
        if (dbg) fprintf(stderr, "ggml-xdna: checking swiglu insts at %s\n", inst_path.c_str());
        std::ifstream f(inst_path);
        if (!f.good()) return false;
    }
    return true;
}

static bool ensure_swiglu_compiled(ggml_backend_xdna_context * ctx,
                                   const std::string & cache_key,
                                   xdna_op_kind op_kind,
                                   int64_t embedding_dim, int64_t hidden_dim,
                                   int64_t seq_len, int num_cols,
                                   int tile_m, int group_size = 0,
                                   int tile_n = 64) {
    // Skip recompilation if we already failed for this key
    if (ctx->swiglu_compile_failed.count(cache_key)) {
        return false;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    const char * const * insts_tags;
    int num_kernels;
    switch (op_kind) {
        case XDNA_OP_SWIGLU_DECODE:
            insts_tags = XDNA_SWIGLU_DECODE_INSTS_TAGS;
            num_kernels = XDNA_SWIGLU_DECODE_NUM_KERNELS;
            break;
        default:  // XDNA_OP_SWIGLU_PREFILL
            insts_tags = XDNA_SWIGLU_PREFILL_INSTS_TAGS;
            num_kernels = XDNA_SWIGLU_PREFILL_NUM_KERNELS;
            break;
    }

    if (swiglu_bundle_present(bundle_dir, insts_tags, num_kernels)) {
        return true;
    }

    char cmd[1024];
    if (op_kind == XDNA_OP_SWIGLU_DECODE) {
        snprintf(cmd, sizeof(cmd),
                 "%s \"%s\" --quiet swiglu-decode --embedding-dim %lld --hidden-dim %lld "
                 "--num-aie-columns %d --out \"%s\"%s",
                 xdna_python_cmd(), ctx->compile_script.c_str(),
                 (long long)embedding_dim, (long long)hidden_dim,
                 num_cols, bundle_dir.c_str(), xdna_null_redirect());
        fprintf(stderr, "ggml-xdna: compiling SwiGLU decode K=%lld N=%lld (first run, will be cached)...\n",
                      (long long)embedding_dim, (long long)hidden_dim);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "%s \"%s\" --quiet swiglu-prefill --seq-len %lld --embedding-dim %lld --hidden-dim %lld "
                 "--num-aie-columns %d --tile-m %d --tile-n %d --out \"%s\"%s",
                 xdna_python_cmd(), ctx->compile_script.c_str(),
                 (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim,
                 num_cols, tile_m, tile_n, bundle_dir.c_str(), xdna_null_redirect());
        fprintf(stderr, "ggml-xdna: compiling SwiGLU prefill M=%lld K=%lld N=%lld tile_m=%d tile_n=%d (first run, will be cached)...\n",
                      (long long)seq_len, (long long)embedding_dim, (long long)hidden_dim, tile_m, tile_n);
    }

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU compilation failed (exit code %d)\n", ret);
        ctx->swiglu_compile_failed.insert(cache_key);
        return false;
    }

    if (!swiglu_bundle_present(bundle_dir, insts_tags, num_kernels)) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU compilation succeeded but bundle files missing in %s\n",
                       bundle_dir.c_str());
        ctx->swiglu_compile_failed.insert(cache_key);
        return false;
    }

    fprintf(stderr, "ggml-xdna: SwiGLU compilation complete, cached at %s\n", bundle_dir.c_str());
    return true;
}

// Prime the hw_context by firing one dummy submit of each sub-kernel right
// after the xclbin is loaded. Agent investigation traced the ~1800 µs first-
// submit cold-start to the XDNA driver's ctx-connect slow path (DRM scheduler
// init + aie2_config_cu mailbox + HMM range walk of BOs). Paying that here
// once per kernel-entry lets real dispatches return in ~20 µs.
//
// Uses scratch BOs sized like the real dispatch for each slot so every BO
// goes through its actual group_id binding. BOs fall out of scope at return
// and are freed — we only need to exercise the first-submit paths.
static void swiglu_warmup_entry(ggml_backend_xdna_context * ctx,
                                xdna_swiglu_kernel_entry * entry) {
    using clk = std::chrono::steady_clock;
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    const auto t_start = clk::now();

    const int64_t embedding_dim = entry->embedding_dim;
    const int64_t hidden_dim    = entry->hidden_dim;
    const int64_t M             = entry->seq_len;  // 1 for decode, seq_len for prefill
    const bool prefill          = (entry->op_kind == XDNA_OP_SWIGLU_PREFILL);

    const size_t input_bytes  = (size_t)embedding_dim * (size_t)M * sizeof(uint16_t);
    const size_t hidden_bytes = (size_t)hidden_dim    * (size_t)M * sizeof(uint16_t);
    const size_t weight_bytes = (size_t)embedding_dim * (size_t)hidden_dim * sizeof(uint16_t);

    // Group IDs match the real dispatcher's persistent-BO allocation pattern.
    const int mm1_in_grp = prefill ? 3 : 4;  // GEMM A vs GEMV vector
    const int mm1_w_grp  = prefill ? 4 : 3;  // GEMM B vs GEMV matrix
    const int mm2_w_grp  = prefill ? 4 : 3;

    try {
        if (prefill) {
            // Prefill: 3 kernels (gate_up, silu_mul, down)
            xrt::bo sc_input(ctx->device, input_bytes,  xrt::bo::flags::host_only,
                             entry->kernels[XDNA_SWIGLU_GATE_UP].group_id(mm1_in_grp));
            xrt::bo sc_output(ctx->device, input_bytes, xrt::bo::flags::host_only,
                              entry->kernels[XDNA_SWIGLU_DOWN_P].group_id(5));
            xrt::bo sc_left(ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                            entry->kernels[XDNA_SWIGLU_GATE_UP].group_id(5));
            xrt::bo sc_right(ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                             entry->kernels[XDNA_SWIGLU_GATE_UP].group_id(5));
            xrt::bo sc_intermediate(ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                                    entry->kernels[XDNA_SWIGLU_SILU_MUL].group_id(5));
            xrt::bo sc_w1(ctx->device, weight_bytes, xrt::bo::flags::host_only,
                          entry->kernels[XDNA_SWIGLU_GATE_UP].group_id(mm1_w_grp));
            xrt::bo sc_w2(ctx->device, weight_bytes, xrt::bo::flags::host_only,
                          entry->kernels[XDNA_SWIGLU_GATE_UP].group_id(mm1_w_grp));
            xrt::bo sc_w3(ctx->device, weight_bytes, xrt::bo::flags::host_only,
                          entry->kernels[XDNA_SWIGLU_DOWN_P].group_id(mm2_w_grp));

            memset(sc_input.map<void*>(),  0, input_bytes);
            memset(sc_output.map<void*>(), 0, input_bytes);
            memset(sc_left.map<void*>(),   0, hidden_bytes);
            memset(sc_right.map<void*>(),  0, hidden_bytes);
            memset(sc_intermediate.map<void*>(), 0, hidden_bytes);
            memset(sc_w1.map<void*>(), 0, weight_bytes);
            memset(sc_w2.map<void*>(), 0, weight_bytes);
            memset(sc_w3.map<void*>(), 0, weight_bytes);
            sc_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sc_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sc_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sc_w3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Prefill warmup: 4 dispatches (gate GEMM, up GEMM, silu_mul, down GEMM)
            const uint32_t gu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_GATE_UP].size();
            const uint32_t sm_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_SILU_MUL].size();
            const uint32_t dn_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_DOWN_P].size();
            xrt::run r;

            r = entry->kernels[XDNA_SWIGLU_GATE_UP](
                3, entry->insts_bo[XDNA_SWIGLU_GATE_UP], gu_isize,
                sc_input, sc_w1, sc_left);
            r.wait();

            r = entry->kernels[XDNA_SWIGLU_GATE_UP](
                3, entry->insts_bo[XDNA_SWIGLU_GATE_UP], gu_isize,
                sc_input, sc_w2, sc_right);
            r.wait();

            r = entry->kernels[XDNA_SWIGLU_SILU_MUL](
                3, entry->insts_bo[XDNA_SWIGLU_SILU_MUL], sm_isize,
                sc_left, sc_right, sc_intermediate);
            r.wait();

            r = entry->kernels[XDNA_SWIGLU_DOWN_P](
                3, entry->insts_bo[XDNA_SWIGLU_DOWN_P], dn_isize,
                sc_intermediate, sc_w3, sc_output);
            r.wait();

        } else {
            // Decode: 2 kernels (fused gate+up+silu+mul, down GEMV)
            // Fused weights are interleaved: size = 2 * embedding_dim * hidden_dim
            const size_t fused_weight_bytes = 2 * weight_bytes;
            xrt::bo sc_input(ctx->device, input_bytes,  xrt::bo::flags::host_only,
                             entry->kernels[XDNA_SWIGLU_FUSED].group_id(4));
            xrt::bo sc_output(ctx->device, input_bytes, xrt::bo::flags::host_only,
                              entry->kernels[XDNA_SWIGLU_DOWN].group_id(5));
            xrt::bo sc_intermediate(ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                                    entry->kernels[XDNA_SWIGLU_FUSED].group_id(5));
            xrt::bo sc_w_fused(ctx->device, fused_weight_bytes, xrt::bo::flags::host_only,
                               entry->kernels[XDNA_SWIGLU_FUSED].group_id(3));
            xrt::bo sc_w_down(ctx->device, weight_bytes, xrt::bo::flags::host_only,
                              entry->kernels[XDNA_SWIGLU_DOWN].group_id(3));

            memset(sc_input.map<void*>(),       0, input_bytes);
            memset(sc_output.map<void*>(),      0, input_bytes);
            memset(sc_intermediate.map<void*>(), 0, hidden_bytes);
            memset(sc_w_fused.map<void*>(), 0, fused_weight_bytes);
            memset(sc_w_down.map<void*>(),  0, weight_bytes);
            sc_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sc_w_fused.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            sc_w_down.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            const uint32_t fu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_FUSED].size();
            const uint32_t dn_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_DOWN].size();
            xrt::run r;

            r = entry->kernels[XDNA_SWIGLU_FUSED](
                3, entry->insts_bo[XDNA_SWIGLU_FUSED], fu_isize,
                sc_w_fused, sc_input, sc_intermediate);
            r.wait();

            r = entry->kernels[XDNA_SWIGLU_DOWN](
                3, entry->insts_bo[XDNA_SWIGLU_DOWN], dn_isize,
                sc_w_down, sc_intermediate, sc_output);
            r.wait();
        }

        if (dbg) {
            auto t_end = clk::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            fprintf(stderr, "ggml-xdna: swiglu warmup %s M=%lld emb=%lld hid=%lld took %lld us\n",
                          prefill ? "prefill" : "decode",
                          (long long)M, (long long)embedding_dim, (long long)hidden_dim, (long long)us);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "ggml-xdna: warning: SwiGLU warmup failed (%s) — first dispatch will pay the cost\n",
                      e.what());
    }
}

static xdna_swiglu_kernel_entry * get_or_load_swiglu_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        xdna_op_kind op_kind,
        int64_t embedding_dim, int64_t hidden_dim,
        int64_t seq_len, int num_cols,
        int group_size = 0,
        int mm1_m_input = 0, int mm2_m_input = 0) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);

    auto it = ctx->swiglu_cache.find(cache_key);
    if (it != ctx->swiglu_cache.end()) {
        return &it->second;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    const char * const * kernel_names;
    const char * const * insts_tags;
    int num_kernels;
    switch (op_kind) {
        case XDNA_OP_SWIGLU_DECODE:
            kernel_names = XDNA_SWIGLU_DECODE_KERNEL_NAMES;
            insts_tags   = XDNA_SWIGLU_DECODE_INSTS_TAGS;
            num_kernels  = XDNA_SWIGLU_DECODE_NUM_KERNELS;
            break;
        default:  // XDNA_OP_SWIGLU_PREFILL
            kernel_names = XDNA_SWIGLU_PREFILL_KERNEL_NAMES;
            insts_tags   = XDNA_SWIGLU_PREFILL_INSTS_TAGS;
            num_kernels  = XDNA_SWIGLU_PREFILL_NUM_KERNELS;
            break;
    }

    // Try loading the cached xclbin.  If it fails (stale cache, wrong
    // kernel name, corrupted xclbin, etc.), delete the bundle so the
    // next dispatch cycle re-enters ensure_swiglu_compiled +
    // get_or_load_swiglu_kernel with fresh binaries.
    try {
        xdna_swiglu_kernel_entry entry;
        entry.op_kind       = op_kind;
        entry.embedding_dim = embedding_dim;
        entry.hidden_dim    = hidden_dim;
        entry.seq_len       = seq_len;
        entry.num_cols      = num_cols;
        entry.group_size    = group_size;
        entry.mm1_m_input   = mm1_m_input;
        entry.mm2_m_input   = mm2_m_input;
        entry.num_kernels   = num_kernels;

        entry.xclbin = xrt::xclbin(bundle_dir + "/combined.xclbin");
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);

        for (int s = 0; s < num_kernels; s++) {
            entry.kernels[s] = xrt::kernel(entry.hw_ctx, kernel_names[s]);

            const std::string insts_path = bundle_dir + "/swiglu_" + insts_tags[s] + ".insts";
            entry.insts_data[s] = read_binary_file(insts_path);
            if (entry.insts_data[s].empty()) {
                GGML_LOG_ERROR("ggml-xdna: failed to read SwiGLU insts file: %s\n",
                               insts_path.c_str());
                return nullptr;
            }
            entry.insts_bo[s] = xrt::bo(ctx->device, entry.insts_data[s].size(),
                                         xrt::bo::flags::cacheable,
                                         entry.kernels[s].group_id(1));
            entry.insts_bo[s].write(entry.insts_data[s].data());
            entry.insts_bo[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        auto [inserted_it, _] = ctx->swiglu_cache.emplace(cache_key, std::move(entry));
        xdna_swiglu_kernel_entry * entry_ptr = &inserted_it->second;
        fprintf(stderr, "ggml-xdna: loaded SwiGLU kernel bundle for %s (%d kernels)\n",
                      cache_key.c_str(), num_kernels);

        // Pay the first-submit slow-path cost (ctx-connect + HMM pin) once
        // here instead of on the first real FFN dispatch. Non-fatal on failure.
        swiglu_warmup_entry(ctx, entry_ptr);

        return entry_ptr;

    } catch (const std::exception & e) {
        // Stale xclbin (wrong kernel name, corrupted, etc.).
        // Delete the bundle so the next dispatch cycle will
        // re-enter ensure_swiglu_compiled + get_or_load_swiglu_kernel
        // with the correct parameters from the caller.
        fprintf(stderr, "ggml-xdna: SwiGLU cache stale for %s (%s), "
                      "invalidating — will recompile on next dispatch\n",
                      cache_key.c_str(), e.what());
#ifdef _WIN32
        std::string rm_cmd = "rd /s /q \"" + bundle_dir + "\" 2>nul";
#else
        std::string rm_cmd = "rm -rf \"" + bundle_dir + "\"";
#endif
        system(rm_cmd.c_str());
        return nullptr;
    }
}

// Warm a single weight slot: copy the bf16/f32 weight into the appropriate BO
// (transposing for prefill, straight copy for decode) and sync to device.
// Must be called under entry->weights_mutex.
// Warm (or look up) a SwiGLU weight BO. Caller must hold entry->weights_mutex.
// Returns nullptr on error. Cache lifetime is the kernel entry's; one entry
// per (layer-weight-tensor) lives forever for the session.
static xrt::bo * swiglu_warm_weight(ggml_backend_xdna_context * ctx,
                                    xdna_swiglu_kernel_entry * entry,
                                    std::unordered_map<const void *, xrt::bo> & cache,
                                    const struct ggml_tensor * weight,
                                    int kernel_slot,      // which sub-kernel consumes this weight
                                    int arg_group_id,     // group_id within that kernel
                                    const char * slot_name) {
    auto it = cache.find(weight->data);
    if (it != cache.end()) {
        return &it->second;
    }

    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];
    const size_t  n_elems = (size_t)K * (size_t)N;
    const size_t  n_bytes = n_elems * sizeof(uint16_t);

    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[kernel_slot].group_id(arg_group_id));

        uint16_t * dst_bf16 = (uint16_t *)new_bo.map<void*>();

        if (entry->op_kind == XDNA_OP_SWIGLU_PREFILL) {
            // GEMM path: transpose [N,K] → [K,N] row-major (matches the
            // standalone GEMM dispatcher's B-buffer layout).
            if (weight->type == GGML_TYPE_F32) {
                const float * src_f32 = (const float *)weight->data;
                for (int64_t k = 0; k < K; k++) {
                    for (int64_t n = 0; n < N; n++) {
                        float val = src_f32[n * K + k];
                        uint32_t bits;
                        memcpy(&bits, &val, sizeof(bits));
                        bits += (0x7FFF + ((bits >> 16) & 1));
                        dst_bf16[k * N + n] = (uint16_t)(bits >> 16);
                    }
                }
            } else if (weight->type == GGML_TYPE_F16) {
                const uint16_t * src_f16 = (const uint16_t *)weight->data;
                for (int64_t k = 0; k < K; k++) {
                    for (int64_t n = 0; n < N; n++) {
                        dst_bf16[k * N + n] = fp16_to_bf16(src_f16[n * K + k]);
                    }
                }
            } else {
                // BF16: direct transpose, no conversion needed.
                const uint16_t * src_bf16 = (const uint16_t *)weight->data;
                for (int64_t k = 0; k < K; k++) {
                    for (int64_t n = 0; n < N; n++) {
                        dst_bf16[k * N + n] = src_bf16[n * K + k];
                    }
                }
            }
        } else {
            // GEMV path: IRON GEMV expects [N,K] row-major — ggml's native layout.
            if (weight->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)weight->data, dst_bf16, n_elems);
            } else if (weight->type == GGML_TYPE_F16) {
                const uint16_t * src_f16 = (const uint16_t *)weight->data;
                for (int64_t i = 0; i < n_elems; i++) {
                    dst_bf16[i] = fp16_to_bf16(src_f16[i]);
                }
            } else {
                memcpy(dst_bf16, weight->data, n_bytes);
            }
        }

        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(weight->data, std::move(new_bo));

        fprintf(stderr, "ggml-xdna: warm SwiGLU %s K=%lld N=%lld weight=%s (%zu cached)\n",
                slot_name, (long long)K, (long long)N, weight->name, cache.size());
        fflush(stderr);
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm SwiGLU %s weight: %s\n", slot_name, e.what());
        return nullptr;
    }
}

// Forward decls for shape-dispatchability checks.
static bool xdna_shape_dispatchable_swiglu_decode(int64_t K_emb, int64_t N_hid, int num_cols);
static bool xdna_shape_dispatchable_swiglu_prefill(int64_t M_seq, int64_t K_emb, int64_t N_hid, int num_cols);

// Dispatch a fused SwiGLU FFN: gate/up MUL_MATs + GLU + down MUL_MAT.
// Args are taken from the matched pattern (see xdna_try_match_swiglu).
// src0_*_w are weight tensors, src1_input is the shared FFN input activation.
// On success, writes gate_dst, up_dst, glu_dst, and dst_final's ->data buffers
// (all four, since downstream consumers may read any of them).
static void ggml_backend_xdna_mul_mat_swiglu(ggml_backend_xdna_context * ctx,
                                             struct ggml_tensor * gate_dst,
                                             struct ggml_tensor * up_dst,
                                             struct ggml_tensor * glu_dst,
                                             struct ggml_tensor * dst_final,
                                             const struct ggml_tensor * src0_gate_w,
                                             const struct ggml_tensor * src0_up_w,
                                             const struct ggml_tensor * src0_down_w,
                                             const struct ggml_tensor * src1_input,
                                             int num_cols) {
    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU called but XRT device invalid\n");
        return;
    }

    const int64_t M             = src1_input->ne[1];
    const int64_t embedding_dim = src1_input->ne[0];
    const int64_t hidden_dim    = src0_gate_w->ne[1];

    const xdna_op_kind op_kind = (M == 1) ? XDNA_OP_SWIGLU_DECODE : XDNA_OP_SWIGLU_PREFILL;

    // Round M up to the next multiple of 64 for GEMM tiling. NPU2 GEMM
    // kernel with emulate_bf16_mmul_with_bfp16 uses 8x8x8 MAC → tile_m >= 16
    // → M must be a multiple of tile_m * 4 = 64. Input is zero-padded;
    // output truncated to actual M rows.
    const int64_t padded_M     = (op_kind == XDNA_OP_SWIGLU_PREFILL)
        ? ((M + 63) / 64) * 64 : M;
    const int64_t seq_len      = (op_kind == XDNA_OP_SWIGLU_DECODE) ? 1 : padded_M;

    // tile_m/tile_n only apply to the prefill inner GEMMs; decode (GEMV-based) ignores them.
    // The pattern matcher rejected M if no tile_m exists, so a 0 here would be a bug.
    const int tile_m = (op_kind == XDNA_OP_SWIGLU_PREFILL)
        ? xdna_pick_swiglu_prefill_tile_m(seq_len) : 0;
    const int tile_n = (op_kind == XDNA_OP_SWIGLU_PREFILL)
        ? xdna_pick_swiglu_prefill_tile_n(embedding_dim, hidden_dim, num_cols) : 64;

    std::string cache_key = make_swiglu_cache_key(op_kind, embedding_dim, hidden_dim,
                                                  seq_len, num_cols, tile_m,
                                                  /*group_size=*/0, tile_n);

    if (!ensure_swiglu_compiled(ctx, cache_key, op_kind,
                                embedding_dim, hidden_dim, seq_len, num_cols, tile_m,
                                /*group_size=*/0, tile_n)) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU compile failed for %s\n", cache_key.c_str());
        return;
    }

    xdna_swiglu_kernel_entry * entry = get_or_load_swiglu_kernel(
        ctx, cache_key, op_kind, embedding_dim, hidden_dim, seq_len, num_cols);
    if (!entry) return;

    try {
        const bool prefill = (op_kind == XDNA_OP_SWIGLU_PREFILL);

        // BO sizes use padded_M (for NPU computation); data copies use actual M.
        const size_t padded_input_elems  = (size_t)embedding_dim * (size_t)padded_M;
        const size_t padded_output_elems = (size_t)embedding_dim * (size_t)padded_M;
        const size_t padded_hidden_elems = (size_t)hidden_dim    * (size_t)padded_M;
        const size_t input_bytes  = padded_input_elems  * sizeof(uint16_t);
        const size_t output_bytes = padded_output_elems * sizeof(uint16_t);
        const size_t hidden_bytes = padded_hidden_elems * sizeof(uint16_t);
        // Actual (unpadded) element counts for host↔device data copies.
        const size_t actual_input_elems  = (size_t)embedding_dim * (size_t)M;
        const size_t actual_output_elems = (size_t)embedding_dim * (size_t)M;
        const size_t actual_hidden_elems = (size_t)hidden_dim    * (size_t)M;

        // Lazily allocate persistent per-call BOs on first dispatch.
        // Group IDs follow the IRON arg_specs:
        //   GEMV:  (matrix=3, vector=4, output=5)
        //   GEMM:  (A=3,      B=4,      C=5)
        //   DualGEMVSiLUMul(fused): (weights_gate_up=3, input=4, output=5)
        if (!entry->input_bo) {
            const int grp = prefill ? 3 : 4;  // GEMM A / GEMV vector
            entry->input_bo = std::make_unique<xrt::bo>(
                ctx->device, input_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SLOT_0].group_id(grp));
        }
        if (!entry->output_bo) {
            const int down_slot = prefill ? XDNA_SWIGLU_DOWN_P : XDNA_SWIGLU_DOWN;
            entry->output_bo = std::make_unique<xrt::bo>(
                ctx->device, output_bytes, xrt::bo::flags::host_only,
                entry->kernels[down_slot].group_id(5));
        }
        if (!entry->left_bo) {
            entry->left_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SLOT_0].group_id(5));
        }
        if (!entry->right_bo) {
            entry->right_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SLOT_0].group_id(5));
        }
        if (!entry->intermediate_bo) {
            const int sm_slot = prefill ? XDNA_SWIGLU_SILU_MUL : XDNA_SWIGLU_FUSED;
            entry->intermediate_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[sm_slot].group_id(5));
        }
        // left_swished_bo is unused in the new architecture (SiLU+Mul fused)
        // but kept in the struct for ABI compatibility. Allocate minimal BO.
        if (!entry->left_swished_bo) {
            entry->left_swished_bo = std::make_unique<xrt::bo>(
                ctx->device, 64, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SLOT_0].group_id(5));
        }

        // Warm (or look up) per-layer weight BOs. Each layer's weight tensor
        // gets its own BO; the kernel entry is shared across layers of the
        // same shape so the maps grow as new layers are first hit.
        xrt::bo * w1_bo_ptr = nullptr;
        xrt::bo * w2_bo_ptr = nullptr;
        xrt::bo * w3_bo_ptr = nullptr;
        xrt::bo * w_fused_bo_ptr = nullptr;  // decode-only: interleaved gate+up
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            if (prefill) {
                // Prefill: gate/up are separate GEMM B-buffers (slot GATE_UP),
                // down is a separate GEMM B-buffer (slot DOWN_P).
                w1_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w1_bo_cache,
                                               src0_gate_w, XDNA_SWIGLU_GATE_UP,
                                               /*arg_group_id=*/4, "w_gate");
                w2_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w2_bo_cache,
                                               src0_up_w,   XDNA_SWIGLU_GATE_UP,
                                               /*arg_group_id=*/4, "w_up");
                w3_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w3_bo_cache,
                                               src0_down_w, XDNA_SWIGLU_DOWN_P,
                                               /*arg_group_id=*/4, "w_down");
                if (!w1_bo_ptr || !w2_bo_ptr || !w3_bo_ptr) return;
            } else {
                // Decode: load gate and up individually (slot FUSED, group_id 3
                // = matrix for GEMV), then interleave into a combined fused BO.
                // Down is loaded normally (slot DOWN, group_id 3).
                w1_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w1_bo_cache,
                                               src0_gate_w, XDNA_SWIGLU_FUSED,
                                               /*arg_group_id=*/3, "w_gate");
                w2_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w2_bo_cache,
                                               src0_up_w,   XDNA_SWIGLU_FUSED,
                                               /*arg_group_id=*/3, "w_up");
                w3_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w3_bo_cache,
                                               src0_down_w, XDNA_SWIGLU_DOWN,
                                               /*arg_group_id=*/3, "w_down");
                if (!w1_bo_ptr || !w2_bo_ptr || !w3_bo_ptr) return;

                // Build or look up the interleaved gate+up fused weight BO.
                // Cache key: combine the two source data pointers into a hash.
                {
                    auto h1 = std::hash<const void*>{}(src0_gate_w->data);
                    auto h2 = std::hash<const void*>{}(src0_up_w->data);
                    uint64_t fused_key = h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));

                    auto it = entry->w_fused_bo_cache.find(fused_key);
                    if (it != entry->w_fused_bo_cache.end()) {
                        w_fused_bo_ptr = &it->second;
                    } else {
                        // Per-column block layout: [gate_col0, up_col0, gate_col1, up_col1, ...]
                        // Each block is rows_per_col contiguous rows of embedding_dim bf16 values.
                        const size_t fused_elems = 2 * (size_t)hidden_dim * (size_t)embedding_dim;
                        const size_t fused_bytes = fused_elems * sizeof(uint16_t);

                        xrt::bo fused_bo(ctx->device, fused_bytes, xrt::bo::flags::host_only,
                                         entry->kernels[XDNA_SWIGLU_FUSED].group_id(3));

                        const uint16_t * w_gate = (const uint16_t *)w1_bo_ptr->map<const void*>();
                        const uint16_t * w_up   = (const uint16_t *)w2_bo_ptr->map<const void*>();
                        uint16_t * dst = (uint16_t *)fused_bo.map<void*>();

                        // Per-column block interleave — must match the Python
                        // interleave_weights() layout that the IRON-compiled
                        // xclbin's DMA pattern expects.
                        // Fused kernel always uses 4 columns (dual-GEMV DMA
                        // constraint), regardless of the device column count.
                        // Layout: [gate_col0, up_col0, gate_col1, up_col1, ...]
                        // Each "block" is rows_per_col contiguous rows of
                        // embedding_dim bf16 values.
                        {
                            const int fused_cols = 4;
                            GGML_ASSERT(hidden_dim % fused_cols == 0 && "hidden_dim must be divisible by 4 for fused SwiGLU interleave");
                            const int64_t rows_per_col = hidden_dim / fused_cols;
                            for (int col = 0; col < fused_cols; col++) {
                                const int64_t start = col * rows_per_col;
                                const int64_t out_start = col * 2 * rows_per_col;
                                memcpy(dst + out_start * embedding_dim,
                                       w_gate + start * embedding_dim,
                                       rows_per_col * embedding_dim * sizeof(uint16_t));
                                memcpy(dst + (out_start + rows_per_col) * embedding_dim,
                                       w_up + start * embedding_dim,
                                       rows_per_col * embedding_dim * sizeof(uint16_t));
                            }
                        }

                        fused_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        auto [ins, ok] = entry->w_fused_bo_cache.emplace(fused_key, std::move(fused_bo));
                        w_fused_bo_ptr = &ins->second;

                        fprintf(stderr, "ggml-xdna: warm SwiGLU fused gate+up K=%lld N=%lld (%zu cached)\n",
                                (long long)embedding_dim, (long long)hidden_dim,
                                entry->w_fused_bo_cache.size());
                        fflush(stderr);
                    }
                }
            }
        }

        // Profiling: split each phase into submit vs wait so we can tell
        // whether per-call overhead is host-submit cost or on-device
        // execution + host stall. Also times host<->device DMA/convert
        // phases. Prints one compact line per dispatch under XDNA_DEBUG.
        using clk = std::chrono::steady_clock;
        static const bool prof = getenv("XDNA_DEBUG") != NULL;
        auto t_in_s  = clk::now();

        // Load input activation (freshly computed per token/step).
        // When M < padded_M (speculative decoding), zero-pad the extra rows.
        if (padded_M > M) {
            memset(entry->input_bo->map<void*>(), 0, input_bytes);
        }
        if (src1_input->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1_input->data,
                        (uint16_t *)entry->input_bo->map<void*>(), actual_input_elems);
        } else {
            memcpy(entry->input_bo->map<void*>(), src1_input->data,
                   actual_input_elems * sizeof(uint16_t));
        }
        entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t_in_e  = clk::now();

        // Build the xrt::runlist. Decode uses 2 runs (fused + down),
        // prefill uses 4 runs (gate GEMM + up GEMM + silu_mul + down GEMM).
        // NOTE: must build runs via explicit ctor + set_arg + rl.add. The
        // kernel functor `k(args...)` implicitly calls run.start(), which is
        // UB on a run that is subsequently added to a runlist.

        auto t_build_s = clk::now();
        xrt::runlist rl(entry->hw_ctx);

        if (prefill) {
            // 4-run prefill: gate GEMM + up GEMM + silu_mul + down GEMM
            const uint32_t gu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_GATE_UP].size();
            const uint32_t sm_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_SILU_MUL].size();
            const uint32_t dn_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_DOWN_P].size();

            // Run 1: gate GEMM (input @ W1.T -> left)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_GATE_UP]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_GATE_UP]);
                r.set_arg(2, gu_isize);
                r.set_arg(3, *entry->input_bo);
                r.set_arg(4, *w1_bo_ptr);
                r.set_arg(5, *entry->left_bo);
                rl.add(r);
            }

            // Run 2: up GEMM (input @ W2.T -> right)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_GATE_UP]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_GATE_UP]);
                r.set_arg(2, gu_isize);
                r.set_arg(3, *entry->input_bo);
                r.set_arg(4, *w2_bo_ptr);
                r.set_arg(5, *entry->right_bo);
                rl.add(r);
            }

            // Run 3: fused SiLU+Mul (left, right -> intermediate)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_SILU_MUL]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_SILU_MUL]);
                r.set_arg(2, sm_isize);
                r.set_arg(3, *entry->left_bo);
                r.set_arg(4, *entry->right_bo);
                r.set_arg(5, *entry->intermediate_bo);
                rl.add(r);
            }

            // Run 4: down GEMM (intermediate @ W3.T -> output)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_DOWN_P]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_DOWN_P]);
                r.set_arg(2, dn_isize);
                r.set_arg(3, *entry->intermediate_bo);
                r.set_arg(4, *w3_bo_ptr);
                r.set_arg(5, *entry->output_bo);
                rl.add(r);
            }
        } else {
            // 2-run decode: fused(gate+up+silu+mul) + down
            const uint32_t fu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_FUSED].size();
            const uint32_t dn_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_DOWN].size();

            // Run 1: fused gate+up+silu+mul
            // DualGEMVSiLUMul args: (opcode, insts, isize, weights_gate_up, input, output)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_FUSED]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_FUSED]);
                r.set_arg(2, fu_isize);
                r.set_arg(3, *w_fused_bo_ptr);  // interleaved gate+up weights
                r.set_arg(4, *entry->input_bo);
                r.set_arg(5, *entry->intermediate_bo);
                rl.add(r);
            }

            // Run 2: down projection GEMV
            // Args: (opcode, insts, isize, weights_down, intermediate, output)
            {
                xrt::run r(entry->kernels[XDNA_SWIGLU_DOWN]);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_DOWN]);
                r.set_arg(2, dn_isize);
                r.set_arg(3, *w3_bo_ptr);
                r.set_arg(4, *entry->intermediate_bo);
                r.set_arg(5, *entry->output_bo);
                rl.add(r);
            }
        }
        auto t_build_e = clk::now();

        rl.execute();
        auto t_exec_e = clk::now();

        rl.wait();
        auto t_wait_e = clk::now();

        // Pull the final output back to host (the only result downstream actually needs).
        // When padded, only copy back the first M rows (actual_*_elems).
        auto t_out_s = clk::now();
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->output_bo->map<void*>(),
                    (float *)dst_final->data, actual_output_elems);
        auto t_out_e = clk::now();

        // Prefill: also read back left/right/intermediate so downstream
        // consumers that inspected the old separate buffers still work.
        // Decode: intermediate_bo is internal to the fused kernel — skip.
        long long wb_l_us = 0, wb_r_us = 0, wb_i_us = 0;
        if (prefill) {
            auto t_wbl_s = clk::now();
            entry->left_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            bf16_to_f32((const uint16_t *)entry->left_bo->map<void*>(),
                        (float *)gate_dst->data, actual_hidden_elems);
            wb_l_us = (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                clk::now() - t_wbl_s).count();

            auto t_wbr_s = clk::now();
            entry->right_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            bf16_to_f32((const uint16_t *)entry->right_bo->map<void*>(),
                        (float *)up_dst->data, actual_hidden_elems);
            wb_r_us = (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                clk::now() - t_wbr_s).count();

            auto t_wbi_s = clk::now();
            entry->intermediate_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            bf16_to_f32((const uint16_t *)entry->intermediate_bo->map<void*>(),
                        (float *)glu_dst->data, actual_hidden_elems);
            wb_i_us = (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                clk::now() - t_wbi_s).count();
        }

        if (prof) {
            auto us = [](clk::time_point a, clk::time_point b) {
                return (long long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                "ggml-xdna: swiglu_prof %s M=%lld K=%lld N=%lld in=%lld "
                "rl_build=%lld rl_exec=%lld rl_wait=%lld "
                "out=%lld wb_l=%lld wb_r=%lld wb_i=%lld total=%lld us\n",
                prefill ? "prefill" : "decode",
                (long long)M, (long long)embedding_dim, (long long)hidden_dim,
                us(t_in_s,    t_in_e),
                us(t_build_s, t_build_e),
                us(t_build_e, t_exec_e),
                us(t_exec_e,  t_wait_e),
                us(t_out_s,   t_out_e),
                wb_l_us, wb_r_us, wb_i_us,
                us(t_in_s,    clk::now()));
        }

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU XRT dispatch failed (%s)\n", e.what());
    }
}

// [DEAD CODE — IRON-windows does not compile INT8 kernels.]
// Wrapped in #if 0 to avoid compiling unreachable code.
#if 0
// Warm (or look up) an INT8 SwiGLU weight BO: repack Q8_0 weights into the
// gemv_int8 per-tile DDR layout and upload. Caller holds entry->weights_mutex.
// Returns nullptr on error. The map-key is the ggml weight tensor's data ptr,
// so multi-layer models allocate one BO per layer's Q8_0 blob.
static xrt::bo * swiglu_warm_weight_int8(ggml_backend_xdna_context * ctx,
                                         xdna_swiglu_kernel_entry * entry,
                                         std::unordered_map<const void *, xrt::bo> & cache,
                                         const struct ggml_tensor * weight,
                                         int kernel_slot,    // which sub-kernel consumes this weight
                                         int arg_group_id,   // group_id within that kernel
                                         int m_input,        // clamped tile_size_input for this stage
                                         const char * slot_name) {
    auto it = cache.find(weight->data);
    if (it != cache.end()) {
        return &it->second;
    }

    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];
    const int group_size = entry->group_size;
    const int cols = entry->num_cols;

    // Packed buffer size mirrors GEMVInt8._packed_buffer_size:
    //   per tile: m_input*K (int8) + m_input*(K/group_size)*2 (bf16 scales)
    //   total:    cols * (rows_per_col / m_input) * per_tile
    const int64_t num_groups_per_row = K / group_size;
    const size_t packed_bytes_per_tile =
        (size_t)m_input * (size_t)K + (size_t)m_input * (size_t)num_groups_per_row * 2;
    const int64_t rows_per_col = N / cols;
    const int64_t tiles_per_col = rows_per_col / m_input;
    const size_t n_bytes = (size_t)cols * (size_t)tiles_per_col * packed_bytes_per_tile;

    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[kernel_slot].group_id(arg_group_id));

        uint8_t * dst = (uint8_t *)new_bo.map<void*>();
        const uint8_t * src_q8_0 = (const uint8_t *)weight->data;

        // Repack in-place into the uint8 BO. Note: ggml Q8_0 tensor stores M=N
        // rows by K columns, each row a contiguous sequence of (K/32) 34-byte
        // blocks. The gemv_int8 "M" parameter is N here (output-row dim).
        xdna_repack_q8_0_to_gemv_int8(src_q8_0, /*M=*/N, K, m_input, cols,
                                      group_size, dst);

        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(weight->data, std::move(new_bo));

        fprintf(stderr,
                "ggml-xdna: warm SwiGLU-int8 %s K=%lld N=%lld g=%d m_in=%d weight=%s (%zu cached)\n",
                slot_name, (long long)K, (long long)N, group_size, m_input,
                weight->name, cache.size());
        fflush(stderr);
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm SwiGLU-int8 %s weight: %s\n",
                       slot_name, e.what());
        return nullptr;
    }
}

// Forward decls for INT8 shape-dispatchability check.
static bool xdna_shape_dispatchable_swiglu_decode_int8(int64_t K_emb, int64_t N_hid,
                                                       int num_cols, int group_size);

// Dispatch a W8A16 fused SwiGLU FFN: Q8_0 gate/up/down × bf16 input.
// Mirrors ggml_backend_xdna_mul_mat_swiglu but with:
//   - Weight BOs are uint8 packed buffers (not transposed bf16).
//   - Weights are repacked from Q8_0 on first touch and cached per data-ptr.
//   - Inner matmul arg order is GEMV-style (matrix, vector, output) — same as
//     the bf16 decode path, since gemv_int8 preserves that ordering.
static void ggml_backend_xdna_mul_mat_swiglu_int8(
        ggml_backend_xdna_context * ctx,
        struct ggml_tensor * gate_dst,
        struct ggml_tensor * up_dst,
        struct ggml_tensor * glu_dst,
        struct ggml_tensor * dst_final,
        const struct ggml_tensor * src0_gate_w,
        const struct ggml_tensor * src0_up_w,
        const struct ggml_tensor * src0_down_w,
        const struct ggml_tensor * src1_input) {
    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-int8 called but XRT device invalid\n");
        return;
    }

    const int64_t M             = src1_input->ne[1];
    const int64_t embedding_dim = src1_input->ne[0];
    const int64_t hidden_dim    = src0_gate_w->ne[1];

    // Use configurable num_cols (default 4, or set by GGML_XDNA_NUM_COLS)
    int num_cols   = ctx->num_cols;
    const int group_size = 32;  // Q8_0 block size

    const xdna_op_kind op_kind = XDNA_OP_SWIGLU_DECODE_INT8;
    const int64_t seq_len      = 1;  // decode only (M==1)
    const int     tile_m       = 0;  // unused for decode path

    // Precompute the clamped tile_size_input values the Python side will end
    // up using inside GEMVInt8.__post_init__. Mirrors that function — the two
    // stages have different (K, tso) so potentially different clamps.
    //   gemv_int8_1: M=hidden_dim, K=embedding_dim, tso=hidden_dim/cols
    //   gemv_int8_2: M=embedding_dim, K=hidden_dim, tso=embedding_dim/cols
    const int mm1_tso = (int)(hidden_dim / num_cols);
    const int mm2_tso = (int)(embedding_dim / num_cols);
    const int mm1_m_input = xdna_clamp_gemv_int8_tile_in(
        /*requested_tsi=*/1, mm1_tso, embedding_dim, group_size);
    const int mm2_m_input = xdna_clamp_gemv_int8_tile_in(
        /*requested_tsi=*/1, mm2_tso, hidden_dim,    group_size);

    std::string cache_key = make_swiglu_cache_key(
        op_kind, embedding_dim, hidden_dim, seq_len, num_cols, tile_m, group_size);

    if (!ensure_swiglu_compiled(ctx, cache_key, op_kind,
                                embedding_dim, hidden_dim, seq_len, num_cols,
                                tile_m, group_size)) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-int8 compile failed for %s\n", cache_key.c_str());
        return;
    }

    xdna_swiglu_kernel_entry * entry = get_or_load_swiglu_kernel(
        ctx, cache_key, op_kind, embedding_dim, hidden_dim, seq_len, num_cols,
        group_size, mm1_m_input, mm2_m_input);
    if (!entry) return;

    try {
        const size_t input_elems  = (size_t)embedding_dim * (size_t)M;
        const size_t output_elems = (size_t)embedding_dim * (size_t)M;
        const size_t hidden_elems = (size_t)hidden_dim    * (size_t)M;
        const size_t input_bytes  = input_elems  * sizeof(uint16_t);
        const size_t output_bytes = output_elems * sizeof(uint16_t);
        const size_t hidden_bytes = hidden_elems * sizeof(uint16_t);

        // Lazily allocate the 6 persistent I/O + intermediate BOs.
        // GEMV arg groups for gemv_int8: (matrix=3, vector=4, output=5) —
        // same as bf16 GEMV.
        if (!entry->input_bo) {
            entry->input_bo = std::make_unique<xrt::bo>(
                ctx->device, input_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(4));
        }
        if (!entry->output_bo) {
            entry->output_bo = std::make_unique<xrt::bo>(
                ctx->device, output_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_2].group_id(5));
        }
        if (!entry->left_bo) {
            entry->left_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(5));
        }
        if (!entry->right_bo) {
            entry->right_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(5));
        }
        if (!entry->left_swished_bo) {
            entry->left_swished_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SILU].group_id(4));
        }
        if (!entry->intermediate_bo) {
            entry->intermediate_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_ELTWISE].group_id(5));
        }

        // Warm (or look up) per-layer packed INT8 weight BOs.
        // gate+up share mm1 config (m_input=mm1_m_input); down uses mm2.
        xrt::bo * w1_bo_ptr = nullptr;
        xrt::bo * w2_bo_ptr = nullptr;
        xrt::bo * w3_bo_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            const int mm1_w_grp = 3;  // gemv_int8 matrix input
            const int mm2_w_grp = 3;
            w1_bo_ptr = swiglu_warm_weight_int8(
                ctx, entry, entry->w1_bo_cache, src0_gate_w,
                XDNA_SWIGLU_MATMUL_1, mm1_w_grp, mm1_m_input, "w_gate");
            w2_bo_ptr = swiglu_warm_weight_int8(
                ctx, entry, entry->w2_bo_cache, src0_up_w,
                XDNA_SWIGLU_MATMUL_1, mm1_w_grp, mm1_m_input, "w_up");
            w3_bo_ptr = swiglu_warm_weight_int8(
                ctx, entry, entry->w3_bo_cache, src0_down_w,
                XDNA_SWIGLU_MATMUL_2, mm2_w_grp, mm2_m_input, "w_down");
            if (!w1_bo_ptr || !w2_bo_ptr || !w3_bo_ptr) return;
        }

        using clk = std::chrono::steady_clock;
        static const bool prof = getenv("XDNA_DEBUG") != NULL;
        auto t_in_s  = clk::now();

        // Load fresh input activation (bf16-from-f32 convert).
        if (src1_input->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1_input->data,
                        (uint16_t *)entry->input_bo->map<void*>(), input_elems);
        } else {
            memcpy(entry->input_bo->map<void*>(), src1_input->data, input_bytes);
        }
        entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t_in_e  = clk::now();

        // 5-invoke SwiGLU sequence, batched as one runlist. Matches bf16 decode
        // arg order (matrix, vector, output) since gemv_int8 preserves it.
        const uint32_t mm1_isize  = (uint32_t)entry->insts_data[XDNA_SWIGLU_MATMUL_1].size();
        const uint32_t mm2_isize  = (uint32_t)entry->insts_data[XDNA_SWIGLU_MATMUL_2].size();
        const uint32_t silu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_SILU].size();
        const uint32_t eltw_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_ELTWISE].size();

        auto t_build_s = clk::now();
        xrt::runlist rl(entry->hw_ctx);

        // 1. gemv_int8_1(w_gate, input) -> left
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(2, mm1_isize);
            r.set_arg(3, *w1_bo_ptr);
            r.set_arg(4, *entry->input_bo);
            r.set_arg(5, *entry->left_bo);
            rl.add(r);
        }
        // 2. gemv_int8_1(w_up, input) -> right
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(2, mm1_isize);
            r.set_arg(3, *w2_bo_ptr);
            r.set_arg(4, *entry->input_bo);
            r.set_arg(5, *entry->right_bo);
            rl.add(r);
        }
        // 3. silu(left) -> left_swished
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_SILU]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_SILU]);
            r.set_arg(2, silu_isize);
            r.set_arg(3, *entry->left_bo);
            r.set_arg(4, *entry->left_swished_bo);
            rl.add(r);
        }
        // 4. eltwise_mul(left_swished, right) -> intermediate
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_ELTWISE]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_ELTWISE]);
            r.set_arg(2, eltw_isize);
            r.set_arg(3, *entry->left_swished_bo);
            r.set_arg(4, *entry->right_bo);
            r.set_arg(5, *entry->intermediate_bo);
            rl.add(r);
        }
        // 5. gemv_int8_2(w_down, intermediate) -> output
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_2]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_MATMUL_2]);
            r.set_arg(2, mm2_isize);
            r.set_arg(3, *w3_bo_ptr);
            r.set_arg(4, *entry->intermediate_bo);
            r.set_arg(5, *entry->output_bo);
            rl.add(r);
        }
        auto t_build_e = clk::now();

        rl.execute();
        auto t_exec_e = clk::now();

        rl.wait();
        auto t_wait_e = clk::now();

        auto t_out_s = clk::now();
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->output_bo->map<void*>(),
                    (float *)dst_final->data, output_elems);
        auto t_out_e = clk::now();

        // Write back intermediates for any downstream consumer that looks past
        // the fused pattern (matches bf16 path).
        auto t_wbl_s = clk::now();
        entry->left_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->left_bo->map<void*>(),
                    (float *)gate_dst->data, hidden_elems);
        auto t_wbl_e = clk::now();

        auto t_wbr_s = clk::now();
        entry->right_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->right_bo->map<void*>(),
                    (float *)up_dst->data, hidden_elems);
        auto t_wbr_e = clk::now();

        auto t_wbi_s = clk::now();
        entry->intermediate_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->intermediate_bo->map<void*>(),
                    (float *)glu_dst->data, hidden_elems);
        auto t_wbi_e = clk::now();

        if (prof) {
            auto us = [](clk::time_point a, clk::time_point b) {
                return (long long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                "ggml-xdna: swiglu_prof decode_int8 M=%lld K=%lld N=%lld in=%lld "
                "rl_build=%lld rl_exec=%lld rl_wait=%lld "
                "out=%lld wb_l=%lld wb_r=%lld wb_i=%lld total=%lld us\n",
                (long long)M, (long long)embedding_dim, (long long)hidden_dim,
                us(t_in_s,    t_in_e),
                us(t_build_s, t_build_e),
                us(t_build_e, t_exec_e),
                us(t_exec_e,  t_wait_e),
                us(t_out_s,   t_out_e),
                us(t_wbl_s,   t_wbl_e),
                us(t_wbr_s,   t_wbr_e),
                us(t_wbi_s,   t_wbi_e),
                us(t_in_s,    t_wbi_e));
        }

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-int8 XRT dispatch failed (%s)\n", e.what());
    }
}

// ============================================================================
// W8A8 INT8 SwiGLU prefill — per-tensor quantized activations + weights
// ============================================================================

// Dispatch a W8A8 fused SwiGLU FFN for prefill (M>=32): Q8_0 weights re-quantized
// to per-tensor int8, bf16 activations quantized to per-tensor int8, GEMM output
// is bf16 (via bf16s scaled accumulation). Inter-stage bf16 intermediates are
// re-quantized to int8 before the down projection GEMM.
static void ggml_backend_xdna_mul_mat_swiglu_prefill_int8(
        ggml_backend_xdna_context * ctx,
        struct ggml_tensor * gate_dst,
        struct ggml_tensor * up_dst,
        struct ggml_tensor * glu_dst,
        struct ggml_tensor * dst_final,
        const struct ggml_tensor * src0_gate_w,
        const struct ggml_tensor * src0_up_w,
        const struct ggml_tensor * src0_down_w,
        const struct ggml_tensor * src1_input) {
    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU prefill INT8 called but XRT device invalid\n");
        return;
    }

    const int64_t M = src1_input->ne[1];
    const int64_t embedding_dim = src1_input->ne[0];
    const int64_t hidden_dim = src0_gate_w->ne[1];
    int num_cols = ctx->num_cols;
    const xdna_op_kind op_kind = XDNA_OP_SWIGLU_PREFILL_INT8;
    const int tile_m = xdna_pick_swiglu_prefill_tile_m(M);
    const int tile_n = xdna_pick_swiglu_prefill_tile_n(embedding_dim, hidden_dim, num_cols);

    std::string cache_key = make_swiglu_cache_key(op_kind, embedding_dim, hidden_dim, M, num_cols, tile_m,
                                                  /*group_size=*/0, tile_n);
    if (!ensure_swiglu_compiled(ctx, cache_key, op_kind, embedding_dim, hidden_dim, M, num_cols, tile_m,
                                /*group_size=*/0, tile_n))
        return;
    auto * entry = get_or_load_swiglu_kernel(ctx, cache_key, op_kind, embedding_dim, hidden_dim, M, num_cols);
    if (!entry) return;

    try {
        const size_t input_i8_bytes    = (size_t)M * embedding_dim;      // int8
        const size_t hidden_bf16_bytes = (size_t)M * hidden_dim * 2;     // bf16
        const size_t output_bf16_bytes = (size_t)M * embedding_dim * 2;  // bf16
        const size_t input_elems       = (size_t)M * embedding_dim;
        const size_t hidden_elems      = (size_t)M * hidden_dim;

        // Lazily allocate persistent BOs.
        // GEMM1 (i8→bf16s): arg3=A(int8), arg4=B(int8), arg5=C(bf16)
        // GEMM2 (bf16):     arg3=A(bf16), arg4=B(bf16), arg5=C(bf16)
        if (!entry->input_bo) {
            entry->input_bo = std::make_unique<xrt::bo>(
                ctx->device, input_i8_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(3));
        }
        if (!entry->output_bo) {
            entry->output_bo = std::make_unique<xrt::bo>(
                ctx->device, output_bf16_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_2].group_id(5));
        }
        if (!entry->left_bo) {
            entry->left_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bf16_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(5));
        }
        if (!entry->right_bo) {
            entry->right_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bf16_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_MATMUL_1].group_id(5));
        }
        if (!entry->left_swished_bo) {
            entry->left_swished_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bf16_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_SILU].group_id(4));
        }
        if (!entry->intermediate_bo) {
            entry->intermediate_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bf16_bytes, xrt::bo::flags::host_only,
                entry->kernels[XDNA_SWIGLU_ELTWISE].group_id(5));
        }

        // Warm INT8 gate/up weights (Q8_0 → per-tensor int8 + scale).
        struct int8_weight_info {
            xrt::bo bo;
            float scale;
        };
        static std::unordered_map<const void *, int8_weight_info> s_int8_wt_cache;
        static std::mutex s_int8_wt_mutex;

        float wt_gate_scale, wt_up_scale;
        xrt::bo * w1_bo_ptr, * w2_bo_ptr;

        {
            std::lock_guard<std::mutex> lock(s_int8_wt_mutex);

            auto warm_int8_weight = [&](const ggml_tensor * wt, int kernel_slot,
                                        const char * name) -> std::pair<xrt::bo *, float> {
                auto it = s_int8_wt_cache.find(wt->data);
                if (it != s_int8_wt_cache.end()) {
                    return {&it->second.bo, it->second.scale};
                }
                const int64_t wK = wt->ne[0];
                const int64_t wN = wt->ne[1];
                xrt::bo new_bo(ctx->device, (size_t)wK * wN, xrt::bo::flags::host_only,
                               entry->kernels[kernel_slot].group_id(4));
                float scale = xdna_repack_q8_0_to_gemm_int8(
                    (const uint8_t *)wt->data, wN, wK,
                    (int8_t *)new_bo.map<void*>());
                new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                fprintf(stderr, "ggml-xdna: warm INT8 prefill %s K=%lld N=%lld scale=%.6f\n",
                        name, (long long)wK, (long long)wN, scale);
                auto [ins, _] = s_int8_wt_cache.emplace(
                    wt->data, int8_weight_info{std::move(new_bo), scale});
                return {&ins->second.bo, ins->second.scale};
            };

            auto [w1, s1] = warm_int8_weight(src0_gate_w, XDNA_SWIGLU_MATMUL_1, "w_gate");
            auto [w2, s2] = warm_int8_weight(src0_up_w, XDNA_SWIGLU_MATMUL_1, "w_up");
            w1_bo_ptr = w1; wt_gate_scale = s1;
            w2_bo_ptr = w2; wt_up_scale = s2;
        }

        // Warm bf16 down weight (Q8_0 → dequant → bf16 transposed).
        // Reuse the existing bf16 weight warmup from the bf16 prefill path.
        xrt::bo * w3_bo_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            w3_bo_ptr = swiglu_warm_weight(ctx, entry, entry->w3_bo_cache,
                                           src0_down_w, XDNA_SWIGLU_MATMUL_2,
                                           4, "w_down");
            if (!w3_bo_ptr) return;
        }

        using clk = std::chrono::steady_clock;
        static const bool prof = getenv("XDNA_DEBUG") != NULL;
        auto t_in_s = clk::now();

        // --- Stage 0: Quantize bf16 activation → int8 ---
        std::vector<uint16_t> input_bf16(input_elems);
        if (src1_input->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1_input->data, input_bf16.data(), input_elems);
        } else {
            memcpy(input_bf16.data(), src1_input->data, input_elems * 2);
        }
        float act_scale = xdna_quantize_bf16_to_int8(
            input_bf16.data(),
            (int8_t *)entry->input_bo->map<void*>(),
            input_elems);
        entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t_in_e = clk::now();

        // Pre-allocated insts BOs for gate and up (different bf16s scales).
        // Allocated once per entry, patched in-place each call to avoid
        // per-invocation xrt::bo allocation overhead.
        struct prefill_int8_insts_cache {
            std::vector<size_t> gemm1_offsets;
            xrt::bo gate_insts_bo;
            xrt::bo up_insts_bo;
            std::vector<char> gate_insts_data;
            std::vector<char> up_insts_data;
            bool initialized = false;
        };
        static std::unordered_map<std::string, prefill_int8_insts_cache> s_insts_cache;
        static std::mutex s_insts_mutex;
        {
            std::lock_guard<std::mutex> lock(s_insts_mutex);
            if (s_insts_cache.find(cache_key) == s_insts_cache.end()) {
                prefill_int8_insts_cache ic;
                ic.gemm1_offsets = xdna_find_bf16s_scale_offsets(
                    entry->insts_data[XDNA_SWIGLU_MATMUL_1]);
                ic.gate_insts_data = entry->insts_data[XDNA_SWIGLU_MATMUL_1];
                ic.up_insts_data   = entry->insts_data[XDNA_SWIGLU_MATMUL_1];
                auto & k = entry->kernels[XDNA_SWIGLU_MATMUL_1];
                size_t sz = ic.gate_insts_data.size();
                ic.gate_insts_bo = xrt::bo(ctx->device, sz,
                    xrt::bo::flags::cacheable, k.group_id(1));
                ic.up_insts_bo = xrt::bo(ctx->device, sz,
                    xrt::bo::flags::cacheable, k.group_id(1));
                ic.initialized = true;
                s_insts_cache[cache_key] = std::move(ic);
            }
        }
        auto & ic = s_insts_cache[cache_key];

        // Patch scales in-place and sync (no BO allocation).
        xdna_patch_bf16s_scale(ic.gate_insts_data, ic.gemm1_offsets, act_scale * wt_gate_scale);
        memcpy(ic.gate_insts_bo.map<void*>(), ic.gate_insts_data.data(), ic.gate_insts_data.size());
        ic.gate_insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xdna_patch_bf16s_scale(ic.up_insts_data, ic.gemm1_offsets, act_scale * wt_up_scale);
        memcpy(ic.up_insts_bo.map<void*>(), ic.up_insts_data.data(), ic.up_insts_data.size());
        ic.up_insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const uint32_t mm1_isize  = (uint32_t)entry->insts_data[XDNA_SWIGLU_MATMUL_1].size();
        const uint32_t mm2_isize  = (uint32_t)entry->insts_data[XDNA_SWIGLU_MATMUL_2].size();
        const uint32_t silu_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_SILU].size();
        const uint32_t eltw_isize = (uint32_t)entry->insts_data[XDNA_SWIGLU_ELTWISE].size();

        // Build single runlist: gate(i8) + up(i8) + silu + mul + down(bf16).
        // Each xrt::run gets its own insts_bo, so gate and up can have different
        // bf16s scales despite sharing the same kernel.
        auto t_build_s = clk::now();
        xrt::runlist rl(entry->hw_ctx);

        // GEMM1 gate (i8 × i8 → bf16s)
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(0, 3u);
            r.set_arg(1, ic.gate_insts_bo);
            r.set_arg(2, mm1_isize);
            r.set_arg(3, *entry->input_bo);
            r.set_arg(4, *w1_bo_ptr);
            r.set_arg(5, *entry->left_bo);
            rl.add(r);
        }

        // GEMM1 up (i8 × i8 → bf16s, different scale)
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_1]);
            r.set_arg(0, 3u);
            r.set_arg(1, ic.up_insts_bo);
            r.set_arg(2, mm1_isize);
            r.set_arg(3, *entry->input_bo);
            r.set_arg(4, *w2_bo_ptr);
            r.set_arg(5, *entry->right_bo);
            rl.add(r);
        }

        // SiLU(left) → left_swished
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_SILU]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_SILU]);
            r.set_arg(2, silu_isize);
            r.set_arg(3, *entry->left_bo);
            r.set_arg(4, *entry->left_swished_bo);
            rl.add(r);
        }

        // MUL(left_swished, right) → intermediate
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_ELTWISE]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_ELTWISE]);
            r.set_arg(2, eltw_isize);
            r.set_arg(3, *entry->left_swished_bo);
            r.set_arg(4, *entry->right_bo);
            r.set_arg(5, *entry->intermediate_bo);
            rl.add(r);
        }

        // GEMM2 down (bf16 × bf16 → bf16, no scale patching)
        {
            xrt::run r(entry->kernels[XDNA_SWIGLU_MATMUL_2]);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->insts_bo[XDNA_SWIGLU_MATMUL_2]);
            r.set_arg(2, mm2_isize);
            r.set_arg(3, *entry->intermediate_bo);
            r.set_arg(4, *w3_bo_ptr);
            r.set_arg(5, *entry->output_bo);
            rl.add(r);
        }
        auto t_build_e = clk::now();

        rl.execute();
        auto t_exec_e = clk::now();

        rl.wait();
        auto t_wait_e = clk::now();

        // --- Read back output ---
        auto t_out_s = clk::now();
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32(entry->output_bo->map<uint16_t*>(),
                    (float *)dst_final->data, input_elems);
        auto t_out_e = clk::now();

        // Write back intermediates for downstream consumers.
        entry->left_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32(entry->left_bo->map<uint16_t*>(),
                    (float *)gate_dst->data, hidden_elems);
        entry->right_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32(entry->right_bo->map<uint16_t*>(),
                    (float *)up_dst->data, hidden_elems);
        entry->intermediate_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32(entry->intermediate_bo->map<uint16_t*>(),
                    (float *)glu_dst->data, hidden_elems);
        auto t_wb_e = clk::now();

        if (prof) {
            auto us = [](clk::time_point a, clk::time_point b) {
                return (long long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                "ggml-xdna: swiglu_prof prefill_int8 M=%lld K=%lld N=%lld "
                "in=%lld rl_build=%lld rl_exec=%lld rl_wait=%lld out=%lld wb=%lld total=%lld us\n",
                (long long)M, (long long)embedding_dim, (long long)hidden_dim,
                us(t_in_s,     t_in_e),
                us(t_build_s,  t_build_e),
                us(t_build_e,  t_exec_e),
                us(t_exec_e,   t_wait_e),
                us(t_out_s,    t_out_e),
                us(t_out_e,    t_wb_e),
                us(t_in_s,     t_wb_e));
        }

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU prefill INT8 XRT dispatch failed (%s)\n", e.what());
    }
}

// ============================================================================
// Fused INT8 SwiGLU — 2-dispatch path (fused gate+up+silu+mul + down GEMV)
// ============================================================================

static std::string make_swiglu_fused_cache_key(int64_t embedding_dim, int64_t hidden_dim,
                                                int num_cols, int group_size) {
    char buf[256];
    snprintf(buf, sizeof(buf), "swiglu_fused_int8_K%lld_N%lld_%dcol_g%d",
             (long long)embedding_dim, (long long)hidden_dim, num_cols, group_size);
    return std::string(buf);
}

static bool swiglu_fused_bundle_present(const std::string & bundle_dir) {
    std::ifstream f1(bundle_dir + "/combined.xclbin");
    if (!f1.good()) return false;
    std::ifstream f2(bundle_dir + "/swiglu_fused.insts");
    if (!f2.good()) return false;
    std::ifstream f3(bundle_dir + "/swiglu_down_gemv_int8.insts");
    if (!f3.good()) return false;
    return true;
}

static bool ensure_swiglu_fused_compiled(ggml_backend_xdna_context * ctx,
                                         const std::string & cache_key,
                                         int64_t embedding_dim, int64_t hidden_dim,
                                         int num_cols, int group_size) {
    // Skip recompilation if we already failed for this key
    if (ctx->swiglu_compile_failed.count(cache_key)) {
        return false;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    if (swiglu_fused_bundle_present(bundle_dir)) {
        return true;
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet swiglu-fused-int8 --embedding-dim %lld --hidden-dim %lld "
             "--num-aie-columns %d --group-size %d --out %s%s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)embedding_dim, (long long)hidden_dim,
             num_cols, group_size, bundle_dir.c_str(), xdna_null_redirect());
    fprintf(stderr, "ggml-xdna: compiling SwiGLU fused INT8 K=%lld N=%lld g=%d (first run, will be cached)...\n",
                  (long long)embedding_dim, (long long)hidden_dim, group_size);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU fused INT8 compilation failed (exit code %d)\n", ret);
        ctx->swiglu_compile_failed.insert(cache_key);
        return false;
    }

    if (!swiglu_fused_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU fused INT8 compilation succeeded but bundle files missing in %s\n",
                       bundle_dir.c_str());
        ctx->swiglu_compile_failed.insert(cache_key);
        return false;
    }

    fprintf(stderr, "ggml-xdna: SwiGLU fused INT8 compilation complete, cached at %s\n", bundle_dir.c_str());
    return true;
}

static xdna_swiglu_fused_entry * get_or_load_swiglu_fused_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t embedding_dim, int64_t hidden_dim,
        int num_cols, int group_size,
        int fused_m_input, int down_m_input) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);

    auto it = ctx->swiglu_fused_cache.find(cache_key);
    if (it != ctx->swiglu_fused_cache.end()) {
        return &it->second;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;

    try {
        xdna_swiglu_fused_entry entry;
        entry.op_kind       = XDNA_OP_SWIGLU_FUSED_INT8;
        entry.embedding_dim = embedding_dim;
        entry.hidden_dim    = hidden_dim;
        entry.num_cols      = num_cols;
        entry.group_size    = group_size;
        entry.fused_m_input = fused_m_input;
        entry.down_m_input  = down_m_input;

        // Load combined xclbin (fused gate+up+silu+mul + down GEMV)
        entry.xclbin = xrt::xclbin(bundle_dir + "/combined.xclbin");
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);
        entry.fused_kernel = xrt::kernel(entry.hw_ctx, "swiglu_fused");
        entry.down_kernel  = xrt::kernel(entry.hw_ctx, "swiglu_down_gemv_int8");

        // Load fused kernel insts
        entry.fused_insts_data = read_binary_file(bundle_dir + "/swiglu_fused.insts");
        if (entry.fused_insts_data.empty()) {
            GGML_LOG_ERROR("ggml-xdna: failed to read swiglu_fused insts file\n");
            return nullptr;
        }
        entry.fused_insts_bo = xrt::bo(ctx->device, entry.fused_insts_data.size(),
                                        xrt::bo::flags::cacheable,
                                        entry.fused_kernel.group_id(1));
        entry.fused_insts_bo.write(entry.fused_insts_data.data());
        entry.fused_insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Load down GEMV kernel insts
        entry.down_insts_data = read_binary_file(bundle_dir + "/swiglu_down_gemv_int8.insts");
        if (entry.down_insts_data.empty()) {
            GGML_LOG_ERROR("ggml-xdna: failed to read swiglu_down_gemv_int8 insts file\n");
            return nullptr;
        }
        entry.down_insts_bo = xrt::bo(ctx->device, entry.down_insts_data.size(),
                                       xrt::bo::flags::cacheable,
                                       entry.down_kernel.group_id(1));
        entry.down_insts_bo.write(entry.down_insts_data.data());
        entry.down_insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto [inserted_it, _] = ctx->swiglu_fused_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded SwiGLU fused INT8 kernel bundle for %s\n", cache_key.c_str());
        return &inserted_it->second;

    } catch (const std::exception & e) {
        fprintf(stderr, "ggml-xdna: SwiGLU fused cache stale for %s (%s), "
                      "invalidating — will recompile on next dispatch\n",
                      cache_key.c_str(), e.what());
#ifdef _WIN32
        std::string rm_cmd = "rd /s /q \"" + bundle_dir + "\" 2>nul";
#else
        std::string rm_cmd = "rm -rf \"" + bundle_dir + "\"";
#endif
        system(rm_cmd.c_str());
        return nullptr;
    }
}

// Warm (or look up) an INT8 weight BO for the fused SwiGLU path. Identical to
// swiglu_warm_weight_int8 but takes an xrt::kernel directly instead of
// entry+slot, since the fused entry has two named kernels (not an array).
static xrt::bo * swiglu_fused_warm_weight_int8(ggml_backend_xdna_context * ctx,
                                                xdna_swiglu_fused_entry * entry,
                                                std::unordered_map<const void *, xrt::bo> & cache,
                                                const struct ggml_tensor * weight,
                                                xrt::kernel & kernel,
                                                int arg_group_id,
                                                int m_input,
                                                int num_cols,
                                                int group_size,
                                                const char * slot_name) {
    auto it = cache.find(weight->data);
    if (it != cache.end()) {
        return &it->second;
    }

    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];

    const int64_t num_groups_per_row = K / group_size;
    const size_t packed_bytes_per_tile =
        (size_t)m_input * (size_t)K + (size_t)m_input * (size_t)num_groups_per_row * 2;
    const int64_t rows_per_col = N / num_cols;
    const int64_t tiles_per_col = rows_per_col / m_input;
    const size_t n_bytes = (size_t)num_cols * (size_t)tiles_per_col * packed_bytes_per_tile;

    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       kernel.group_id(arg_group_id));

        uint8_t * dst = (uint8_t *)new_bo.map<void*>();
        const uint8_t * src_q8_0 = (const uint8_t *)weight->data;

        xdna_repack_q8_0_to_gemv_int8(src_q8_0, /*M=*/N, K, m_input, num_cols,
                                      group_size, dst);

        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(weight->data, std::move(new_bo));

        fprintf(stderr,
                "ggml-xdna: warm SwiGLU-fused-int8 %s K=%lld N=%lld g=%d m_in=%d weight=%s (%zu cached)\n",
                slot_name, (long long)K, (long long)N, group_size, m_input,
                weight->name, cache.size());
        fflush(stderr);
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm SwiGLU-fused-int8 %s weight: %s\n",
                       slot_name, e.what());
        return nullptr;
    }
}

// Dispatch a fused INT8 SwiGLU FFN via runlist (single combined xclbin):
//   Run 1: fused kernel (gate_gemv + up_gemv + silu + eltwise_mul) → intermediate
//   Run 2: down GEMV kernel (intermediate → output)
// Both kernels share a single hw_context from the combined xclbin. The runlist
// executes both runs atomically in add() order — no intermediate host sync needed.
static void ggml_backend_xdna_mul_mat_swiglu_fused_int8(
        ggml_backend_xdna_context * ctx,
        struct ggml_tensor * gate_dst,
        struct ggml_tensor * up_dst,
        struct ggml_tensor * glu_dst,
        struct ggml_tensor * dst_final,
        const struct ggml_tensor * src0_gate_w,
        const struct ggml_tensor * src0_up_w,
        const struct ggml_tensor * src0_down_w,
        const struct ggml_tensor * src1_input) {
    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-fused-int8 called but XRT device invalid\n");
        return;
    }

    const int64_t embedding_dim = src1_input->ne[0];
    const int64_t hidden_dim    = src0_gate_w->ne[1];

    const int num_cols   = 8;
    const int group_size = 32;

    // Compute clamped tile_size_input for fused (gate/up) and down stages.
    //   fused: M=hidden_dim, K=embedding_dim, tso=hidden_dim/cols
    //   down:  M=embedding_dim, K=hidden_dim, tso=embedding_dim/cols
    const int fused_tso = (int)(hidden_dim / num_cols);
    const int down_tso  = (int)(embedding_dim / num_cols);
    const int fused_m_input = xdna_best_gemv_int8_tile_in(
        fused_tso, embedding_dim, group_size);
    const int down_m_input = xdna_best_gemv_int8_tile_in(
        down_tso, hidden_dim, group_size);

    std::string cache_key = make_swiglu_fused_cache_key(
        embedding_dim, hidden_dim, num_cols, group_size);

    if (!ensure_swiglu_fused_compiled(ctx, cache_key,
                                      embedding_dim, hidden_dim, num_cols, group_size)) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-fused-int8 compile failed for %s\n", cache_key.c_str());
        return;
    }

    xdna_swiglu_fused_entry * entry = get_or_load_swiglu_fused_kernel(
        ctx, cache_key, embedding_dim, hidden_dim, num_cols, group_size,
        fused_m_input, down_m_input);
    if (!entry) return;

    try {
        const size_t input_elems  = (size_t)embedding_dim;
        const size_t output_elems = (size_t)embedding_dim;
        const size_t hidden_elems = (size_t)hidden_dim;
        const size_t input_bytes  = input_elems  * sizeof(uint16_t);
        const size_t output_bytes = output_elems * sizeof(uint16_t);
        const size_t hidden_bytes = hidden_elems * sizeof(uint16_t);

        // Lazily allocate persistent BOs.
        // Fused kernel args: (opcode, insts, isize, w_gate, w_up, input, intermediate)
        //   w_gate @ group_id(3), w_up @ group_id(4), input @ group_id(5), output @ group_id(6)
        // Down kernel args: (opcode, insts, isize, w_down, intermediate, output)
        //   w_down @ group_id(3), intermediate @ group_id(4), output @ group_id(5)
        if (!entry->input_bo) {
            entry->input_bo = std::make_unique<xrt::bo>(
                ctx->device, input_bytes, xrt::bo::flags::host_only,
                entry->fused_kernel.group_id(5));
        }
        if (!entry->intermediate_bo) {
            entry->intermediate_bo = std::make_unique<xrt::bo>(
                ctx->device, hidden_bytes, xrt::bo::flags::host_only,
                entry->fused_kernel.group_id(6));
        }
        if (!entry->output_bo) {
            entry->output_bo = std::make_unique<xrt::bo>(
                ctx->device, output_bytes, xrt::bo::flags::host_only,
                entry->down_kernel.group_id(5));
        }

        // Warm (or look up) per-layer packed INT8 weight BOs.
        xrt::bo * w_gate_bo = nullptr;
        xrt::bo * w_up_bo   = nullptr;
        xrt::bo * w_down_bo = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            w_gate_bo = swiglu_fused_warm_weight_int8(
                ctx, entry, entry->w_gate_bo_cache, src0_gate_w,
                entry->fused_kernel, 3, fused_m_input, num_cols, group_size, "w_gate");
            w_up_bo = swiglu_fused_warm_weight_int8(
                ctx, entry, entry->w_up_bo_cache, src0_up_w,
                entry->fused_kernel, 4, fused_m_input, num_cols, group_size, "w_up");
            w_down_bo = swiglu_fused_warm_weight_int8(
                ctx, entry, entry->w_down_bo_cache, src0_down_w,
                entry->down_kernel, 3, down_m_input, num_cols, group_size, "w_down");
            if (!w_gate_bo || !w_up_bo || !w_down_bo) return;
        }

        using clk = std::chrono::steady_clock;
        static const bool prof = getenv("XDNA_DEBUG") != NULL;
        auto t_in_s = clk::now();

        // Load fresh input activation (bf16-from-f32 convert).
        if (src1_input->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1_input->data,
                        (uint16_t *)entry->input_bo->map<void*>(), input_elems);
        } else {
            memcpy(entry->input_bo->map<void*>(), src1_input->data, input_bytes);
        }
        entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto t_in_e = clk::now();

        // Build runlist: fused kernel + down GEMV, dispatched atomically.
        // No intermediate host sync needed — both runs share one hw_context
        // and the runlist executes them in add() order on-device.
        auto t_rl_build_s = clk::now();
        xrt::runlist rl(entry->hw_ctx);

        // Run 1: fused kernel (gate + up + silu + eltwise_mul)
        // Args: (opcode=3, insts_bo, insts_size, w_gate, w_up, input, intermediate)
        {
            xrt::run r(entry->fused_kernel);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->fused_insts_bo);
            r.set_arg(2, (uint32_t)entry->fused_insts_data.size());
            r.set_arg(3, *w_gate_bo);
            r.set_arg(4, *w_up_bo);
            r.set_arg(5, *entry->input_bo);
            r.set_arg(6, *entry->intermediate_bo);
            rl.add(r);
        }

        // Run 2: down GEMV kernel
        // Args: (opcode=3, insts_bo, insts_size, w_down, intermediate, output)
        {
            xrt::run r(entry->down_kernel);
            r.set_arg(0, 3u);
            r.set_arg(1, entry->down_insts_bo);
            r.set_arg(2, (uint32_t)entry->down_insts_data.size());
            r.set_arg(3, *w_down_bo);
            r.set_arg(4, *entry->intermediate_bo);
            r.set_arg(5, *entry->output_bo);
            rl.add(r);
        }
        auto t_rl_build_e = clk::now();

        auto t_rl_exec_s = clk::now();
        rl.execute();
        auto t_rl_exec_e = clk::now();

        auto t_rl_wait_s = clk::now();
        rl.wait();
        auto t_rl_wait_e = clk::now();

        // Sync output back and convert bf16 → f32.
        auto t_out_s = clk::now();
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->output_bo->map<void*>(),
                    (float *)dst_final->data, output_elems);
        auto t_out_e = clk::now();

        // Write back intermediate to glu_dst for downstream inspection tools.
        // The fused kernel doesn't produce separate gate/up outputs — those
        // intermediate tensors are never read by the graph (i += 3 skips them).
        // Zero them as a safety measure.
        auto t_wb_s = clk::now();
        bf16_to_f32((const uint16_t *)entry->intermediate_bo->map<void*>(),
                    (float *)glu_dst->data, hidden_elems);
        memset(gate_dst->data, 0, (size_t)hidden_dim * sizeof(float));
        memset(up_dst->data,   0, (size_t)hidden_dim * sizeof(float));
        auto t_wb_e = clk::now();

        if (prof) {
            auto us = [](clk::time_point a, clk::time_point b) {
                return (long long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                "ggml-xdna: swiglu_prof fused_int8 K=%lld N=%lld in=%lld "
                "rl_build=%lld rl_exec=%lld rl_wait=%lld out=%lld wb=%lld total=%lld us\n",
                (long long)embedding_dim, (long long)hidden_dim,
                us(t_in_s,       t_in_e),
                us(t_rl_build_s, t_rl_build_e),
                us(t_rl_exec_s,  t_rl_exec_e),
                us(t_rl_wait_s,  t_rl_wait_e),
                us(t_out_s,      t_out_e),
                us(t_wb_s,       t_wb_e),
                us(t_in_s,       t_wb_e));
        }

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: SwiGLU-fused-int8 XRT dispatch failed (%s)\n", e.what());
    }
}
#endif  // INT8 dead code

// ============================================================================
// QKV fused projection — single combined xclbin, 3 bf16 GEMVs sharing input
// ============================================================================

static std::string make_qkv_cache_key(int64_t embedding_dim,
                                      int64_t q_dim, int64_t k_dim, int64_t v_dim,
                                      int num_cols) {
    char buf[256];
    snprintf(buf, sizeof(buf), "qkv_K%lld_Nq%lld_Nk%lld_Nv%lld_%dcol",
             (long long)embedding_dim, (long long)q_dim, (long long)k_dim, (long long)v_dim, num_cols);
    return std::string(buf);
}

static bool qkv_bundle_present(const std::string & bundle_dir) {
    std::ifstream f1(bundle_dir + "/combined.xclbin");
    if (!f1.good()) return false;
    std::ifstream f2(bundle_dir + "/qkv_main.insts");
    if (!f2.good()) return false;
    return true;
}

static bool ensure_qkv_compiled(ggml_backend_xdna_context * ctx,
                                const std::string & cache_key,
                                int64_t embedding_dim,
                                int64_t q_dim, int64_t k_dim, int64_t v_dim,
                                int num_cols) {
    // Skip recompilation if we already failed for this key
    if (ctx->qkv_compile_failed.count(cache_key)) {
        return false;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    if (qkv_bundle_present(bundle_dir)) {
        return true;
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet qkv --embedding-dim %lld --q-dim %lld --k-dim %lld --v-dim %lld "
             "--num-aie-columns %d --out %s%s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)embedding_dim, (long long)q_dim, (long long)k_dim, (long long)v_dim,
             num_cols, bundle_dir.c_str(), xdna_null_redirect());
    fprintf(stderr, "ggml-xdna: compiling QKV K=%lld Nq=%lld Nk=%lld Nv=%lld cols=%d (first run, will be cached)...\n",
                  (long long)embedding_dim, (long long)q_dim, (long long)k_dim, (long long)v_dim, num_cols);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: QKV compilation failed (exit code %d)\n", ret);
        ctx->qkv_compile_failed.insert(cache_key);
        return false;
    }

    if (!qkv_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: QKV compilation succeeded but bundle files missing in %s\n",
                       bundle_dir.c_str());
        ctx->qkv_compile_failed.insert(cache_key);
        return false;
    }

    fprintf(stderr, "ggml-xdna: QKV compilation complete, cached at %s\n", bundle_dir.c_str());
    return true;
}

// Try to load the QKV xclbin and instantiate the kernel.
// Returns true on success, false if the xclbin is stale/incompatible
// (caller should invalidate cache and retry).
static bool try_load_qkv_entry(xdna_qkv_entry & entry,
                               ggml_backend_xdna_context * ctx,
                               const std::string & bundle_dir) {
    entry.xclbin = xrt::xclbin(bundle_dir + "/combined.xclbin");
    ctx->device.register_xclbin(entry.xclbin);
    auto uuid = entry.xclbin.get_uuid();
    entry.hw_ctx = xrt::hw_context(ctx->device, uuid);
    // IRON-generated kernels use the default "MLIR_AIE" kernel name
    // (the function symbol in the ELF is always MLIR_AIE, regardless of
    // --xclbin-kernel-name display label).
    entry.fused_kernel = xrt::kernel(entry.hw_ctx, "MLIR_AIE");

    entry.insts_data = read_binary_file(bundle_dir + "/qkv_main.insts");
    if (entry.insts_data.empty()) {
        GGML_LOG_ERROR("ggml-xdna: failed to read qkv_main insts file\n");
        return false;
    }
    entry.insts_bo = xrt::bo(ctx->device, entry.insts_data.size(),
                              xrt::bo::flags::cacheable,
                              entry.fused_kernel.group_id(1));
    entry.insts_bo.write(entry.insts_data.data());
    entry.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return true;
}

static xdna_qkv_entry * get_or_load_qkv_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t embedding_dim,
        int64_t q_dim, int64_t k_dim, int64_t v_dim,
        int num_cols) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);

    auto it = ctx->qkv_cache.find(cache_key);
    if (it != ctx->qkv_cache.end()) {
        return &it->second;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;

    // First attempt: try loading the cached xclbin as-is.
    // If it fails (stale cache, wrong kernel name, etc.), delete the
    // bundle and recompile once.
    for (int attempt = 0; attempt < 2; attempt++) {
        try {
            xdna_qkv_entry entry;
            entry.op_kind       = XDNA_OP_QKV;
            entry.embedding_dim = embedding_dim;
            entry.q_dim         = q_dim;
            entry.k_dim         = k_dim;
            entry.v_dim         = v_dim;
            entry.total_out     = q_dim + k_dim + v_dim;
            entry.num_cols      = num_cols;

            if (!try_load_qkv_entry(entry, ctx, bundle_dir)) {
                GGML_LOG_ERROR("ggml-xdna: QKV bundle files incomplete in %s\n",
                               bundle_dir.c_str());
                return nullptr;
            }

            auto [inserted_it, _] = ctx->qkv_cache.emplace(cache_key, std::move(entry));
            fprintf(stderr, "ggml-xdna: loaded QKV kernel bundle for %s\n", cache_key.c_str());
            return &inserted_it->second;

        } catch (const std::exception & e) {
            if (attempt == 0) {
                // First attempt failed — likely a stale xclbin with a
                // different kernel name.  Nuke the bundle and recompile.
                fprintf(stderr, "ggml-xdna: QKV cache stale for %s (%s), "
                              "invalidating and recompiling...\n",
                              cache_key.c_str(), e.what());
#ifdef _WIN32
                std::string rm_cmd = "rd /s /q \"" + bundle_dir + "\" 2>nul";
#else
                std::string rm_cmd = "rm -rf \"" + bundle_dir + "\"";
#endif
                system(rm_cmd.c_str());
                if (!ensure_qkv_compiled(ctx, cache_key,
                                         embedding_dim, q_dim, k_dim, v_dim,
                                         num_cols)) {
                    GGML_LOG_ERROR("ggml-xdna: QKV recompilation failed for %s\n",
                                   cache_key.c_str());
                    return nullptr;
                }
                // Loop around for second attempt with fresh xclbin.
            } else {
                GGML_LOG_ERROR("ggml-xdna: failed to load QKV kernel %s "
                               "even after recompilation: %s\n",
                               cache_key.c_str(), e.what());
                return nullptr;
            }
        }
    }
    return nullptr;  // unreachable
}

// Concatenate Wq, Wk, Wv row-wise into one buffer and upload.
// Weights are [N,K] row-major in ggml. Concatenated = [(Nq+Nk+Nv), K].
// Cache by q_ptr (assumes q/k/v weights change together per layer).
static xrt::bo * qkv_warm_concat_weight(
        ggml_backend_xdna_context * ctx,
        xdna_qkv_entry * entry,
        const struct ggml_tensor * w_q,
        const struct ggml_tensor * w_k,
        const struct ggml_tensor * w_v) {
    auto it = entry->w_concat_bo_cache.find(w_q->data);
    if (it != entry->w_concat_bo_cache.end()) {
        return &it->second;
    }

    const int64_t K = w_q->ne[0];  // embedding_dim
    const int64_t total_out = entry->total_out;
    const size_t total_bytes = (size_t)total_out * (size_t)K * sizeof(uint16_t);

    xrt::bo new_bo(ctx->device, total_bytes, xrt::bo::flags::host_only,
                   entry->fused_kernel.group_id(3));
    uint16_t * dst = (uint16_t *)new_bo.map<void*>();

    // Copy Wq
    size_t q_bytes = (size_t)entry->q_dim * (size_t)K * sizeof(uint16_t);
    if (w_q->type == GGML_TYPE_F32) {
        f32_to_bf16((const float *)w_q->data, dst, entry->q_dim * K);
    } else {
        memcpy(dst, w_q->data, q_bytes);
    }

    // Copy Wk
    size_t k_bytes = (size_t)entry->k_dim * (size_t)K * sizeof(uint16_t);
    uint16_t * dst_k = dst + entry->q_dim * K;
    if (w_k->type == GGML_TYPE_F32) {
        f32_to_bf16((const float *)w_k->data, dst_k, entry->k_dim * K);
    } else {
        memcpy(dst_k, w_k->data, k_bytes);
    }

    // Copy Wv
    size_t v_bytes = (size_t)entry->v_dim * (size_t)K * sizeof(uint16_t);
    uint16_t * dst_v = dst + (entry->q_dim + entry->k_dim) * K;
    if (w_v->type == GGML_TYPE_F32) {
        f32_to_bf16((const float *)w_v->data, dst_v, entry->v_dim * K);
    } else {
        memcpy(dst_v, w_v->data, v_bytes);
    }

    new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    fprintf(stderr,
            "ggml-xdna: warm QKV concat K=%lld total_out=%lld (%zu cached)\n",
            (long long)K, (long long)total_out, entry->w_concat_bo_cache.size() + 1);
    fflush(stderr);

    auto [ins, _] = entry->w_concat_bo_cache.emplace(w_q->data, std::move(new_bo));
    return &ins->second;
}

// Dispatch 1 fused bf16 QKV GEMV as a single xrt::run (single combined xclbin).
// Weights Wq, Wk, Wv are concatenated row-wise on host into one matrix BO.
// The fused kernel computes all 3 projections in one pass; host splits the
// output buffer into Q, K, V segments after execution.
static void ggml_backend_xdna_mul_mat_qkv(
        ggml_backend_xdna_context * ctx,
        struct ggml_tensor * q_dst,
        struct ggml_tensor * k_dst,
        struct ggml_tensor * v_dst,
        const struct ggml_tensor * src0_w_q,
        const struct ggml_tensor * src0_w_k,
        const struct ggml_tensor * src0_w_v,
        const struct ggml_tensor * src1_input) {
    if (!ctx->device_valid) {
        GGML_LOG_ERROR("ggml-xdna: QKV called but XRT device invalid\n");
        return;
    }

    const int64_t embedding_dim = src1_input->ne[0];
    const int64_t q_dim = src0_w_q->ne[1];
    const int64_t k_dim = src0_w_k->ne[1];
    const int64_t v_dim = src0_w_v->ne[1];

    int num_cols = ctx->num_cols;

    std::string cache_key = make_qkv_cache_key(
        embedding_dim, q_dim, k_dim, v_dim, num_cols);

    if (!ensure_qkv_compiled(ctx, cache_key,
                             embedding_dim, q_dim, k_dim, v_dim, num_cols)) {
        GGML_LOG_ERROR("ggml-xdna: QKV compile failed for %s\n", cache_key.c_str());
        return;
    }

    xdna_qkv_entry * entry = get_or_load_qkv_kernel(
        ctx, cache_key, embedding_dim, q_dim, k_dim, v_dim, num_cols);
    if (!entry) return;

    try {
        const size_t input_elems  = (size_t)embedding_dim;
        const size_t total_elems  = (size_t)(q_dim + k_dim + v_dim);
        const size_t input_bytes  = input_elems * sizeof(uint16_t);
        const size_t output_bytes = total_elems * sizeof(uint16_t);

        // Lazily allocate persistent BOs.
        // Fused GEMV args: (opcode, insts, isize, matrix, vector, output)
        //   matrix @ group_id(3), vector(input) @ group_id(4), output @ group_id(5)
        if (!entry->input_bo) {
            entry->input_bo = std::make_unique<xrt::bo>(
                ctx->device, input_bytes, xrt::bo::flags::host_only,
                entry->fused_kernel.group_id(4));
        }
        if (!entry->output_bo) {
            entry->output_bo = std::make_unique<xrt::bo>(
                ctx->device, output_bytes, xrt::bo::flags::host_only,
                entry->fused_kernel.group_id(5));
        }

        // Warm (or look up) concatenated weight BO.
        xrt::bo * w_concat_bo = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            w_concat_bo = qkv_warm_concat_weight(
                ctx, entry, src0_w_q, src0_w_k, src0_w_v);
            if (!w_concat_bo) return;
        }

        using clk = std::chrono::steady_clock;
        static const bool prof = getenv("XDNA_DEBUG") != NULL;

        const int64_t M = src1_input->ne[1];
        const size_t row_bytes = embedding_dim * sizeof(uint16_t);

        // Output stride per row in f32 (dst tensors are [dim, M] row-major).
        const size_t q_row_f32 = (size_t)q_dim;
        const size_t k_row_f32 = (size_t)k_dim;
        const size_t v_row_f32 = (size_t)v_dim;

        auto t_total_s = clk::now();

        for (int64_t m = 0; m < M; m++) {
            // Load input row m (bf16-from-f32 convert).
            const float * in_f32_row = (const float *)src1_input->data + m * embedding_dim;
            if (src1_input->type == GGML_TYPE_F32) {
                f32_to_bf16(in_f32_row,
                            (uint16_t *)entry->input_bo->map<void*>(), input_elems);
            } else {
                memcpy(entry->input_bo->map<void*>(),
                       (const char *)src1_input->data + m * row_bytes, row_bytes);
            }
            entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Run single GEMV.
            xrt::runlist rl(entry->hw_ctx);
            {
                xrt::run r(entry->fused_kernel);
                r.set_arg(0, 3u);
                r.set_arg(1, entry->insts_bo);
                r.set_arg(2, (uint32_t)entry->insts_data.size());
                r.set_arg(3, *w_concat_bo);
                r.set_arg(4, *entry->input_bo);
                r.set_arg(5, *entry->output_bo);
                rl.add(r);
            }
            rl.execute();
            rl.wait();

            // Sync output and split into Q, K, V row m.
            entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const uint16_t * out_bf16 = (const uint16_t *)entry->output_bo->map<void*>();
            bf16_to_f32(out_bf16,                    (float *)q_dst->data + m * q_row_f32, (size_t)q_dim);
            bf16_to_f32(out_bf16 + q_dim,            (float *)k_dst->data + m * k_row_f32, (size_t)k_dim);
            bf16_to_f32(out_bf16 + q_dim + k_dim,    (float *)v_dst->data + m * v_row_f32, (size_t)v_dim);
        }

        auto t_total_e = clk::now();

        if (prof) {
            auto us_total = (long long)std::chrono::duration_cast<std::chrono::microseconds>(
                t_total_e - t_total_s).count();
            fprintf(stderr,
                "ggml-xdna: qkv_prof K=%lld Nq=%lld Nk=%lld Nv=%lld M=%lld "
                "total=%lld us (%lld us/row)\n",
                (long long)embedding_dim, (long long)q_dim, (long long)k_dim, (long long)v_dim,
                (long long)M, us_total, M > 0 ? us_total / M : 0);
        }

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: QKV XRT dispatch failed (%s)\n", e.what());
    }
}

// Shape-dispatchability for chained QKV: mirror validate_qkv_shapes in compile.py.
// All three GEMV stages must tile at num_cols and satisfy kernel_vector_size (64).
static bool xdna_shape_dispatchable_qkv(int64_t K_emb, int64_t q_dim,
                                         int64_t k_dim, int64_t v_dim,
                                         int num_cols) {
    if (num_cols != 4 && num_cols != 8) return false;
    if (K_emb < 64 || K_emb % 64 != 0) return false;  // GEMV kernel_vector_size
    for (int64_t m : {q_dim, k_dim, v_dim}) {
        if (m <= 0) return false;
        if (m % num_cols != 0) return false;
        if ((m / num_cols) < 8) return false;
        if (m > 32768) return false;  // BD-overflow cap
    }
    return true;
}

// Shape-dispatchability for fused SwiGLU decode. Must be a superset of
// validate_swiglu_decode_shapes in compile.py.
static bool xdna_shape_dispatchable_swiglu_decode(int64_t K_emb, int64_t N_hid, int num_cols) {
    if (num_cols != 4 && num_cols != 8) return false;
    if (K_emb < 64 || K_emb % 64 != 0) return false;  // GEMV K constraint (kernel_vector_size)
    if (N_hid < 64 || N_hid % 64 != 0) return false;  // inner GEMV_2 K = hidden_dim
    if (N_hid % (num_cols * 2) != 0)   return false;  // SiLU per-column tile
    if (K_emb % num_cols != 0)         return false;
    if (N_hid > 32768 || K_emb > 32768) return false;  // BD-overflow cap

    // Each GEMV stage must tile cleanly — per-col size must be >= 8 for tile_out.
    if ((N_hid / num_cols) < 8) return false;
    if ((K_emb / num_cols) < 8) return false;
    return true;
}

// Shape-dispatchability for fused SwiGLU prefill. With tile_m/tile_n override
// support wired through the bridge, we accept any M for which
// xdna_pick_swiglu_prefill_tile_m returns a valid tile, and any dims for which
// xdna_pick_swiglu_prefill_tile_n finds a valid tile_n.
// tile_k still uses IRON's default of 64.
static bool xdna_shape_dispatchable_swiglu_prefill(int64_t M_seq, int64_t K_emb,
                                                   int64_t N_hid, int num_cols) {
    if (num_cols != 4 && num_cols != 8) return false;
    if (xdna_pick_swiglu_prefill_tile_m(M_seq) == 0) return false;
    if (K_emb < 64  || K_emb % 64  != 0) return false;   // tile_k=64
    if (xdna_pick_swiglu_prefill_tile_n(K_emb, N_hid, num_cols) == 0) return false;
    if (N_hid > 32768 || K_emb > 32768) return false;
    if (N_hid % (num_cols * 2) != 0) return false;       // SiLU per-column tile
    return true;
}

// Shape-dispatchability for W8A16 SwiGLU decode. Superset of
// validate_swiglu_decode_int8_shapes in compile.py. group_size fixed at 32
// (Q8_0 block size); caller verifies the weight tensors are Q8_0-typed.
static bool xdna_shape_dispatchable_swiglu_decode_int8(int64_t K_emb, int64_t N_hid,
                                                       int num_cols, int group_size) {
    if (num_cols != 4 && num_cols != 8) return false;
    if (group_size != 32) return false;               // Q8_0 block size
    if (K_emb % group_size != 0) return false;
    if (N_hid % group_size != 0) return false;
    if (K_emb < 64  || K_emb % 64  != 0) return false;  // GEMV kernel_vector_size
    if (N_hid < 64  || N_hid % 64  != 0) return false;
    if (N_hid % (num_cols * 2) != 0) return false;    // SiLU per-column tile
    if (K_emb % num_cols      != 0) return false;
    if (N_hid > 32768 || K_emb > 32768) return false; // BD-overflow cap
    if ((N_hid / num_cols) < 8) return false;
    if ((K_emb / num_cols) < 8) return false;
    return true;
}

// Validate that (w_q, w_k, w_v, input) form a dispatchable QKV triple.
// Checks types, contiguity, 2D, K-dim consistency, M=1, shape tileability.
// Shared between the cgraph pre-scan (xdna_plan_qkv) and any future callers.
static bool xdna_validate_qkv_triple(const struct ggml_tensor * w_q,
                                      const struct ggml_tensor * w_k,
                                      const struct ggml_tensor * w_v,
                                      const struct ggml_tensor * input,
                                      int * out_num_cols) {
    for (const auto * w : {w_q, w_k, w_v}) {
        if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16 && w->type != GGML_TYPE_F16) return false;
        if (!ggml_is_contiguous(w)) return false;
        if (w->ne[2] != 1 || w->ne[3] != 1) return false;
    }
    if (input->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(input)) return false;
    if (input->ne[2] != 1 || input->ne[3] != 1) return false;

    const int64_t embedding_dim = input->ne[0];
    if (w_q->ne[0] != embedding_dim) return false;
    if (w_k->ne[0] != embedding_dim) return false;
    if (w_v->ne[0] != embedding_dim) return false;

    if (input->ne[1] < 1 || input->ne[1] > 4) return false;  // M=1..4 (loop GEMV per row)

    int num_cols = 4;
    const char * cols_env = getenv("GGML_XDNA_NUM_COLS");
    if (cols_env) num_cols = atoi(cols_env);

    if (!xdna_shape_dispatchable_qkv(embedding_dim, w_q->ne[1], w_k->ne[1], w_v->ne[1], num_cols))
        return false;

    if (out_num_cols) *out_num_cols = num_cols;
    return true;
}

// Pre-scan result: QKV triples planned for fused dispatch.
struct xdna_qkv_plan {
    // Q node idx -> (Q idx, K idx, V idx). Looked up at Q's position during
    // graph walk. The other 2 indices appear in skip_indices.
    std::unordered_map<int, std::array<int,3>> triple_at;
    // K and V node indices — skipped during main loop (already dispatched at Q).
    std::unordered_set<int> skip_indices;
};

// Scan cgraph for QKV triples: MUL_MAT nodes grouped by src[1] that form
// groups of exactly 3 and pass validation. The 3 node indices are stored in
// ascending cgraph order; we dispatch all 3 at the first (lowest-index) node
// position, then skip the other 2 when the main loop reaches them.
//
// The Qwen3.5 decode pattern places RMSNorm/view ops between Q and K/V, so a
// linear 3-consecutive-node matcher never fires. This pre-scan sidesteps that
// by grouping on shared src[1] across the whole graph.
static void xdna_plan_qkv(const struct ggml_cgraph * cgraph, xdna_qkv_plan * out) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    static std::atomic<int> dbg_remaining{dbg ? 3 : 0};  // cap verbose dumps
    const bool verbose = dbg && dbg_remaining.fetch_sub(1) > 0;

    std::unordered_map<const struct ggml_tensor *, std::vector<int>> by_src1;
    int n_mulmat = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_MUL_MAT) {
            by_src1[n->src[1]].push_back(i);
            n_mulmat++;
            if (verbose) {
                const struct ggml_tensor * w = n->src[0];
                const struct ggml_tensor * a = n->src[1];
                fprintf(stderr,
                        "ggml-xdna:   MUL_MAT[%d] w=%s(type=%d ne=[%lld,%lld]) "
                        "a=%s(type=%d ne=[%lld,%lld,%lld,%lld]) dst=%s\n",
                        i,
                        w->name[0] ? w->name : "?", w->type,
                        (long long)w->ne[0], (long long)w->ne[1],
                        a->name[0] ? a->name : "?", a->type,
                        (long long)a->ne[0], (long long)a->ne[1],
                        (long long)a->ne[2], (long long)a->ne[3],
                        n->name[0] ? n->name : "?");
            }
        }
    }
    if (verbose) {
        // Also tally op-type counts so we know *what* is in these segments.
        std::unordered_map<int, int> op_counts;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            op_counts[cgraph->nodes[i]->op]++;
        }
        fprintf(stderr, "ggml-xdna:   op breakdown:");
        for (auto & oc : op_counts) {
            fprintf(stderr, " %s=%d", ggml_op_name((enum ggml_op)oc.first), oc.second);
        }
        fprintf(stderr, "\n");
    }

    int n_groups_2 = 0, n_groups_3 = 0, n_groups_other = 0;
    int n_triples_validated = 0;
    for (auto & kv : by_src1) {
        const auto & indices = kv.second;
        if      (indices.size() == 2) n_groups_2++;
        else if (indices.size() == 3) n_groups_3++;
        else if (indices.size() != 1) n_groups_other++;

        if (verbose && indices.size() >= 2) {
            const struct ggml_tensor * src1 = kv.first;
            const struct ggml_tensor * n0 = cgraph->nodes[indices[0]];
            fprintf(stderr,
                    "ggml-xdna: QKV scan group sz=%zu src1=%p name=%s type=%d "
                    "ne=[%lld,%lld,%lld,%lld] — n0 src0: type=%d ne=[%lld,%lld]\n",
                    indices.size(), (const void*)src1,
                    src1->name[0] ? src1->name : "(unnamed)", src1->type,
                    (long long)src1->ne[0], (long long)src1->ne[1],
                    (long long)src1->ne[2], (long long)src1->ne[3],
                    n0->src[0]->type,
                    (long long)n0->src[0]->ne[0], (long long)n0->src[0]->ne[1]);
        }

        if (indices.size() != 3) continue;

        const int i_q = indices[0];
        const int i_k = indices[1];
        const int i_v = indices[2];
        const struct ggml_tensor * n_q = cgraph->nodes[i_q];
        const struct ggml_tensor * n_k = cgraph->nodes[i_k];
        const struct ggml_tensor * n_v = cgraph->nodes[i_v];

        int num_cols = 0;
        bool ok = xdna_validate_qkv_triple(n_q->src[0], n_k->src[0], n_v->src[0],
                                           n_q->src[1], &num_cols);
        if (verbose) {
            fprintf(stderr,
                    "ggml-xdna:   group-3 validate=%d q_w ne=[%lld,%lld] k_w ne=[%lld,%lld] "
                    "v_w ne=[%lld,%lld] input type=%d ne[1]=%lld contig(input)=%d\n",
                    (int)ok,
                    (long long)n_q->src[0]->ne[0], (long long)n_q->src[0]->ne[1],
                    (long long)n_k->src[0]->ne[0], (long long)n_k->src[0]->ne[1],
                    (long long)n_v->src[0]->ne[0], (long long)n_v->src[0]->ne[1],
                    n_q->src[1]->type, (long long)n_q->src[1]->ne[1],
                    (int)ggml_is_contiguous(n_q->src[1]));
        }
        if (!ok) continue;

        out->triple_at[i_q] = {i_q, i_k, i_v};
        out->skip_indices.insert(i_k);
        out->skip_indices.insert(i_v);
        n_triples_validated++;
    }

    if (dbg) {
        fprintf(stderr,
                "ggml-xdna: QKV scan: n_nodes=%d n_mul_mat=%d unique_src1=%zu "
                "groups(sz=2:%d sz=3:%d other:%d) triples_validated=%d\n",
                cgraph->n_nodes, n_mulmat, by_src1.size(),
                n_groups_2, n_groups_3, n_groups_other, n_triples_validated);
        fflush(stderr);
    }
}

// ============================================================================
// FlowKV Decode Attention — streaming decode attention with online softmax
// and fused RoPE. Single xclbin with 2-tile pipeline per KV head group.
// ============================================================================

static std::string make_flowkv_cache_key(int64_t num_heads, int64_t num_kv_heads,
                                         int64_t head_dim, int64_t seq_len,
                                         int64_t chunk_size, int num_cols) {
    char buf[256];
    snprintf(buf, sizeof(buf), "flowkv_H%lld_KV%lld_d%lld_S%lld_C%lld_%dcol",
             (long long)num_heads, (long long)num_kv_heads,
             (long long)head_dim, (long long)seq_len,
             (long long)chunk_size, num_cols);
    return std::string(buf);
}

static bool flowkv_bundle_present(const std::string & bundle_dir) {
    std::ifstream f1(bundle_dir + "/combined.xclbin");
    if (!f1.good()) return false;
    std::ifstream f2(bundle_dir + "/flowkv_decode_main.insts");
    if (!f2.good()) return false;
    return true;
}

static bool ensure_flowkv_compiled(ggml_backend_xdna_context * ctx,
                                   const std::string & cache_key,
                                   int64_t num_heads, int64_t num_kv_heads,
                                   int64_t head_dim, int64_t seq_len,
                                   int64_t chunk_size, int num_cols) {
    if (ctx->flowkv_compile_failed.count(cache_key)) return false;

    std::string bundle_dir = ctx->cache_dir + "/" + cache_key;
    if (flowkv_bundle_present(bundle_dir)) return true;

    // Invoke compile.py flowkv-decode
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet flowkv-decode"
             " --num-heads %lld --num-kv-heads %lld --head-dim %lld"
             " --seq-len %lld --chunk-size %lld --num-cols %d"
             " --out %s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)num_heads, (long long)num_kv_heads,
             (long long)head_dim, (long long)seq_len,
             (long long)chunk_size, num_cols,
             bundle_dir.c_str());

    fprintf(stderr, "ggml-xdna: compiling FlowKV decode H=%lld KV=%lld d=%lld S=%lld C=%lld cols=%d "
            "(first run, will be cached)...\n",
            (long long)num_heads, (long long)num_kv_heads,
            (long long)head_dim, (long long)seq_len,
            (long long)chunk_size, num_cols);
    fflush(stderr);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: FlowKV decode compilation failed (exit code %d)\n", ret);
        ctx->flowkv_compile_failed.insert(cache_key);
        return false;
    }
    if (!flowkv_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: FlowKV decode compilation succeeded but bundle files missing in %s\n",
                       bundle_dir.c_str());
        ctx->flowkv_compile_failed.insert(cache_key);
        return false;
    }
    fprintf(stderr, "ggml-xdna: FlowKV decode compilation complete, cached at %s\n",
            bundle_dir.c_str());
    fflush(stderr);
    return true;
}

static bool try_load_flowkv_entry(xdna_flowkv_entry & entry,
                                  ggml_backend_xdna_context * ctx,
                                  const std::string & bundle_dir) {
    std::string xclbin_path = bundle_dir + "/combined.xclbin";
    std::string insts_path  = bundle_dir + "/flowkv_decode_main.insts";

    if (!std::ifstream(xclbin_path).good() || !std::ifstream(insts_path).good()) return false;

    try {
        entry.xclbin = xrt::xclbin(xclbin_path);
        entry.hw_ctx = xrt::hw_context(ctx->device, entry.xclbin);
        entry.kernel = xrt::kernel(entry.hw_ctx, "flowkv_decode");
        entry.insts_data = read_binary_file(insts_path);
        entry.insts_bo = xrt::bo(ctx->device, entry.insts_data.size(),
                                 xrt::bo::flags::cacheable, entry.kernel.group_id(3));
        memcpy(entry.insts_bo.map<void*>(), entry.insts_data.data(), entry.insts_data.size());
        entry.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load FlowKV decode xclbin: %s\n", e.what());
        return false;
    }
    return true;
}

static xdna_flowkv_entry * get_or_load_flowkv_kernel(
        ggml_backend_xdna_context * ctx,
        int64_t num_heads, int64_t num_kv_heads,
        int64_t head_dim, int64_t seq_len,
        int64_t chunk_size, int num_cols) {
    std::string cache_key = make_flowkv_cache_key(
        num_heads, num_kv_heads, head_dim, seq_len, chunk_size, num_cols);

    {
        std::lock_guard<std::mutex> lock(ctx->cache_mutex);
        auto it = ctx->flowkv_cache.find(cache_key);
        if (it != ctx->flowkv_cache.end()) return &it->second;
    }

    std::string bundle_dir = ctx->cache_dir + "/" + cache_key;

    // Retry loop: compile if missing, then load.
    for (int attempt = 0; attempt < 2; attempt++) {
        if (!flowkv_bundle_present(bundle_dir)) {
            if (!ensure_flowkv_compiled(ctx, cache_key,
                                        num_heads, num_kv_heads, head_dim,
                                        seq_len, chunk_size, num_cols)) {
                return nullptr;
            }
        }

        xdna_flowkv_entry entry;
        entry.num_heads    = num_heads;
        entry.num_kv_heads = num_kv_heads;
        entry.head_dim     = head_dim;
        entry.seq_len      = seq_len;
        entry.chunk_size   = chunk_size;
        entry.num_cols     = num_cols;
        entry.cache_key    = cache_key;

        if (try_load_flowkv_entry(entry, ctx, bundle_dir)) {
            std::lock_guard<std::mutex> lock(ctx->cache_mutex);
            auto [ins, _] = ctx->flowkv_cache.emplace(cache_key, std::move(entry));
            fprintf(stderr, "ggml-xdna: loaded FlowKV decode kernel bundle for %s\n",
                    cache_key.c_str());
            fflush(stderr);
            return &ins->second;
        }

        // Stale cache — delete and recompile.
        if (attempt == 0) {
            fprintf(stderr, "ggml-xdna: FlowKV cache stale for %s, recompiling...\n",
                    cache_key.c_str());
            fflush(stderr);
            // Best-effort cleanup of stale bundle directory.
            std::string rm_cmd = "rm -rf \"" + bundle_dir + "\"";
            (void)system(rm_cmd.c_str());
            ctx->flowkv_compile_failed.erase(cache_key);
        }
    }
    GGML_LOG_ERROR("ggml-xdna: failed to load FlowKV decode kernel after recompile\n");
    return nullptr;
}

// ============================================================================
// Expanded decode-attention matcher for FlowKV (per-head dispatch).
//
// In ggml's expanded attention path for decode, each query head has its own
// Q@K^T MUL_MAT (M=1). For GQA (e.g. Llama 3.2 1B: 32 Q heads, 8 KV heads),
// multiple Q heads share the same K/V source (one per KV head group).
//
// Strategy: pre-scan the graph to find ALL attention MUL_MAT pairs, group
// them by KV head (shared K source), and dispatch each KV head group as
// one FlowKV call with num_heads=group_size, num_kv_heads=1, num_cols=1.
// ============================================================================

struct xdna_flowkv_head {
    int qk_idx;   // index of Q@K^T MUL_MAT
    int pv_idx;   // index of scores@V MUL_MAT
};

struct xdna_flowkv_group {
    std::vector<xdna_flowkv_head> heads;  // per-head MUL_MAT pairs
    const void * k_src_ptr;               // shared K weight pointer (KV head ID)
    int64_t head_dim;
    int64_t seq_len;
    int64_t kv_head_idx;                  // which KV head this group belongs to
};

// Pre-scan the graph for expanded decode attention patterns and group by KV head.
// Returns groups ready for per-KV-head FlowKV dispatch.
static std::vector<xdna_flowkv_group> xdna_plan_flowkv(
        const struct ggml_cgraph * cgraph) {
    std::vector<xdna_flowkv_group> groups;
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    int n = cgraph->n_nodes;

    // Map from K source pointer → group index in 'groups'.
    std::unordered_map<const void *, int> k_to_group;

    // Find all Q@K^T MUL_MATs (M=1, K=64) and their matching
    // scores@V MUL_MATs (via SOFT_MAX in between).
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        if (node->src[1]->ne[1] != 1) continue;  // not decode

        int64_t K = node->src[0]->ne[0];
        int64_t N = node->src[0]->ne[1];
        if (K != 64) continue;  // head_dim must be 64

        // Check if this looks like Q@K^T: result ne[0] = seq_len (should be > 1)
        if (node->ne[0] <= 1) continue;

        int64_t seq_len = node->ne[0];
        // N should be seq_len (single KV head).
        if (N != seq_len) continue;

        // Walk forward to find SOFT_MAX and then scores@V MUL_MAT.
        int softmax_idx = -1;
        for (int j = i + 1; j < n && j < i + 6; j++) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_SOFT_MAX) { softmax_idx = j; break; }
            if (op == GGML_OP_SCALE || op == GGML_OP_ADD) continue;
            break;
        }
        if (softmax_idx < 0) continue;

        int pv_idx = -1;
        for (int j = softmax_idx + 1; j < n && j < softmax_idx + 6; j++) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_MUL_MAT) {
                if (cgraph->nodes[j]->src[1]->ne[1] != 1) break;
                pv_idx = j;
                break;
            }
            if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                op == GGML_OP_CONT || op == GGML_OP_PERMUTE) continue;
            break;
        }
        if (pv_idx < 0) continue;

        // Validate PV shape: src[0] should have same head_dim.
        if (cgraph->nodes[pv_idx]->src[0]->ne[0] != K) continue;

        const void * k_ptr = node->src[0]->data;

        // Group by K source pointer (same KV head = same K weight).
        auto it = k_to_group.find(k_ptr);
        if (it == k_to_group.end()) {
            int gidx = (int)groups.size();
            k_to_group[k_ptr] = gidx;
            xdna_flowkv_group grp;
            grp.k_src_ptr = k_ptr;
            grp.head_dim = K;
            grp.seq_len = seq_len;
            grp.kv_head_idx = gidx;
            groups.push_back(grp);
            it = k_to_group.find(k_ptr);
        }
        groups[it->second].heads.push_back({i, pv_idx});
    }

    if (dbg && !groups.empty()) {
        size_t total_heads = 0;
        for (auto & g : groups) total_heads += g.heads.size();
        fprintf(stderr, "ggml-xdna: FlowKV plan: %zu KV head groups, %zu total heads\n",
                groups.size(), total_heads);
        for (size_t g = 0; g < groups.size(); g++) {
            fprintf(stderr, "  group[%zu]: kv_head=%lld seq_len=%lld heads=%zu\n",
                    g, (long long)groups[g].kv_head_idx,
                    (long long)groups[g].seq_len, groups[g].heads.size());
        }
        fflush(stderr);
    }
    return groups;
}

// Per-KV-head FlowKV dispatch. Processes one KV head group (group_size Q heads
// sharing the same K/V cache) with num_kv_heads=1, num_cols=1.
//
// For Llama 3.2 1B: 8 dispatches per layer (one per KV head), each processing
// 4 Q heads (group_size=4).
static bool ggml_backend_xdna_flowkv_per_head(
        ggml_backend_xdna_context * ctx,
        const xdna_flowkv_group & group,
        const struct ggml_cgraph * cgraph) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;

    if (!ctx || !ctx->device_valid) return false;

    int64_t group_size = (int64_t)group.heads.size();
    int64_t head_dim = group.head_dim;
    int64_t seq_len = group.seq_len;
    int64_t num_heads = group_size;      // Q heads in this group
    int64_t num_kv_heads = 1;            // one KV head per dispatch
    int chunk_size = 32;
    int num_cols = 1;

    if (head_dim != 64) return false;
    if (seq_len % chunk_size != 0) return false;

    xdna_flowkv_entry * entry = get_or_load_flowkv_kernel(
        ctx, num_heads, num_kv_heads, head_dim, seq_len, chunk_size, num_cols);
    if (!entry) return false;

    try {
        // Allocate BOs on first use.
        if (!entry->bo_kv) {
            size_t kv_size = 1 * seq_len * 2 * head_dim * sizeof(uint16_t);
            entry->bo_kv = std::make_unique<xrt::bo>(
                xrt::bo(ctx->device, kv_size, xrt::bo::flags::host_only,
                        entry->kernel.group_id(0)));
        }
        if (!entry->bo_q) {
            size_t q_size = 1 * (group_size * head_dim + head_dim) * sizeof(uint16_t);
            entry->bo_q = std::make_unique<xrt::bo>(
                xrt::bo(ctx->device, q_size, xrt::bo::flags::host_only,
                        entry->kernel.group_id(1)));
        }
        if (!entry->bo_out) {
            size_t out_size = group_size * head_dim * sizeof(uint16_t);
            entry->bo_out = std::make_unique<xrt::bo>(
                xrt::bo(ctx->device, out_size, xrt::bo::flags::host_only,
                        entry->kernel.group_id(2)));
        }

        // --- Prepare KV cache buffer (interleaved K and V for one KV head) ---
        // DDR layout: (1, seq_len, 2, head_dim) flattened
        // For each position: K row then V row.
        {
            auto kv_ptr = entry->bo_kv->map<char *>();
            // Use the first head's K/V sources (all heads in group share same K/V).
            struct ggml_tensor * k_tensor = cgraph->nodes[group.heads[0].qk_idx]->src[0];
            struct ggml_tensor * v_tensor = cgraph->nodes[group.heads[0].pv_idx]->src[0];
            const char * k_data = (const char *)k_tensor->data;
            const char * v_data = (const char *)v_tensor->data;
            size_t row_bytes = head_dim * sizeof(uint16_t);

            for (int64_t pos = 0; pos < seq_len; pos++) {
                size_t dst_k = pos * 2 * row_bytes;
                size_t src_k = pos * row_bytes;
                memcpy(kv_ptr + dst_k, k_data + src_k, row_bytes);

                size_t dst_v = (pos * 2 + 1) * row_bytes;
                size_t src_v = pos * row_bytes;
                memcpy(kv_ptr + dst_v, v_data + src_v, row_bytes);
            }
            entry->bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // --- Prepare Q buffer (packed Q + identity RoPE angles) ---
        // DDR layout: [Q_group (gs*hd) | angles (hd)]
        // Q is already rotated by the graph, so pass identity angles.
        {
            auto q_ptr = entry->bo_q->map<char *>();
            size_t q_head_bytes = head_dim * sizeof(uint16_t);

            // Identity angles: cos=1.0 (bf16 0x3F80), sin=0.0 (bf16 0x0000)
            char angle_cos_bf16[2] = {0x80, 0x3F};
            char angle_sin_bf16[2] = {0x00, 0x00};

            // Copy Q for each head in the group.
            for (int64_t g = 0; g < group_size; g++) {
                struct ggml_tensor * q_tensor =
                    cgraph->nodes[group.heads[g].qk_idx]->src[1];
                const char * q_data = (const char *)q_tensor->data;
                memcpy(q_ptr + g * q_head_bytes, q_data, q_head_bytes);
            }

            // Write identity angles after Q data.
            size_t angles_off = group_size * q_head_bytes;
            for (int d = 0; d < head_dim; d += 2) {
                memcpy(q_ptr + angles_off + d * 2, angle_cos_bf16, 2);
                memcpy(q_ptr + angles_off + d * 2 + 2, angle_sin_bf16, 2);
            }
            entry->bo_q->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // --- Dispatch ---
        auto run = xrt::run(entry->kernel);
        run.set_arg(0, *entry->bo_kv);
        run.set_arg(1, *entry->bo_q);
        run.set_arg(2, *entry->bo_out);
        run.set_arg(3, entry->insts_bo);

        {
            std::lock_guard<std::mutex> lock(*entry->mu);
            auto t0 = std::chrono::steady_clock::now();
            run.start();
            auto state = run.wait(30000);
            auto t1 = std::chrono::steady_clock::now();

            if (state != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("ggml-xdna: FlowKV per-head dispatch failed, state=%d\n",
                               (int)state);
                return false;
            }

            if (dbg) {
                float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                fprintf(stderr, "ggml-xdna: FlowKV KV=%lld gs=%lld S=%lld → %.2f ms\n",
                        (long long)group.kv_head_idx, (long long)group_size,
                        (long long)seq_len, ms);
                fflush(stderr);
            }
        }

        // --- Read back output and scatter to per-head result tensors ---
        entry->bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        {
            auto out_ptr = entry->bo_out->map<char *>();
            size_t out_head_bytes = head_dim * sizeof(uint16_t);
            for (int64_t g = 0; g < group_size; g++) {
                struct ggml_tensor * out_tensor =
                    cgraph->nodes[group.heads[g].pv_idx];
                memcpy(out_tensor->data, out_ptr + g * out_head_bytes,
                       out_head_bytes);
            }
        }

        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: FlowKV per-head exception: %s\n", e.what());
        return false;
    }
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;
    int n = cgraph->n_nodes;

    // Reset attention-prefill per-phase profiling samples (no-op when gate off).
    xdna_attn_prof_reset();

    static const bool debug = getenv("XDNA_DEBUG") != NULL;
    static const bool attention_prefill_dbg_enabled = getenv("XDNA_ENABLE_ATTENTION_PREFILL") != NULL;
    static const bool transformer_block_dbg_enabled =
        (getenv("XDNA_ENABLE_TRANSFORMER_BLOCK") != NULL) ||
        (getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL);
    if (debug) {
        int n_mulmat = 0, n_mulmat_disp = 0;
        int n_glu = 0, n_glu_swiglu = 0, n_swiglu_window = 0, n_swiglu_match = 0;
        int n_attn_window = 0, n_attn_match = 0;
        int n_tblock_window = 0, n_tblock_match = 0;
        for (int i = 0; i < n; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT) {
                n_mulmat++;
                if (xdna_node_npu_dispatchable(node)) n_mulmat_disp++;
            }
            if (node->op == GGML_OP_GLU) {
                n_glu++;
                if (ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU) n_glu_swiglu++;
            }
            // Look for the structural 4-node window MM, MM, GLU, MM
            if (i + 3 < n) {
                if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT &&
                    cgraph->nodes[i+1]->op == GGML_OP_MUL_MAT &&
                    cgraph->nodes[i+2]->op == GGML_OP_GLU &&
                    cgraph->nodes[i+3]->op == GGML_OP_MUL_MAT) {
                    n_swiglu_window++;
                    xdna_swiglu_match m{};
                    if (xdna_try_match_swiglu(cgraph, i, &m)) n_swiglu_match++;
                }
            }
            // Attention-prefill window: each RMS_NORM is a candidate start.
            // Count both the window count (pattern anchors scanned) and
            // full matches that walked through to the residual ADD.
            if (attention_prefill_dbg_enabled && node->op == GGML_OP_RMS_NORM) {
                n_attn_window++;
                xdna_attention_match am{};
                if (xdna_try_match_attention_prefill(cgraph, i, &am)) n_attn_match++;
            }
            if (transformer_block_dbg_enabled && node->op == GGML_OP_RMS_NORM) {
                n_tblock_window++;
                xdna_transformer_block_match tm{};
                if (xdna_try_match_transformer_block_prefill(cgraph, i, &tm)) n_tblock_match++;
            }
        }
        fprintf(stderr,
                "ggml-xdna: graph_compute n_nodes=%d mul_mat=%d npu_dispatchable=%d "
                "glu=%d swiglu=%d swiglu_window=%d swiglu_match=%d "
                "attn_window=%d attn_match=%d "
                "tblock_window=%d tblock_match=%d\n",
                n, n_mulmat, n_mulmat_disp,
                n_glu, n_glu_swiglu, n_swiglu_window, n_swiglu_match,
                n_attn_window, n_attn_match,
                n_tblock_window, n_tblock_match);
        fflush(stderr);
    }

    // Sweep the graph. Walk nodes in order: collect runs of consecutive
    // CPU-bound nodes and delegate them as one cgraph view; break out to
    // dispatch each NPU-capable MUL_MAT individually. View ops are skipped
    // (included in the CPU run — CPU handles them as no-ops but keeping them
    // in-range preserves sched invariants).
    static const bool swiglu_enabled = getenv("XDNA_ENABLE_SWIGLU") != NULL;
    static const bool qkv_enabled    = getenv("XDNA_ENABLE_QKV")    != NULL;
    static const bool rms_norm_enabled = getenv("XDNA_ENABLE_RMS_NORM") != NULL;
    static const bool attention_prefill_enabled = getenv("XDNA_ENABLE_ATTENTION_PREFILL") != NULL;
    static const bool transformer_block_enabled =
        (getenv("XDNA_ENABLE_TRANSFORMER_BLOCK") != NULL) ||
        (getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL);
    static const bool flowkv_decode_enabled = getenv("XDNA_ENABLE_FLOWKV_DECODE") != NULL;

    // Pre-scan for QKV triples. Llama.cpp's Qwen3.5 decode interleaves
    // RMSNorm/view ops between Q and K/V MUL_MATs, so a 3-consecutive-node
    // matcher never fires. We group MUL_MATs by shared src[1] activation
    // across the whole graph instead. Dispatch happens at the Q node
    // position; K/V are skipped when the main loop reaches them.
    xdna_qkv_plan qkv_plan;
    if (qkv_enabled) {
        xdna_plan_qkv(cgraph, &qkv_plan);
        static const bool qkv_dbg = getenv("XDNA_DEBUG") != NULL;
        if (qkv_dbg) {
            fprintf(stderr, "ggml-xdna: QKV plan: %zu triples (%zu skip nodes)\n",
                    qkv_plan.triple_at.size(), qkv_plan.skip_indices.size());
            fflush(stderr);
        }
    }

    // Pre-scan for decode GEMV batching. Identifies standalone MUL_MAT M=1
    // nodes eligible for runlist batching (gated by XDNA_ENABLE_DECODE_BATCH).
    xdna_decode_batcher decode_batcher;
    std::unordered_set<int> decode_batch_indices;
    if (decode_batcher.is_enabled()) {
        decode_batch_indices = xdna_plan_decode_batch(cgraph, qkv_plan);
    }

    // Bulk pre-warm all attention-prefill weights for this cgraph before the
    // per-node loop walks it serially. Without this, each of the N attention
    // layers' first dispatch pays ~44ms of host-side transpose+DMA on the
    // critical path. Uploads run in parallel across CPU cores; the per-weight
    // cache check in the dispatch path then finds each BO already resident
    // and becomes a pointer lookup. Guarded per-cgraph so repeated prefills
    // on the same cgraph ptr don't re-scan.
    if (attention_prefill_enabled) {
        // No per-cgraph guard: ggml-backend-sched reuses cgraph pointers across
        // calls with different shapes (warmup probe at M=2, then real prefill).
        // The scan itself is cheap (~33 RMS_NORMs); the per-weight cache check
        // inside attn_prefill_warm_gemm_weight skips already-uploaded weights.
        attn_prefill_bulk_prewarm(ctx, cgraph);
    }
    if (transformer_block_enabled && getenv("XDNA_ENABLE_TBLOCK_FUSED") == NULL) {
        // L3A prewarm uploads per-kernel weight BOs that the fused path
        // doesn't use. Skip it in fused mode — fused weights are written
        // at first dispatch per layer, into the single input_bo.
        tblock_prefill_bulk_prewarm(ctx, cgraph);
    }
    if (transformer_block_enabled && getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL) {
        // Fused path: upload all per-layer weights/gains into each entry's
        // input_bo in parallel before the first dispatch of this cgraph,
        // eliminating the first-batch 27→150 t/s cliff.
        tblock_fused_bulk_prewarm(ctx, cgraph);
    }

    int cpu_run_start = -1;
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        // Determine if this node is a batchable decode GEMV. If NOT, flush
        // any pending batched GEMVs first — the current node may depend on
        // their results (sequential graph dependency).
        const bool is_batchable_gemv = decode_batcher.is_enabled()
            && decode_batch_indices.count(i)
            && node->op == GGML_OP_MUL_MAT
            && node->src[1]->ne[1] == 1;
        if (!is_batchable_gemv && !decode_batcher.empty()) {
            decode_batcher.flush(ctx);
        }

        // QKV triple: dispatch Q+K+V as one xrt::runlist at Q's position.
        // K/V indices come later in the linear walk and are in skip_indices.
        if (qkv_enabled) {
            auto it = qkv_plan.triple_at.find(i);
            if (it != qkv_plan.triple_at.end()) {
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }
                const auto & trip = it->second;
                struct ggml_tensor * q_mm = cgraph->nodes[trip[0]];
                struct ggml_tensor * k_mm = cgraph->nodes[trip[1]];
                struct ggml_tensor * v_mm = cgraph->nodes[trip[2]];
                ggml_backend_xdna_mul_mat_qkv(
                    ctx, q_mm, k_mm, v_mm,
                    q_mm->src[0], k_mm->src[0], v_mm->src[0],
                    q_mm->src[1]);
                continue;  // natural ++i; intermediate CPU nodes still run via accumulator
            }
            if (qkv_plan.skip_indices.count(i)) {
                // K or V — already dispatched as part of its QKV triple at Q's
                // position. Flush any pending CPU run so delegate_range's
                // contiguous view stops just before this node, then skip it.
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }
                continue;
            }
        }

        // FlowKV decode attention: per-KV-head dispatch.
        // Pre-scan builds groups of per-head attention MUL_MAT pairs.
        // Each group is dispatched as one FlowKV call (num_kv_heads=1).
        // We check here if the current node is a Q@K^T MUL_MAT that was
        // already dispatched as part of a FlowKV group.
        if (flowkv_decode_enabled) {
            static std::vector<xdna_flowkv_group> flowkv_groups;
            static std::unordered_set<int> flowkv_dispatched_qk;
            static std::unordered_set<int> flowkv_dispatched_pv;
            static const struct ggml_cgraph * flowkv_last_cgraph = nullptr;

            // Build plan once per cgraph.
            if (flowkv_last_cgraph != cgraph) {
                flowkv_groups = xdna_plan_flowkv(cgraph);
                flowkv_dispatched_qk.clear();
                flowkv_dispatched_pv.clear();
                flowkv_last_cgraph = cgraph;

                // Dispatch all groups eagerly before the main loop.
                for (auto & grp : flowkv_groups) {
                    if (cpu_run_start >= 0) {
                        // Flush CPU run before the first head in this group.
                        ggml_status s = xdna_delegate_range(
                            ctx, cgraph, cpu_run_start, grp.heads[0].qk_idx);
                        if (s != GGML_STATUS_SUCCESS) return s;
                        cpu_run_start = -1;
                    }
                    bool ok = ggml_backend_xdna_flowkv_per_head(ctx, grp, cgraph);
                    if (ok) {
                        for (auto & h : grp.heads) {
                            flowkv_dispatched_qk.insert(h.qk_idx);
                            flowkv_dispatched_pv.insert(h.pv_idx);
                            // Also mark intermediate nodes (SCALE, ADD, SOFT_MAX)
                            // between QK and PV as dispatched.
                            for (int j = h.qk_idx + 1; j < h.pv_idx; j++) {
                                enum ggml_op op = cgraph->nodes[j]->op;
                                if (op == GGML_OP_SCALE || op == GGML_OP_ADD ||
                                    op == GGML_OP_SOFT_MAX) {
                                    flowkv_dispatched_qk.insert(j);
                                }
                            }
                        }
                    } else if (debug) {
                        fprintf(stderr, "ggml-xdna: FlowKV group kv=%lld failed, "
                                "falling back to CPU\n",
                                (long long)grp.kv_head_idx);
                        fflush(stderr);
                    }
                }
            }

            // Skip nodes that were dispatched by FlowKV.
            if (flowkv_dispatched_qk.count(i) || flowkv_dispatched_pv.count(i)) {
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }
                continue;
            }
        }

        // TransformerBlockPrefill matcher — tries to consume the entire
        // attention+FFN range as one chained xclbin dispatch. Takes precedence
        // over the attention-prefill branch below when XDNA_ENABLE_TRANSFORMER_BLOCK
        // is set. Same Phase-C sub-graph filter pattern but extended over the
        // FFN nodes as well.
        //
        // Layer 4B multi-layer path: if this index starts an N-block group
        // (registered during bulk_prewarm), dispatch the grouped N-block ELF
        // and skip all 16*N nodes in one shot. Phase-C is extended over the
        // full range. Tail indices fall through to the single-block path.
        if (transformer_block_enabled && node->op == GGML_OP_RMS_NORM) {
            // Layer 4B multi-layer group lookup — only when the opt-in
            // env var is set, so N=1 users pay zero cost. The groups map
            // is only populated when XDNA_ENABLE_TBLOCK_FUSED_N>1 during
            // bulk_prewarm.
            static const int tbf_multi_N = xdna_tblock_fused_group_N();
            const xdna_tblock_fused_group * group_ptr = nullptr;
            if (tbf_multi_N > 1) {
                std::lock_guard<std::mutex> lock(ctx->cache_mutex);
                auto cg_it = ctx->tblock_fused_groups_per_cgraph.find(cgraph);
                if (cg_it != ctx->tblock_fused_groups_per_cgraph.end()) {
                    auto g_it = cg_it->second.find(i);
                    if (g_it != cg_it->second.end()) {
                        group_ptr = &g_it->second;
                    }
                }
            }

            if (group_ptr != nullptr) {
                static const bool tp_dbg = getenv("XDNA_DEBUG") != NULL;
                const auto & group = *group_ptr;
                if (tp_dbg) {
                    fprintf(stderr,
                        "ggml-xdna: tblock-fused N=%d group @%d..%d (seq=%lld)\n",
                        group.N, group.range_begin, group.range_end,
                        (long long)group.tms.front().attn.seq_len);
                    fflush(stderr);
                }

                // Phase-C validation over the full N-block range. Same
                // whitelist as single-block; we require at least one
                // side-effect op (SET_ROWS/CPY/DUP — KV-cache writes) per
                // block, otherwise full-CPU fallback.
                const int range_begin = group.range_begin;
                const int range_end   = group.range_end;
                bool unknown_op_in_range = false;
                int total_in_range = range_end - range_begin + 1;
                int side_effect_count = 0;
                for (int j = range_begin; j <= range_end; j++) {
                    enum ggml_op op = cgraph->nodes[j]->op;
                    switch (op) {
                        case GGML_OP_VIEW:
                        case GGML_OP_RESHAPE:
                        case GGML_OP_CONT:
                        case GGML_OP_PERMUTE:
                        case GGML_OP_TRANSPOSE:
                        case GGML_OP_MUL_MAT:
                        case GGML_OP_RMS_NORM:
                        case GGML_OP_NORM:
                        case GGML_OP_MUL:
                        case GGML_OP_ROPE:
                        case GGML_OP_FLASH_ATTN_EXT:
                        case GGML_OP_ADD:
                        case GGML_OP_GLU:
                            break;
                        case GGML_OP_SET_ROWS:
                        case GGML_OP_CPY:
                        case GGML_OP_DUP:
                            side_effect_count++;
                            break;
                        default:
                            unknown_op_in_range = true;
                            break;
                    }
                    if (unknown_op_in_range) break;
                }
                const bool use_phase_c = !unknown_op_in_range && side_effect_count > 0;

                if (!use_phase_c) {
                    // Run the whole range on CPU; skip the NPU dispatch.
                    if (tp_dbg) {
                        fprintf(stderr,
                            "ggml-xdna: tblock-fused N=%d PhaseB fallback @%d..%d (%s)\n",
                            group.N, range_begin, range_end,
                            unknown_op_in_range ? "unknown_op" : "no_side_effects");
                        fflush(stderr);
                    }
                    if (cpu_run_start < 0) cpu_run_start = i;
                    ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                       cpu_run_start,
                                                       range_end + 1);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                    i = range_end;
                    continue;
                }

                // Multi-block strategy: run the FULL range on CPU to
                // populate all KV caches + intermediate residuals correctly,
                // then dispatch NPU and OVERWRITE only the final block's
                // residual output. This is perf-conservative (CPU does
                // all the work the NPU would) but correctness-safe — the
                // NPU overwrite only affects the final y. If the NPU
                // output matches CPU within bf16 rounding, downstream
                // nodes (next group's inpL, lm_head, etc.) get the NPU
                // value and the multi-block path is proven end-to-end.
                //
                // Once stable we can revisit Phase-C style interleaving
                // to drop the CPU-side matmul cost; the earlier attempt
                // tripped ggml-alloc aliasing + backend coherence issues
                // that need deeper investigation.
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                       cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }

                {
                    ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                       range_begin,
                                                       range_end + 1);
                    if (s != GGML_STATUS_SUCCESS) return s;
                }

                const bool dispatched =
                    ggml_backend_xdna_tblock_fused_multi(ctx, group, cgraph);
                if (dispatched) {
                    // Overwrite only final y (L = N-1). Intermediate
                    // y_L values were already computed by CPU and are
                    // in the right tensors; skipping their overwrite
                    // avoids the alias hazard that corrupted memory
                    // in the earlier interleaved approach.
                    if (!tblock_fused_multi_copy_y(group, group.N - 1, cgraph)) {
                        if (tp_dbg) {
                            fprintf(stderr,
                                "ggml-xdna: tblock-fused N=%d final y copy-back failed @%d..%d; "
                                "keeping CPU result as authoritative\n",
                                group.N, range_begin, range_end);
                            fflush(stderr);
                        }
                    }
                } else if (tp_dbg) {
                    fprintf(stderr,
                        "ggml-xdna: tblock-fused N=%d dispatch failed @%d..%d; "
                        "CPU result is authoritative\n",
                        group.N, range_begin, range_end);
                    fflush(stderr);
                }

                i = range_end;
                continue;
            }

            xdna_transformer_block_match tm{};
            if (xdna_try_match_transformer_block_prefill(cgraph, i, &tm)
                && tm.attn.seq_len >= 256) {
                static const bool tp_dbg = getenv("XDNA_DEBUG") != NULL;
                const auto & am = tm.attn;
                if (tp_dbg) {
                    fprintf(stderr,
                        "ggml-xdna: transformer_block matched @%d..%d "
                        "(seq=%lld embed=%lld H=%lld KV=%lld d=%lld F=%lld) "
                        "rms=%d qmm=%d kmm=%d vmm=%d fa=%d omm=%d attn_add=%d "
                        "ffn_norm=%d gate=%d up=%d glu=%d down=%d ffn_add=%d\n",
                        am.rms_norm_idx, tm.ffn_residual_add_idx,
                        (long long)am.seq_len, (long long)am.embed_dim,
                        (long long)am.num_heads, (long long)am.num_kv_heads,
                        (long long)am.head_dim, (long long)tm.ffn_hidden_dim,
                        am.rms_norm_idx, am.q_proj_idx, am.k_proj_idx,
                        am.v_proj_idx, am.attn_core_idx, am.o_proj_idx,
                        am.residual_add_idx, tm.ffn_norm_idx,
                        tm.gate_mm_idx, tm.up_mm_idx, tm.glu_idx,
                        tm.down_mm_idx, tm.ffn_residual_add_idx);
                    fflush(stderr);
                }

                static const bool tp_skip_dispatch =
                    getenv("XDNA_TBLOCK_PREFILL_SKIP_DISPATCH") != NULL;
                static const bool tp_force_full_cpu =
                    getenv("XDNA_TBLOCK_PREFILL_FULL_CPU") != NULL;

                const int range_end = tm.ffn_residual_add_idx;
                bool unknown_op_in_range = false;
                int total_in_range = range_end - am.rms_norm_idx + 1;
                int side_effect_count = 0;
                for (int j = am.rms_norm_idx; j <= range_end; j++) {
                    enum ggml_op op = cgraph->nodes[j]->op;
                    switch (op) {
                        case GGML_OP_VIEW:
                        case GGML_OP_RESHAPE:
                        case GGML_OP_CONT:
                        case GGML_OP_PERMUTE:
                        case GGML_OP_TRANSPOSE:
                        case GGML_OP_MUL_MAT:
                        case GGML_OP_RMS_NORM:
                        case GGML_OP_NORM:
                        case GGML_OP_MUL:
                        case GGML_OP_ROPE:
                        case GGML_OP_FLASH_ATTN_EXT:
                        case GGML_OP_ADD:
                        case GGML_OP_GLU:
                            break;
                        case GGML_OP_SET_ROWS:
                        case GGML_OP_CPY:
                        case GGML_OP_DUP:
                            side_effect_count++;
                            break;
                        default:
                            unknown_op_in_range = true;
                            break;
                    }
                    if (unknown_op_in_range) break;
                }

                const bool use_phase_c = !tp_force_full_cpu && !unknown_op_in_range
                                         && side_effect_count > 0;

                if (!use_phase_c) {
                    if (tp_dbg) {
                        const char * why =
                            tp_force_full_cpu   ? "force_full_cpu" :
                            unknown_op_in_range ? "unknown_op"    :
                                                  "no_side_effects";
                        fprintf(stderr,
                            "ggml-xdna: tblock-prefill PhaseB fallback @%d..%d (%s)\n",
                            am.rms_norm_idx, range_end, why);
                        fflush(stderr);
                    }
                    if (cpu_run_start < 0) cpu_run_start = i;
                    ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                       cpu_run_start,
                                                       range_end + 1);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                } else {
                    if (cpu_run_start >= 0) {
                        ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                           cpu_run_start, i);
                        if (s != GGML_STATUS_SUCCESS) return s;
                        cpu_run_start = -1;
                    }
                    const size_t graph_cap = 512;
                    struct ggml_init_params p = {
                        /*.mem_size   =*/ ggml_tensor_overhead() * graph_cap +
                                          ggml_graph_overhead_custom(graph_cap, false),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    struct ggml_context * sub_ctx = ggml_init(p);
                    if (sub_ctx == nullptr) {
                        if (tp_dbg) {
                            fprintf(stderr,
                                "ggml-xdna: tblock-prefill ggml_init failed @%d..%d, "
                                "falling back to Phase B\n",
                                am.rms_norm_idx, range_end);
                            fflush(stderr);
                        }
                        ggml_status s = xdna_delegate_range(ctx, cgraph, i,
                                                           range_end + 1);
                        if (s != GGML_STATUS_SUCCESS) return s;
                    } else {
                        struct ggml_cgraph * sub = ggml_new_graph_custom(
                            sub_ctx, graph_cap, false);
                        const int npu_skip[] = {
                            am.q_proj_idx,
                            am.q_rope_idx,
                            am.attn_core_idx,
                            am.o_proj_idx,
                            am.residual_add_idx,
                            tm.ffn_norm_idx,
                            tm.ffn_norm_mul_idx,
                            tm.gate_mm_idx,
                            tm.up_mm_idx,
                            tm.glu_idx,
                            tm.down_mm_idx,
                            tm.ffn_residual_add_idx,
                        };
                        auto is_npu_skipped = [&](int j) -> bool {
                            for (int k = 0; k < (int)(sizeof(npu_skip)/sizeof(npu_skip[0])); k++) {
                                if (npu_skip[k] == j) return true;
                            }
                            return false;
                        };
                        for (int j = am.rms_norm_idx; j <= range_end; j++) {
                            if (is_npu_skipped(j)) continue;
                            if (sub->n_nodes >= (int)sub->size) break;
                            sub->nodes[sub->n_nodes++] = cgraph->nodes[j];
                        }
                        const int sub_n = ggml_graph_n_nodes(sub);
                        if (tp_dbg) {
                            fprintf(stderr,
                                "ggml-xdna: tblock-prefill PhaseC @%d..%d "
                                "(side_effects=%d sub_nodes=%d total_in_range=%d "
                                "npu_skipped=%d)\n",
                                am.rms_norm_idx, range_end,
                                side_effect_count, sub_n, total_in_range,
                                total_in_range - sub_n);
                            fflush(stderr);
                        }
                        if (sub_n > 0) {
                            ggml_status s = ggml_backend_graph_compute(
                                ctx->cpu_backend, sub);
                            if (s != GGML_STATUS_SUCCESS) {
                                ggml_free(sub_ctx);
                                return s;
                            }
                        }
                        ggml_free(sub_ctx);
                    }
                }

                if (tp_skip_dispatch) {
                    i = range_end;
                    continue;
                }
                static const bool tp_fused_enabled =
                    getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL;
                const bool dispatched = tp_fused_enabled
                    ? ggml_backend_xdna_transformer_block_prefill_fused(ctx, tm, cgraph)
                    : ggml_backend_xdna_transformer_block_prefill(ctx, tm, cgraph);
                if (!dispatched) {
                    if (tp_dbg) {
                        fprintf(stderr,
                            "ggml-xdna: tblock-prefill dispatch failed @%d..%d, "
                            "running full range on CPU as recovery\n",
                            am.rms_norm_idx, range_end);
                        fflush(stderr);
                    }
                    if (use_phase_c) {
                        ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                           am.rms_norm_idx,
                                                           range_end + 1);
                        if (s != GGML_STATUS_SUCCESS) return s;
                    }
                }
                i = range_end;
                continue;
            }
        }

        // Attention-prefill matcher (Phase A — scaffolding). We only log on
        // match and then fall through: the matched nodes continue to CPU via
        // the normal per-node loop / CPU accumulator. Dispatch is intentionally
        // deferred to Phase B. This keeps the model producing correct output
        // while we prove the matcher fires reliably on real graphs.
        if (attention_prefill_enabled && node->op == GGML_OP_RMS_NORM) {
            xdna_attention_match am{};
            if (xdna_try_match_attention_prefill(cgraph, i, &am) && am.seq_len >= 256) {
                static const bool ap_dbg = getenv("XDNA_DEBUG") != NULL;
                if (ap_dbg) {
                    fprintf(stderr,
                        "ggml-xdna: attention_prefill matched @%d..%d "
                        "(seq=%lld embed=%lld H=%lld KV=%lld d=%lld) "
                        "rms=%d mul=%d Q=%d K=%d V=%d qrope=%d krope=%d fa=%d O=%d add=%d\n",
                        am.rms_norm_idx, am.residual_add_idx,
                        (long long)am.seq_len, (long long)am.embed_dim,
                        (long long)am.num_heads, (long long)am.num_kv_heads,
                        (long long)am.head_dim,
                        am.rms_norm_idx, am.attn_norm_mul_idx,
                        am.q_proj_idx, am.k_proj_idx, am.v_proj_idx,
                        am.q_rope_idx, am.k_rope_idx,
                        am.attn_core_idx, am.o_proj_idx, am.residual_add_idx);
                    fflush(stderr);
                }
                // Phase C: skip the redundant compute-only CPU work. The
                // matched range [rms_norm_idx..residual_add_idx] contains
                // ~24 nodes; the NPU dispatch reproduces the EXPENSIVE work
                // (Q_MM, Q_ROPE, FLASH_ATTN_EXT, O_MM, residual ADD) and
                // writes the post-residual output directly into
                // cgraph->nodes[residual_add_idx]->data. The remaining
                // CPU-side responsibility is the SET_ROWS / CPY nodes that
                // write K_rope / V into the KV cache for future decode steps,
                // plus their true dependencies (RMS_NORM, gain MUL,
                // K_MM, V_MM, K_ROPE, plus view/reshape glue).
                //
                // We build a sub-graph of just the side-effect nodes and let
                // ggml_build_forward_expand pull in deps transitively. The
                // sub-graph lives in a temporary ggml_context whose nodes
                // are POINTERS into the main cgraph (no_alloc=true). This is
                // the same pattern xdna_delegate_range uses via
                // ggml_graph_view, generalized to non-contiguous selection.
                //
                // Safety net: env var XDNA_ATTN_PREFILL_FULL_CPU=1 falls
                // back to Phase B's full-range CPU delegation. Also: if the
                // matched range contains an unexpected op (something not
                // VIEW/RESHAPE/CONT/PERMUTE/TRANSPOSE/MUL_MAT/RMS_NORM/NORM/
                // MUL/ROPE/FLASH_ATTN_EXT/ADD/SET_ROWS/CPY/DUP), we
                // conservatively fall back to full-range CPU.
                static const bool skip_dispatch =
                    getenv("XDNA_ATTN_PREFILL_SKIP_DISPATCH") != NULL;
                static const bool force_full_cpu =
                    getenv("XDNA_ATTN_PREFILL_FULL_CPU") != NULL;

                // Op-set whitelist scan over the matched range. Anything
                // outside this set means we don't understand the block well
                // enough to drop nodes safely — fall back to full CPU.
                bool unknown_op_in_range = false;
                int total_in_range = am.residual_add_idx - am.rms_norm_idx + 1;
                int side_effect_count = 0;
                for (int j = am.rms_norm_idx; j <= am.residual_add_idx; j++) {
                    enum ggml_op op = cgraph->nodes[j]->op;
                    switch (op) {
                        case GGML_OP_VIEW:
                        case GGML_OP_RESHAPE:
                        case GGML_OP_CONT:
                        case GGML_OP_PERMUTE:
                        case GGML_OP_TRANSPOSE:
                        case GGML_OP_MUL_MAT:
                        case GGML_OP_RMS_NORM:
                        case GGML_OP_NORM:
                        case GGML_OP_MUL:
                        case GGML_OP_ROPE:
                        case GGML_OP_FLASH_ATTN_EXT:
                        case GGML_OP_ADD:
                            break;
                        case GGML_OP_SET_ROWS:
                        case GGML_OP_CPY:
                        case GGML_OP_DUP:
                            side_effect_count++;
                            break;
                        default:
                            unknown_op_in_range = true;
                            break;
                    }
                    if (unknown_op_in_range) break;
                }

                const bool use_phase_c = !force_full_cpu && !unknown_op_in_range
                                         && side_effect_count > 0;

                if (!use_phase_c) {
                    // Phase B fallback: run the full matched range on CPU,
                    // then NPU overwrites the residual-ADD output. Correct
                    // but slow (double compute).
                    if (ap_dbg) {
                        const char * why =
                            force_full_cpu      ? "force_full_cpu" :
                            unknown_op_in_range ? "unknown_op"    :
                                                  "no_side_effects";
                        fprintf(stderr,
                            "ggml-xdna: attn-prefill PhaseB fallback @%d..%d (%s)\n",
                            am.rms_norm_idx, am.residual_add_idx, why);
                        fflush(stderr);
                    }
                    if (cpu_run_start < 0) cpu_run_start = i;
                    ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                       cpu_run_start,
                                                       am.residual_add_idx + 1);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                } else {
                    // Phase C: flush CPU accumulator for nodes BEFORE the
                    // matched range, then run only the side-effect sub-graph.
                    if (cpu_run_start >= 0) {
                        ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                           cpu_run_start, i);
                        if (s != GGML_STATUS_SUCCESS) return s;
                        cpu_run_start = -1;
                    }
                    // Build a temporary cgraph holding pointers to nodes from
                    // the matched range only. We DO NOT call
                    // ggml_build_forward_expand — that walks parents
                    // transitively and would pull in the entire prior-layer
                    // chain (inpL → previous residual → ... → embeddings),
                    // which both blows past the cap and would re-execute prior
                    // layers, corrupting the residual stream.
                    //
                    // Instead: manually append every node in
                    // [rms_norm_idx, residual_add_idx] EXCEPT the ones the NPU
                    // dispatch reproduces (Q_MM, Q_ROPE, FLASH_ATTN_EXT, O_MM,
                    // residual ADD). All true dependencies for the K/V
                    // side-effects (RMS_NORM, gain MUL, K_MM, V_MM, K_ROPE)
                    // live within this range and execute in original topo
                    // order. View/reshape/cont/cpy glue is harmless to run on
                    // already-computed tensors.
                    //
                    // Manual append safety: ggml_new_graph_custom allocates
                    // cgraph->nodes as a plain ggml_tensor** array sized to
                    // `cap`, with n_nodes=0. Direct write +n_nodes++ matches
                    // the ggml_graph_view pattern (which sets n_nodes without
                    // populating hash_set / use_counts and is consumed
                    // directly by CPU backend graph_compute). No ref counting,
                    // no leafs, no hash bookkeeping required for execution.
                    const size_t graph_cap = 256;  // ~24 nodes typical, headroom
                    struct ggml_init_params p = {
                        /*.mem_size   =*/ ggml_tensor_overhead() * graph_cap +
                                          ggml_graph_overhead_custom(graph_cap, false),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    struct ggml_context * sub_ctx = ggml_init(p);
                    if (sub_ctx == nullptr) {
                        // OOM on the tiny scratch ctx — extremely unlikely.
                        // Fall back to Phase B.
                        if (ap_dbg) {
                            fprintf(stderr,
                                "ggml-xdna: attn-prefill ggml_init failed @%d..%d, "
                                "falling back to Phase B\n",
                                am.rms_norm_idx, am.residual_add_idx);
                            fflush(stderr);
                        }
                        ggml_status s = xdna_delegate_range(ctx, cgraph, i,
                                                           am.residual_add_idx + 1);
                        if (s != GGML_STATUS_SUCCESS) return s;
                    } else {
                        struct ggml_cgraph * sub = ggml_new_graph_custom(
                            sub_ctx, graph_cap, false);
                        // Indices the NPU dispatch reproduces — must NOT be
                        // re-executed on CPU.
                        const int npu_skip[] = {
                            am.q_proj_idx,
                            am.q_rope_idx,
                            am.attn_core_idx,
                            am.o_proj_idx,
                            am.residual_add_idx,
                        };
                        auto is_npu_skipped = [&](int j) -> bool {
                            for (int k = 0; k < (int)(sizeof(npu_skip)/sizeof(npu_skip[0])); k++) {
                                if (npu_skip[k] == j) return true;
                            }
                            return false;
                        };
                        for (int j = am.rms_norm_idx; j <= am.residual_add_idx; j++) {
                            if (is_npu_skipped(j)) continue;
                            // Bounds check defensive — graph_cap=256 >> 24.
                            if (sub->n_nodes >= (int)sub->size) break;
                            sub->nodes[sub->n_nodes++] = cgraph->nodes[j];
                        }
                        const int sub_n = ggml_graph_n_nodes(sub);
                        if (ap_dbg) {
                            fprintf(stderr,
                                "ggml-xdna: attn-prefill PhaseC @%d..%d "
                                "(side_effects=%d sub_nodes=%d total_in_range=%d "
                                "npu_skipped=%d)\n",
                                am.rms_norm_idx, am.residual_add_idx,
                                side_effect_count, sub_n, total_in_range,
                                total_in_range - sub_n);
                            fflush(stderr);
                        }
                        if (sub_n > 0) {
                            ggml_status s = ggml_backend_graph_compute(
                                ctx->cpu_backend, sub);
                            if (s != GGML_STATUS_SUCCESS) {
                                ggml_free(sub_ctx);
                                return s;
                            }
                        }
                        ggml_free(sub_ctx);
                    }
                }

                // NPU dispatch — overwrites residual_add_idx output.
                if (skip_dispatch) {
                    i = am.residual_add_idx;
                    continue;
                }
                if (!ggml_backend_xdna_attention_prefill(ctx, am, cgraph)) {
                    // Dispatch failed. In Phase B the CPU output covers us;
                    // in Phase C we skipped the compute, so the residual
                    // tensor data is stale. Recover by running the full
                    // range on CPU now.
                    if (ap_dbg) {
                        fprintf(stderr,
                            "ggml-xdna: attn-prefill dispatch failed @%d..%d, "
                            "running full range on CPU as recovery\n",
                            am.rms_norm_idx, am.residual_add_idx);
                        fflush(stderr);
                    }
                    if (use_phase_c) {
                        ggml_status s = xdna_delegate_range(ctx, cgraph,
                                                           am.rms_norm_idx,
                                                           am.residual_add_idx + 1);
                        if (s != GGML_STATUS_SUCCESS) return s;
                    }
                }
                i = am.residual_add_idx;  // for-loop ++i lands on next node
                continue;
            }
        }

        // First try the fused SwiGLU pattern — matches node[i..i+3] as a
        // gate/up MUL_MAT + SwiGLU GLU + down MUL_MAT chain and collapses
        // it to a single chained-xclbin dispatch.
        if (swiglu_enabled) {
            xdna_swiglu_match m{};
            if (xdna_try_match_swiglu(cgraph, i, &m)) {
                // Before committing to dispatch, check tile-dispatchability for
                // paths that don't pad M. INT8 prefill requires M % 64 == 0
                // (tile_m >= 16 → M % (tile_m*4) == 0). Partial micro-batches
                // from llama.cpp's prefill tail (e.g. M=46 for a 1500-token
                // prompt at -ub 256) cannot tile and must fall through to
                // the normal per-node loop so they reach CPU cleanly. Without
                // this gate, INT8 compile fails mid-dispatch and the pattern's
                // 4 nodes produce uninitialized output.
                bool can_dispatch = true;
                if (m.is_int8) {
                    const int64_t M_match = m.input->ne[1];
                    if (M_match > 1 && M_match % 64 != 0) can_dispatch = false;
                }
                if (!can_dispatch) {
                    // Not consumed — let normal dispatch + CPU accumulator handle these.
                    if (cpu_run_start < 0) cpu_run_start = i;
                    // Do not advance i here; the for-loop ++i + per-node logic
                    // will process each of the 4 match nodes. Falls through to
                    // the CPU accumulator because individual Q8_0 matmuls /
                    // GLU are not xdna_node_npu_dispatchable.
                    continue;
                }
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }
                if (m.is_int8) {
                    // INT8 SwiGLU not available in IRON-windows build —
                    // compile.py raises NotImplementedError for INT8 kernels.
                    // Fall through to the bf16 path instead of crashing.
                    fprintf(stderr, "ggml-xdna: INT8 SwiGLU not available in IRON-windows build, using bf16 fallback\n");
                }
                {
                    ggml_backend_xdna_mul_mat_swiglu(
                        ctx,
                        m.gate_mm, m.up_mm, m.glu, m.down_mm,
                        m.gate_w, m.up_w, m.down_w, m.input,
                        ctx->num_cols);
                }
                i += 3;  // for-loop ++i then lands on the node after down_mm
                continue;
            }
        }

        // Standalone RMSNorm dispatch (opt-in via XDNA_ENABLE_RMS_NORM=1).
        // ggml_backend_xdna_rms_norm() itself gates on the env var and the
        // node shape/dtype/eps; on any rejection it returns false and the node
        // accumulates into the next CPU-delegated range.
        if (rms_norm_enabled && node->op == GGML_OP_RMS_NORM) {
            if (cpu_run_start >= 0) {
                ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                if (s != GGML_STATUS_SUCCESS) return s;
                cpu_run_start = -1;
            }
            if (ggml_backend_xdna_rms_norm(ctx, node)) {
                continue;
            }
            // Dispatch rejected the node — fall it into the CPU accumulator.
            if (cpu_run_start < 0) cpu_run_start = i;
            continue;
        }

        if (xdna_node_npu_dispatchable(node)) {
            // Batchable decode GEMV: collect in the batcher instead of
            // dispatching individually. The batcher will flush these as
            // a single xrt::runlist when a non-batchable node is reached.
            if (is_batchable_gemv) {
                if (cpu_run_start >= 0) {
                    ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                    if (s != GGML_STATUS_SUCCESS) return s;
                    cpu_run_start = -1;
                }
                // Ensure kernel is compiled and entry is loaded.
                const int64_t K = node->src[0]->ne[0];
                const int64_t N = node->src[0]->ne[1];
                int num_cols = ctx->num_cols;
                std::string cache_key = make_cache_key(XDNA_OP_GEMV, 1, K, N, "bf16", num_cols);
                if (ensure_compiled(ctx, cache_key, XDNA_OP_GEMV, 1, K, N, "bf16", num_cols)) {
                    xdna_kernel_entry * entry = get_or_load_kernel(ctx, cache_key, XDNA_OP_GEMV, 1, K, N);
                    if (entry) {
                        decode_batcher.add(entry, node, i);
                        continue;
                    }
                }
                // Fallback: if compile/load failed, dispatch individually.
                if (debug) {
                    fprintf(stderr, "ggml-xdna: decode_batch fallback individual @%d\n", i);
                    fflush(stderr);
                }
            }
            // Flush any pending batched GEMVs before individual dispatch.
            if (!decode_batcher.empty()) {
                decode_batcher.flush(ctx);
            }
            // Flush any pending CPU run up to this point.
            if (cpu_run_start >= 0) {
                ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                if (s != GGML_STATUS_SUCCESS) return s;
                cpu_run_start = -1;
            }
            // Dispatch to NPU.
            ggml_backend_xdna_mul_mat(ctx, node);
            continue;
        }

        // View-only nodes are pure metadata — they still need to be in the
        // CPU run (CPU treats them as no-ops), so we just extend the range.
        if (cpu_run_start < 0) cpu_run_start = i;
    }

    // Flush any remaining batched decode GEMVs before the trailing CPU run.
    if (!decode_batcher.empty()) {
        decode_batcher.flush(ctx);
    }

    // Flush trailing CPU run.
    if (cpu_run_start >= 0) {
        ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, n);
        if (s != GGML_STATUS_SUCCESS) return s;
    }

    // Print per-phase attention-prefill profile (no-op when gate off / no samples).
    xdna_attn_prof_print();

    return GGML_STATUS_SUCCESS;

    GGML_UNUSED(backend);
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ============================================================================
// Backend init
// ============================================================================

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0xad, 0x1a, 0xfe, 0x42, 0xab, 0xcd, 0x00, 0x01,
                              0x58, 0x44, 0x4e, 0x41, 0x32, 0x00, 0x00, 0x00 };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_context * ctx = new ggml_backend_xdna_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

// ============================================================================
// Device interface
// ============================================================================

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return "XDNA";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    return "AMD XDNA NPU";

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    size_t ram = (size_t)status.ullTotalPhys;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    size_t ram = (pages > 0 && page_size > 0) ? (size_t)pages * (size_t)page_size : (8ULL << 30);
#endif
    *total = ram / 2;
    *free  = *total;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xdna_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

// ============================================================================
// Buffer type — owns weight allocations so scheduler routes ops to us
// ============================================================================

static const char * ggml_backend_xdna_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "XDNA";
    GGML_UNUSED(buft);
}

static void ggml_backend_xdna_buffer_free(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_xdna_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void ggml_backend_xdna_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);
    g_set_tensor_bytes.fetch_add(size, std::memory_order_relaxed);
    g_set_tensor_calls.fetch_add(1, std::memory_order_relaxed);
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
}

static void ggml_backend_xdna_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);
    g_get_tensor_bytes.fetch_add(size, std::memory_order_relaxed);
    g_get_tensor_calls.fetch_add(1, std::memory_order_relaxed);
    GGML_UNUSED(buffer);
}

static void ggml_backend_xdna_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_xdna_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        size_t nbytes = ggml_nbytes(src);
        memcpy(dst->data, src->data, nbytes);
        g_cpy_tensor_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        g_cpy_tensor_calls.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void ggml_backend_xdna_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i ggml_backend_xdna_buffer_i = {
    /* .free_buffer     = */ ggml_backend_xdna_buffer_free,
    /* .get_base        = */ ggml_backend_xdna_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_xdna_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_xdna_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_xdna_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_xdna_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_xdna_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_xdna_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size_t aligned_size = (size + 63) & ~size_t(63);
    void * data = aligned_alloc(64, aligned_size);
    if (!data) {
        GGML_LOG_ERROR("ggml-xdna: failed to allocate %zu bytes\n", size);
        return NULL;
    }
    static const bool debug = getenv("XDNA_DEBUG") != NULL;
    if (debug) {
        fprintf(stderr, "ggml-xdna: alloc_buffer size=%zu (%.1f MiB)\n", size, size / 1048576.0);
        fflush(stderr);
    }
    return ggml_backend_buffer_init(buft, ggml_backend_xdna_buffer_i, data, size);
}

static size_t ggml_backend_xdna_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;
    GGML_UNUSED(buft);
}

static bool ggml_backend_xdna_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    // NPU memory IS host RAM (shared via DMA). is_host signals tensor-
    // placement compatibility, not who runs the op — supports_op still gates
    // execution separately. Advertising true lets CPU operate on our
    // tensors without a scheduler split at every CPU↔XDNA boundary
    // (graph INPUTs like inp_pos / inp_out_ids / kq_mask would otherwise
    // fragment the decoder graph into per-layer sub-graphs).
    return true;
    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name      = */ ggml_backend_xdna_buffer_type_get_name,
            /* .alloc_buffer  = */ ggml_backend_xdna_buffer_type_alloc_buffer,
            /* .get_alignment = */ ggml_backend_xdna_buffer_type_get_alignment,
            /* .get_max_size  = */ NULL,
            /* .get_alloc_size= */ NULL,
            /* .is_host       = */ ggml_backend_xdna_buffer_type_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    if (buft.device == NULL) {
        buft.device = ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0);
    }
    return &buft;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_xdna_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

// NPU dispatchability: returns true only if the shape tiles cleanly on the NPU.
// Kernel invariants (aie_kernels/aie2p/mm.cc bf16_f32 4x8x8 MAC):
//   r=4 → tile_m % (2*r) == 0  → tile_m >= 8,  M = tile_m*4 so min M = 32.
//   t=8 → tile_n % (2*t) == 0  → tile_n >= 16, N = tile_n*cols so min N = 16*cols.
// aie2 4x8x4 bf16 has equivalent t=4, n%(4*t)==0 → same tile_n >= 16 floor.
// Small-N matmuls (e.g. hybrid-architecture SSM projections at N<128) fail
// these constraints and must stay on CPU; without this gate, dispatch picks
// tile_n=8 and aiecc's clang static_asserts on the MAC kernel.
// Cap N at 32768 to skip vocab projection (248320) — aiecc BD overflow at
// N-tiles per shim >~485 (248320 = 2^9*5*97, can't tile friendlier without
// N-chunking in the backend).
static bool xdna_shape_dispatchable(int64_t M, int64_t K, int64_t N) {
    if (M < 32 || K < 32 || N < 32 || N > 32768) return false;
    int num_cols = 4;
    const char * cols_env = getenv("GGML_XDNA_NUM_COLS");
    if (cols_env) num_cols = atoi(cols_env);

    const int64_t tiles_m[] = {64, 32, 16, 8};
    const int64_t tiles_k[] = {64, 32, 16, 8};
    const int64_t tiles_n[] = {64, 32, 16};   // min tile_n=16 for bf16 (both aie2/aie2p)
    bool m_ok = false, k_ok = false, n_ok = false;
    for (int i = 0; i < 4 && !m_ok; i++) if (M % (tiles_m[i] * 4)        == 0) m_ok = true;
    for (int i = 0; i < 4 && !k_ok; i++) if (K %  tiles_k[i]             == 0) k_ok = true;
    for (int i = 0; i < 3 && !n_ok; i++) if (N % (tiles_n[i] * num_cols) == 0) n_ok = true;
    return m_ok && k_ok && n_ok;
}

// GEMV (M=1 decode) dispatchability. IRON GEMV constraints:
//   K % kernel_vector_size == 0  (default 64)
//   N % num_cols == 0, per_col = N/num_cols >= 8, (per_col % tile_out) == 0 for
//   some tile_out <= per_col. With the candidate set in compile.py
//   select_gemv_tiles, any per_col >= 8 with per_col a power-of-two multiple works.
// Same vocab-proj N cap applies (BD-overflow territory).
static bool xdna_shape_dispatchable_gemv(int64_t K, int64_t N) {
    static const bool gemv_enabled = getenv("XDNA_ENABLE_GEMV") != NULL;
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;

    if (!gemv_enabled) return false;
    int num_cols = 4;
    const char * cols_env = getenv("GGML_XDNA_NUM_COLS");
    if (cols_env) num_cols = atoi(cols_env);

    if (K < 64 || K % 64 != 0) {
        if (dbg) fprintf(stderr, "ggml-xdna: gemv reject: K=%lld (must be >=64 and mod 64)\n", (long long)K);
        return false;
    }
    if (N < num_cols || N % num_cols != 0) {
        if (dbg) fprintf(stderr, "ggml-xdna: gemv reject: N=%lld (must be mod %d)\n", (long long)N, num_cols);
        return false;
    }
    const int64_t per_col = N / num_cols;
    if (per_col < 8) {
        if (dbg) fprintf(stderr, "ggml-xdna: gemv reject: per_col=%lld (must be >=8)\n", (long long)per_col);
        return false;
    }
    if (N > 32768) return false;
    return true;
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    static const bool rms_norm_enabled = getenv("XDNA_ENABLE_RMS_NORM") != NULL;
    // Liberal claim list (OpenVINO-style): the scheduler aborts if no backend
    // claims an op. We claim everything we can plausibly run, then decide
    // NPU-vs-CPU inside graph_compute. Unclaimable ops (e.g. training-only)
    // fall through to `default: false`.
    switch (op->op) {
        // Metadata / no-op — always claim.
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        // MUL_MAT: keep type/contig gates (these fail the whole graph if we lie);
        // dispatchability is decided per-node in graph_compute.
        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (!ggml_is_contiguous(src0)) return false;
            if (!ggml_is_contiguous(src1)) return false;
            if (src0->ne[2] != 1 || src0->ne[3] != 1) return false;
            if (src1->ne[2] != 1 || src1->ne[3] != 1) return false;
            if (src1->type != GGML_TYPE_F32) return false;
            // Q8_0 weights are accepted when any Q8_0-capable composite gate
            // is active:
            //   - XDNA_ENABLE_SWIGLU_INT8  → standalone int8 SwiGLU dispatch
            //   - tblock+W8A16             → fused tblock consumes Q8_0 via
            //     tblock_fused_upload_w8a16_weight's Q8_0 branch.
            // The dispatch layer never runs a bare Q8_0 MUL_MAT on the NPU
            // (no standalone gemv_int8 wiring yet) — bare Q8_0 mm falls back
            // to CPU inside graph_compute. Claiming here keeps the scheduler
            // from splitting the attention / FFN pattern across backends.
            static const bool int8_ok = getenv("XDNA_ENABLE_SWIGLU_INT8") != NULL;
            static const bool tblock_w8a16_ok =
                (getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL) &&
                (getenv("XDNA_ENABLE_TBLOCK_FUSED_W8A16") != NULL);
            if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) return true;
            if ((int8_ok || tblock_w8a16_ok) && src0->type == GGML_TYPE_Q8_0) return true;
            return false;
        }

        // Elementwise arithmetic.
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SUB:
        case GGML_OP_DIV:
            return true;

        // Normalization.
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
            return rms_norm_enabled;

        // Scaling.
        case GGML_OP_SCALE:
            return true;

        // Copy / contiguity fixes.
        case GGML_OP_CPY:
        case GGML_OP_CONT:
        case GGML_OP_DUP:
            return true;

        // Indexing (embeddings, KV cache writes).
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            return true;

        // Attention-adjacent.
        case GGML_OP_ROPE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;

        // Activations — umbrella ops covering SILU/GELU/RELU/... and SWIGLU/GEGLU/...
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
            return true;

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Strict: only claim our own buffer type. The scheduler uses this to decide
    // where tensors live; claiming host buffers too would make us indistinguishable
    // from CPU and break routing.
    return buft == ggml_backend_xdna_buffer_type();

    GGML_UNUSED(dev);
}

// Scheduler hint: deprioritize us for shapes that NPU can't dispatch.
// Ops still come to us (we own weight buffers), but this tells the scheduler
// that for shapes where we'd just fall back to CPU, there's no speed win.
static bool ggml_backend_xdna_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // QKV fusion mode (also used for attention-prefill Phase A matcher
    // scaffolding): be aggressive about keeping ops here so the whole
    // transformer attention block (RMSNorm → Q/K/V → RoPE → FlashAttn → O →
    // residual add) lands in a single graph segment. Without this, the
    // scheduler splits at every CPU-bound intermediate (VIEW/RESHAPE/NORM/
    // RoPE/FLASH_ATTN_EXT) and multi-node patterns never form. The claimed
    // op list is identical whether QKV or attention-prefill is enabled —
    // the matcher in graph_compute decides which pattern to fire.
    static const bool qkv_enabled                 = getenv("XDNA_ENABLE_QKV") != NULL;
    static const bool attention_prefill_enabled   = getenv("XDNA_ENABLE_ATTENTION_PREFILL") != NULL;
    static const bool transformer_block_enabled   =
        (getenv("XDNA_ENABLE_TRANSFORMER_BLOCK") != NULL) ||
        (getenv("XDNA_ENABLE_TBLOCK_FUSED") != NULL);
    static const bool flowkv_decode_enabled       = getenv("XDNA_ENABLE_FLOWKV_DECODE") != NULL;
    if (qkv_enabled || attention_prefill_enabled || transformer_block_enabled || flowkv_decode_enabled) {
        // Gate aggressive non-MUL_MAT claims on prefill shape. During decode
        // (seq=1), claiming intermediates like ADD/MUL/RMS_NORM/ROPE just
        // fragments scheduling with delegate_range round-trips and has no
        // fused-pattern benefit (the matcher gates on seq>=256). Heuristic:
        // look at this op's ne[1]; fall back to src[0]->ne[1]. Prefill ops
        // carry the seq dim here; decode ops carry 1.
        //
        // FlowKV decode is an exception: it needs the scheduler to keep
        // expanded-attention intermediates (SCALE, ADD mask, SOFT_MAX)
        // in the same segment during decode (seq dim in ne[0] of src[0]).
        auto is_prefill_op = [](const struct ggml_tensor * op) -> bool {
            int64_t seq = op->ne[1];
            if (seq <= 1 && op->src[0] != nullptr) seq = op->src[0]->ne[1];
            return seq >= 32;
        };
        // For FlowKV: check if this op is part of an expanded attention
        // decode pattern. The key dimension is in src[0]->ne[0] (head_dim=64)
        // or src[0]->ne[1] (seq_len). If seq_len > 1 and M=1, it's decode attn.
        auto is_flowkv_decode_op = [](const struct ggml_tensor * op) -> bool {
            if (op->src[0] == nullptr) return false;
            int64_t dim1 = op->src[0]->ne[1];
            // SOFT_MAX result: ne[0]=seq_len, ne[1]=1 for decode
            if (op->op == GGML_OP_SOFT_MAX) {
                return op->ne[0] > 1 && op->ne[1] == 1;
            }
            // SCALE/ADD on attention scores: src carries seq_len dim
            if (op->op == GGML_OP_SCALE || op->op == GGML_OP_ADD) {
                return dim1 > 1 || (op->src[1] != nullptr && op->src[1]->ne[1] > 1);
            }
            return false;
        };
        switch (op->op) {
            case GGML_OP_MUL_MAT: {
                const int64_t K = op->src[0]->ne[0];
                const int64_t N = op->src[0]->ne[1];
                const int64_t M = op->src[1]->ne[1];
                if (M == 1) {
                    // For FlowKV: claim M=1 MUL_MATs where K=head_dim=64
                    // and N=seq_len (Q@K^T) or N=seq_len (scores@V).
                    // These are the attention score and output computations.
                    if (flowkv_decode_enabled && K == 64 && N > 1) {
                        return true;
                    }
                    // Aggressive M=1 claim only makes sense for QKV (decode
                    // matmul fusion). Attention-prefill and transformer-block
                    // are prefill-only; claiming M=1 MUL_MATs for them
                    // fragments decode scheduling without benefit.
                    if (!qkv_enabled) return false;
                    if (K < 64 || K % 64 != 0) return false;
                    if (N < 8  || N > 32768)   return false;
                    if (N % 8 != 0)            return false;
                    return true;
                }
                return xdna_shape_dispatchable(M, K, N);
            }
            // Intermediate ops between Q/K/V and around the attention core —
            // keep them here DURING PREFILL so the scheduler doesn't split.
            // graph_compute's CPU fallback delegates them via
            // xdna_delegate_range (cheap during prefill). During decode,
            // return false to let them stay on CPU natively.
            // For FlowKV: also claim SCALE, ADD, SOFT_MAX in decode attention.
            case GGML_OP_SCALE:
            case GGML_OP_ADD:
            case GGML_OP_SOFT_MAX:
                if (flowkv_decode_enabled && is_flowkv_decode_op(op)) return true;
                return is_prefill_op(op);
            case GGML_OP_MUL:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_RMS_NORM:
            case GGML_OP_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_DUP:
            case GGML_OP_ROPE:
            case GGML_OP_FLASH_ATTN_EXT:
            case GGML_OP_UNARY:
            case GGML_OP_GLU:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                return is_prefill_op(op);
            default:
                return false;
        }
    }

    // Default path (QKV disabled): only claim MUL_MATs with dispatchable shape.
    if (op->op != GGML_OP_MUL_MAT) return false;
    const int64_t K = op->src[0]->ne[0];
    const int64_t N = op->src[0]->ne[1];
    const int64_t M = op->src[1]->ne[1];
    if (M == 1) return xdna_shape_dispatchable_gemv(K, N);
    return xdna_shape_dispatchable(M, K, N);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ ggml_backend_xdna_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================================
// Backend registration
// ============================================================================

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return "XDNA";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xdna_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
