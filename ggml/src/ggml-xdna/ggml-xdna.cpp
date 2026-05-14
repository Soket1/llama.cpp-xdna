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

// Forward declaration: env-flag helper used throughout the file.
// Returns true when the env var is set AND is not "0", "OFF", or "off".
static bool xdna_env_enabled(const char * name);

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

        // Diagnostic: print GEMM kernel group_ids
        fprintf(stderr, "ggml-xdna: GEMM kernel group_ids:");
        for (int a = 0; a < 10; a++) {
            try { fprintf(stderr, " [%d]=%zu", a, (size_t)entry.kernel.group_id(a)); }
            catch (...) { fprintf(stderr, " [%d]=ERR", a); break; }
        }
        fprintf(stderr, " key=%s\n", cache_key.c_str());
        fflush(stderr);

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
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);
        entry.kernel = xrt::kernel(entry.hw_ctx, "MLIR_AIE");
        entry.insts_data = read_binary_file(insts_path);
        entry.insts_bo = xrt::bo(ctx->device, entry.insts_data.size(),
                                 xrt::bo::flags::cacheable, entry.kernel.group_id(1));
        memcpy(entry.insts_bo.map<void*>(), entry.insts_data.data(), entry.insts_data.size());
        entry.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // Diagnostic: print all group_ids
        fprintf(stderr, "ggml-xdna: FlowKV kernel loaded, group_ids:");
        for (int a = 0; a < 10; a++) {
            try { fprintf(stderr, " [%d]=%zu", a, (size_t)entry.kernel.group_id(a)); }
            catch (...) { fprintf(stderr, " [%d]=ERR", a); break; }
        }
        fprintf(stderr, "\n");
        fflush(stderr);
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
        // Skip if a previous load attempt failed — don't recompile.
        if (ctx->flowkv_compile_failed.count(cache_key)) return nullptr;
    }

    std::string bundle_dir = ctx->cache_dir + "/" + cache_key;

    // Retry loop: compile if missing, then load.
    for (int attempt = 0; attempt < 2; attempt++) {
        if (!flowkv_bundle_present(bundle_dir)) {
            if (!ensure_flowkv_compiled(ctx, cache_key,
                                        num_heads, num_kv_heads, head_dim,
                                        seq_len, chunk_size, num_cols)) {
                ctx->flowkv_compile_failed.insert(cache_key);
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
            // Use _rmdir or platform-independent approach (rm -rf is Unix-only).
#ifdef _WIN32
            std::string rm_cmd = "rmdir /s /q \"" + bundle_dir + "\"";
#else
            std::string rm_cmd = "rm -rf \"" + bundle_dir + "\"";
#endif
            (void)system(rm_cmd.c_str());
            ctx->flowkv_compile_failed.erase(cache_key);
        }
    }
    GGML_LOG_ERROR("ggml-xdna: failed to load FlowKV decode kernel after recompile\n");
    ctx->flowkv_compile_failed.insert(cache_key);
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

// Pending Q@K^T head from a previous segment where scores@V was not found.
// Stored across graph_compute calls to enable cross-segment matching.
struct xdna_flowkv_pending {
    int qk_idx;            // Q@K^T index in original segment (for skip marking)
    int softmax_idx;       // SOFT_MAX index in original segment
    const void * k_data;   // K cache data pointer (for KV head grouping)
    const void * v_data;   // V cache data pointer
    const void * q_data;   // Q data pointer
    int64_t head_dim;
    int64_t seq_len;
};

struct xdna_flowkv_group {
    std::vector<xdna_flowkv_head> heads;  // per-head MUL_MAT pairs
    const void * k_src_ptr;               // shared K weight pointer (KV head ID)
    int64_t head_dim;
    int64_t seq_len;
    int64_t kv_head_idx;                  // which KV head this group belongs to
    // Cross-segment tensor pointers. When non-NULL, dispatch uses these
    // directly instead of looking up tensors by cgraph index.
    const struct ggml_tensor * k_tensor;  // K cache tensor
    const struct ggml_tensor * v_tensor;  // V cache tensor
    const struct ggml_tensor * q_tensor;  // Q tensor
};

// Pre-scan the graph for expanded decode attention patterns and group by KV head.
// Returns groups ready for per-KV-head FlowKV dispatch.
//
// Handles cross-segment matching: when the graph scheduler splits the attention
// pattern across segments (Q@K^T + SOFT_MAX in one segment, scores@V in the
// next), pending Q@K^T heads are stored and matched with scores@V in a
// subsequent call.
static std::vector<xdna_flowkv_group> xdna_plan_flowkv(
        const struct ggml_cgraph * cgraph) {
    std::vector<xdna_flowkv_group> groups;
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    static const bool sched_dbg = getenv("XNA_SCHED_DEBUG") != NULL;
    int n = cgraph->n_nodes;

    // Persistent state: pending Q@K^T heads from previous segments where
    // scores@V was not found (graph split across segments).
    static std::vector<xdna_flowkv_pending> pending_heads;
    static const struct ggml_cgraph * pending_cgraph = nullptr;
    if (pending_cgraph != cgraph) {
        pending_heads.clear();
        pending_cgraph = cgraph;
    }

    // XNA_SCHED_DEBUG: dump all nodes in this segment for FlowKV analysis
    if (sched_dbg) {
        fprintf(stderr, "  FlowKV plan scan: %d nodes in segment\n", n);
        for (int i = 0; i < n; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            fprintf(stderr, "    [%3d] %-12s %-25s", i, ggml_op_name(node->op),
                node->name ? node->name : "(null)");
            if (node->op == GGML_OP_MUL_MAT) {
                fprintf(stderr, "  K=%lld N=%lld M=%lld",
                    (long long)node->src[0]->ne[0],
                    (long long)node->src[0]->ne[1],
                    (long long)node->src[1]->ne[1]);
            }
            fprintf(stderr, "\n");
        }
        fflush(stderr);
    }

    // Map from K source pointer → group index in 'groups'.
    std::unordered_map<const void *, int> k_to_group;

    // ── Phase 1: Cross-segment matching ──────────────────────────────
    // Check if any MUL_MAT in this segment is the scores@V for a pending
    // Q@K^T from a previous segment.
    //
    // In the expanded attention path, scores@V is a batched MUL_MAT that
    // produces the concatenated output for all heads:
    //   src[0] = V cache [model_dim, seq_len], src[1] = scores [seq_len, 1]
    //   output = [model_dim, 1]  where model_dim = head_dim * q_heads_per_kv
    //
    // So we match on: MUL_MAT, M=1, ne[0] = model_dim (not head_dim).
    if (!pending_heads.empty()) {
        for (int i = 0; i < n; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT) continue;
            if (node->src[1]->ne[1] != 1) continue;
            if (node->ne[0] <= 1) continue;

            // Check if this is the batched scores@V:
            //   output ne[0] = model_dim (= V cache width = src[0]->ne[0])
            //   src[0] ne[1] = seq_len
            //   output ne[0] > head_dim (it's the full model dim, not per-head)
            int64_t out_dim = node->ne[0];
            int64_t K_dim = node->src[0]->ne[0];
            int64_t N_dim = node->src[0]->ne[1];
            bool matched_any = false;
            for (auto & ph : pending_heads) {
                if (out_dim > ph.head_dim && out_dim == K_dim &&
                    N_dim == ph.seq_len) {
                    matched_any = true;
                    break;
                }
            }
            if (!matched_any) continue;

            // Match found: group pending heads by k_data (KV head identity).
            std::unordered_map<const void *, std::vector<int>> k_to_pending;
            for (int pi = 0; pi < (int)pending_heads.size(); pi++) {
                if (pending_heads[pi].head_dim == out_dim) {
                    k_to_pending[pending_heads[pi].k_data].push_back(pi);
                }
            }

            // Build FlowKV groups from pending heads.
            for (auto & [k_ptr, pidxs] : k_to_pending) {
                auto it = k_to_group.find(k_ptr);
                if (it == k_to_group.end()) {
                    int gidx = (int)groups.size();
                    k_to_group[k_ptr] = gidx;
                    xdna_flowkv_group grp;
                    grp.k_src_ptr = k_ptr;
                    grp.head_dim = pending_heads[pidxs[0]].head_dim;
                    grp.seq_len = pending_heads[pidxs[0]].seq_len;
                    grp.kv_head_idx = gidx;
                    // Cross-segment: store tensor pointers from pending heads.
                    // K and Q come from the original segment (pending_heads).
                    // V comes from the current segment (scores@V's src[0]).
                    grp.k_tensor = nullptr;  // will use k_data pointer directly
                    grp.v_tensor = node->src[0];  // V cache from scores@V
                    grp.q_tensor = nullptr;  // will use q_data pointer directly
                    groups.push_back(grp);
                    it = k_to_group.find(k_ptr);
                }
                for (int pi : pidxs) {
                    groups[it->second].heads.push_back(
                        {pending_heads[pi].qk_idx, i});
                }
            }

            if (dbg) {
                fprintf(stderr, "ggml-xdna: FlowKV cross-segment match: "
                        "scores@V at [%d] matched %zu pending heads\n",
                        i, pending_heads.size());
                fflush(stderr);
            }

            // Remove matched pending heads.
            std::vector<xdna_flowkv_pending> remaining;
            for (int pi = 0; pi < (int)pending_heads.size(); pi++) {
                if (!(out_dim > pending_heads[pi].head_dim &&
                      out_dim == K_dim && N_dim == pending_heads[pi].seq_len))
                    remaining.push_back(pending_heads[pi]);
            }
            pending_heads = std::move(remaining);
            break;  // one scores@V match per segment is enough
        }
    }

    // ── Phase 2: Same-segment matching ───────────────────────────────
    // Find Q@K^T → SOFT_MAX → scores@V within this segment.
    // Queue unmatched Q@K^T as pending for cross-segment matching.
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        if (node->src[1]->ne[1] != 1) {
            if (sched_dbg) fprintf(stderr, "    FlowKV reject [%d] MUL_MAT: M=%lld != 1 (not decode)\n", i, (long long)node->src[1]->ne[1]);
            continue;
        }

        int64_t K = node->src[0]->ne[0];
        int64_t N = node->src[0]->ne[1];
        if (K != 64) {
            if (sched_dbg) fprintf(stderr, "    FlowKV reject [%d] MUL_MAT: K=%lld != 64\n", i, (long long)K);
            continue;
        }

        // Check if this looks like Q@K^T: result ne[0] = seq_len (should be > 1)
        if (node->ne[0] <= 1) {
            if (sched_dbg) fprintf(stderr, "    FlowKV reject [%d] MUL_MAT: ne[0]=%lld <= 1\n", i, (long long)node->ne[0]);
            continue;
        }

        int64_t seq_len = node->ne[0];
        // N should be seq_len (single KV head).
        if (N != seq_len) {
            if (sched_dbg) fprintf(stderr, "    FlowKV reject [%d] MUL_MAT: N=%lld != seq_len=%lld\n", i, (long long)N, (long long)seq_len);
            continue;
        }

        // Walk forward to find SOFT_MAX and then scores@V MUL_MAT.
        int softmax_idx = -1;
        for (int j = i + 1; j < n && j < i + 6; j++) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_SOFT_MAX) { softmax_idx = j; break; }
            if (op == GGML_OP_SCALE || op == GGML_OP_ADD) continue;
            break;
        }
        if (softmax_idx < 0) {
            if (sched_dbg) fprintf(stderr, "    FlowKV reject [%d] MUL_MAT: no SOFT_MAX found within 6 nodes forward\n", i);
            continue;
        }

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

        const void * k_ptr = node->src[0]->data;

        if (pv_idx < 0) {
            // No scores@V in this segment — queue as pending for cross-segment.
            if (sched_dbg) fprintf(stderr, "    FlowKV pending [%d]: no scores@V after SOFT_MAX at [%d] (cross-segment)\n", i, softmax_idx);
            pending_heads.push_back({
                i, softmax_idx,
                node->src[0]->data,   // k_data
                nullptr,              // v_data (not needed for cross-seg)
                node->src[1]->data,   // q_data
                K, seq_len
            });
            continue;
        }

        // Validate PV shape: src[0] should have same head_dim.
        if (cgraph->nodes[pv_idx]->src[0]->ne[0] != K) continue;

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
            grp.k_tensor = nullptr;
            grp.v_tensor = nullptr;
            grp.q_tensor = nullptr;
            groups.push_back(grp);
            it = k_to_group.find(k_ptr);
        }
        groups[it->second].heads.push_back({i, pv_idx});
    }

    if (dbg && !groups.empty()) {
        size_t total_heads = 0;
        for (auto & g : groups) total_heads += g.heads.size();
        fprintf(stderr, "ggml-xdna: FlowKV plan: %zu KV head groups, %zu total heads "
                "(%zu pending)\n",
                groups.size(), total_heads, pending_heads.size());
        for (size_t g = 0; g < groups.size(); g++) {
            fprintf(stderr, "  group[%zu]: kv_head=%lld seq_len=%lld heads=%zu\n",
                    g, (long long)groups[g].kv_head_idx,
                    (long long)groups[g].seq_len, groups[g].heads.size());
        }
        fflush(stderr);
    }
    return groups;
}

// Helper: copy a single head slice from a batched 4D tensor.
static void flowkv_copy_head_slice(
        char * dst, const char * src_base,
        const struct ggml_tensor * t, int64_t head_idx,
        int64_t head_dim, size_t head_bytes) {
    const char * src = (t->ne[2] <= 1)
        ? src_base
        : src_base + head_idx * t->nb[2];

    if (t->type == GGML_TYPE_F16) {
        // f16 → bf16 conversion
        for (int64_t d = 0; d < head_dim; d++) {
            ggml_fp16_t f16_val;
            memcpy(&f16_val, src + d * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
            float f32_val = ggml_fp16_to_fp32(f16_val);
            uint16_t bf16_val;
            f32_to_bf16(&f32_val, &bf16_val, 1);
            memcpy(dst + d * 2, &bf16_val, 2);
        }
    } else if (t->type == GGML_TYPE_F32) {
        // f32 → bf16 conversion
        f32_to_bf16((const float *)src, (uint16_t *)dst, (size_t)head_dim);
    } else {
        // bf16: direct copy
        memcpy(dst, src, head_bytes);
    }
}

// Per-KV-head FlowKV dispatch. Processes one KV head group (group_size Q heads
// sharing the same K/V cache) with num_kv_heads=1, num_cols=1.
//
// Supports both:
//   - Expanded path: 32 separate Q@K^T MUL_MATs (per-head tensors)
//   - Batched path:   1 batched Q@K^T MUL_MAT (ne[2]=32, ne[3]=1)
static bool ggml_backend_xdna_flowkv_per_head(
        ggml_backend_xdna_context * ctx,
        const xdna_flowkv_group & group,
        const struct ggml_cgraph * cgraph) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;

    if (!ctx || !ctx->device_valid) return false;

    int64_t head_dim = group.head_dim;
    int64_t seq_len = group.seq_len;
    int chunk_size = 32;
    int num_cols = 1;

    if (head_dim != 64) return false;
    if (seq_len % chunk_size != 0) return false;

    // Detect batched mode: all heads reference the same qk_idx.
    bool batched = true;
    int ref_qk = group.heads[0].qk_idx;
    for (size_t h = 1; h < group.heads.size(); h++) {
        if (group.heads[h].qk_idx != ref_qk) { batched = false; break; }
    }

    // Resolve tensor pointers: use cross-segment pointers when available,
    // fall back to cgraph index lookup for same-segment groups.
    struct ggml_tensor * k_tensor = group.k_tensor
        ? const_cast<struct ggml_tensor *>(group.k_tensor)
        : cgraph->nodes[ref_qk]->src[0];
    struct ggml_tensor * v_tensor = group.v_tensor
        ? const_cast<struct ggml_tensor *>(group.v_tensor)
        : cgraph->nodes[group.heads[0].pv_idx]->src[0];
    struct ggml_tensor * q_tensor = group.q_tensor
        ? const_cast<struct ggml_tensor *>(group.q_tensor)
        : cgraph->nodes[ref_qk]->src[1];

    int64_t n_kv_heads = 1;
    int64_t n_q_heads_total = (int64_t)group.heads.size();
    if (batched && group.heads.size() > 1) {
        n_kv_heads = k_tensor->ne[2];
        if (n_kv_heads < 1) n_kv_heads = 1;
    }
    int64_t q_heads_per_kv = n_q_heads_total / n_kv_heads;

    bool all_ok = true;

    // === DIAG: print tensor metadata once per FlowKV call ===
    if (dbg) {
        fprintf(stderr, "\n[DIAG-FLOWKV] Tensor metadata:\n");
        fprintf(stderr, "    K: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
            k_tensor->data, (int)k_tensor->type,
            (long long)k_tensor->ne[0], (long long)k_tensor->ne[1], (long long)k_tensor->ne[2],
            (long long)k_tensor->nb[0], (long long)k_tensor->nb[1], (long long)k_tensor->nb[2]);
        fprintf(stderr, "    V: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
            v_tensor->data, (int)v_tensor->type,
            (long long)v_tensor->ne[0], (long long)v_tensor->ne[1], (long long)v_tensor->ne[2],
            (long long)v_tensor->nb[0], (long long)v_tensor->nb[1], (long long)v_tensor->nb[2]);
        fprintf(stderr, "    Q: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%lld,%lld,%lld]\n",
            q_tensor->data, (int)q_tensor->type,
            (long long)q_tensor->ne[0], (long long)q_tensor->ne[1], (long long)q_tensor->ne[2],
            (long long)q_tensor->nb[0], (long long)q_tensor->nb[1], (long long)q_tensor->nb[2]);
        fprintf(stderr, "    seq_len=%lld head_dim=%lld n_kv_heads=%lld group_size=%lld batched=%d\n",
            (long long)seq_len, (long long)head_dim, (long long)n_kv_heads,
            (long long)q_heads_per_kv, (int)batched);
        fflush(stderr);
    }
    // === END DIAG ===

    for (int64_t kv_h = 0; kv_h < n_kv_heads; kv_h++) {
        int64_t group_size = batched ? q_heads_per_kv : 1;
        int64_t num_heads = group_size;

        xdna_flowkv_entry * entry = get_or_load_flowkv_kernel(
            ctx, num_heads, /*num_kv_heads=*/1, head_dim, seq_len, chunk_size, num_cols);
        if (!entry) { all_ok = false; continue; }

        try {
            if (!entry->bo_kv) {
                size_t kv_size = 1 * seq_len * 2 * head_dim * sizeof(uint16_t);
                entry->bo_kv = std::make_unique<xrt::bo>(
                    xrt::bo(ctx->device, kv_size, xrt::bo::flags::host_only,
                            entry->kernel.group_id(3)));
            }
            if (!entry->bo_q) {
                size_t q_size = 1 * (group_size * head_dim + head_dim) * sizeof(uint16_t);
                entry->bo_q = std::make_unique<xrt::bo>(
                    xrt::bo(ctx->device, q_size, xrt::bo::flags::host_only,
                            entry->kernel.group_id(4)));
            }
            if (!entry->bo_out) {
                size_t out_size = group_size * head_dim * sizeof(uint16_t);
                entry->bo_out = std::make_unique<xrt::bo>(
                    xrt::bo(ctx->device, out_size, xrt::bo::flags::host_only,
                            entry->kernel.group_id(5)));
            }

            // --- Prepare KV cache buffer ---
            {
                auto kv_ptr = entry->bo_kv->map<char *>();
                const char * k_data = (const char *)k_tensor->data;
                const char * v_data = (const char *)v_tensor->data;
                size_t row_bytes = head_dim * sizeof(uint16_t);
                size_t kv_region = (size_t)seq_len * row_bytes;  // size of K or V region

                size_t k_pos_stride = k_tensor->nb[1];
                size_t k_head_stride = k_tensor->ne[2] > 1 ? k_tensor->nb[2] : 0;
                size_t v_pos_stride = v_tensor->nb[1];
                size_t v_head_stride = v_tensor->ne[2] > 1 ? v_tensor->nb[2] : 0;

                const bool k_f16 = (k_tensor->type == GGML_TYPE_F16);
                const bool v_f16 = (v_tensor->type == GGML_TYPE_F16);

                for (int64_t pos = 0; pos < seq_len; pos++) {
                    size_t dst_k = pos * row_bytes;
                    size_t src_k = pos * k_pos_stride + kv_h * k_head_stride;
                    if (k_f16) {
                        for (int64_t d = 0; d < head_dim; d++) {
                            ggml_fp16_t f16_val;
                            memcpy(&f16_val, k_data + src_k + d * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                            float f32_val = ggml_fp16_to_fp32(f16_val);
                            uint16_t bf16_val;
                            f32_to_bf16(&f32_val, &bf16_val, 1);
                            memcpy(kv_ptr + dst_k + d * 2, &bf16_val, 2);
                        }
                    } else {
                        memcpy(kv_ptr + dst_k, k_data + src_k, row_bytes);
                    }

                    size_t dst_v = kv_region + pos * row_bytes;
                    size_t src_v = pos * v_pos_stride + kv_h * v_head_stride;
                    if (v_f16) {
                        for (int64_t d = 0; d < head_dim; d++) {
                            ggml_fp16_t f16_val;
                            memcpy(&f16_val, v_data + src_v + d * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                            float f32_val = ggml_fp16_to_fp32(f16_val);
                            uint16_t bf16_val;
                            f32_to_bf16(&f32_val, &bf16_val, 1);
                            memcpy(kv_ptr + dst_v + d * 2, &bf16_val, 2);
                        }
                    } else {
                        memcpy(kv_ptr + dst_v, v_data + src_v, row_bytes);
                    }
                }
                entry->bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // === DIAG: verify host→DDR coherency ===
                // Re-read bo_kv from device memory and compare with what we wrote.
                // If these differ → host cache not flushed to DDR.
                // If these match but kernel still sees wrong data → NoC/DMA cache issue.
                if (dbg && kv_h == 0) {
                    // Save what we wrote
                    uint16_t pre_k0[8], pre_k1[8], pre_v0[8];
                    memcpy(pre_k0, kv_ptr + 0, 16);
                    memcpy(pre_k1, kv_ptr + row_bytes, 16);
                    memcpy(pre_v0, kv_ptr + kv_region, 16);

                    // Sync back from device (read DDR into host buffer)
                    entry->bo_kv->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    auto verify_ptr = reinterpret_cast<const uint16_t*>(kv_ptr);

                    fprintf(stderr, "  [DIAG-SYNC] bo_kv verify after FROM_DEVICE:\n");
                    fprintf(stderr, "    K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        verify_ptr[0], verify_ptr[1], verify_ptr[2], verify_ptr[3],
                        verify_ptr[4], verify_ptr[5], verify_ptr[6], verify_ptr[7]);
                    fprintf(stderr, "    K[1][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        verify_ptr[64], verify_ptr[65], verify_ptr[66], verify_ptr[67],
                        verify_ptr[68], verify_ptr[69], verify_ptr[70], verify_ptr[71]);
                    fprintf(stderr, "    V[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        verify_ptr[kv_region/2], verify_ptr[kv_region/2+1],
                        verify_ptr[kv_region/2+2], verify_ptr[kv_region/2+3],
                        verify_ptr[kv_region/2+4], verify_ptr[kv_region/2+5],
                        verify_ptr[kv_region/2+6], verify_ptr[kv_region/2+7]);

                    // Check match
                    bool k0_match = (memcmp(pre_k0, kv_ptr, 16) == 0);
                    bool k1_match = (memcmp(pre_k1, kv_ptr + row_bytes, 16) == 0);
                    fprintf(stderr, "    K[0] match=%s  K[1] match=%s\n",
                        k0_match ? "YES" : "NO!!", k1_match ? "YES" : "NO!!");
                    fflush(stderr);

                    // Re-sync TO_DEVICE since FROM_DEVICE overwrote our buffer
                    entry->bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                }
                // === END DIAG ===
            }

            // --- Prepare Q buffer ---
            {
                auto q_ptr = entry->bo_q->map<char *>();
                size_t q_head_bytes = head_dim * sizeof(uint16_t);

                char angle_cos_bf16[2] = {(char)0x80, (char)0x3F};
                char angle_sin_bf16[2] = {(char)0x00, (char)0x00};

                struct ggml_tensor * q_tsr = q_tensor;
                const char * q_data = (const char *)q_tsr->data;

                for (int64_t g = 0; g < group_size; g++) {
                    int64_t q_head = batched ? (kv_h * q_heads_per_kv + g) : g;
                    flowkv_copy_head_slice(
                        q_ptr + g * q_head_bytes, q_data,
                        q_tensor, q_head, head_dim, q_head_bytes);
                }

                size_t angles_off = group_size * q_head_bytes;
                for (int d = 0; d < head_dim; d += 2) {
                    memcpy(q_ptr + angles_off + d * 2, angle_cos_bf16, 2);
                    memcpy(q_ptr + angles_off + d * 2 + 2, angle_sin_bf16, 2);
                }
                entry->bo_q->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }

            // --- Dispatch ---
            // IRON xclbin arg layout: opcode(0), insts(1), insts_size(2),
            // DDR_buf_0(3)=kv, DDR_buf_1(4)=q, DDR_buf_2(5)=out
            auto run = entry->kernel(
                3, entry->insts_bo, (uint32_t)entry->insts_data.size(),
                *entry->bo_kv, *entry->bo_q, *entry->bo_out);

            {
                std::lock_guard<std::mutex> lock(*entry->mu);
                auto t0 = std::chrono::steady_clock::now();
                run.start();
                auto state = run.wait(30000);
                auto t1 = std::chrono::steady_clock::now();

                if (state != ERT_CMD_STATE_COMPLETED) {
                    GGML_LOG_ERROR("ggml-xdna: FlowKV per-head dispatch failed, state=%d\n",
                                   (int)state);
                    all_ok = false;
                    continue;
                }

                if (dbg) {
                    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    fprintf(stderr, "ggml-xdna: [FlowKV-POC] kv_h=%lld dispatched %.2f ms\n",
                            (long long)kv_h, ms);

                    // === DIAG: post-dispatch bo_kv integrity check ===
                    // Verify DMA didn't corrupt the source buffer.
                    if (kv_h == 0) {
                        entry->bo_kv->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        auto post_ptr = reinterpret_cast<const uint16_t*>(
                            entry->bo_kv->map<char *>());
                        fprintf(stderr, "  [DIAG-POST] bo_kv after dispatch:\n");
                        fprintf(stderr, "    K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                            post_ptr[0], post_ptr[1], post_ptr[2], post_ptr[3],
                            post_ptr[4], post_ptr[5], post_ptr[6], post_ptr[7]);
                        fprintf(stderr, "    bo_kv addr=%p bo_out addr=%p\n",
                            (void*)post_ptr, (void*)entry->bo_out->map<char *>());
                        fflush(stderr);
                    }
                    // === END DIAG ===
                }
            }

            // --- Read back output and scatter ---
            entry->bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            {
                auto out_ptr = entry->bo_out->map<char *>();
                size_t out_head_bytes = head_dim * sizeof(uint16_t);

                // === DIAG: compare K_DIAG (from kernel) with host K[0] ===
                if (dbg && kv_h == 0) {
                    // K_DIAG is written to the last head slot by the kernel
                    size_t kdiag_off = (num_heads - 1) * out_head_bytes;
                    const uint16_t * kdiag = reinterpret_cast<const uint16_t*>(out_ptr + kdiag_off);
                    // Re-read bo_kv to get host K[0]
                    entry->bo_kv->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    auto kv_verify = reinterpret_cast<const uint16_t*>(entry->bo_kv->map<char *>());

                    fprintf(stderr, "  [DIAG-COMPARE] kv_h=0:\n");
                    fprintf(stderr, "    Host  K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        kv_verify[0], kv_verify[1], kv_verify[2], kv_verify[3],
                        kv_verify[4], kv_verify[5], kv_verify[6], kv_verify[7]);
                    fprintf(stderr, "    K_DIAG    [0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        kdiag[0], kdiag[1], kdiag[2], kdiag[3],
                        kdiag[4], kdiag[5], kdiag[6], kdiag[7]);

                    // Check if any of the first 8 match
                    int match_count = 0;
                    for (int i = 0; i < 8; i++) {
                        if (kdiag[i] == kv_verify[i]) match_count++;
                    }
                    fprintf(stderr, "    Match count (first 8): %d/8 → %s\n",
                        match_count, match_count > 0 ? "PARTIAL MATCH" : "ZERO MATCH — DMA reads wrong data!");

                    // Also dump bo_out raw first 8 for context
                    fprintf(stderr, "    bo_out raw[0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                        reinterpret_cast<const uint16_t*>(out_ptr)[0],
                        reinterpret_cast<const uint16_t*>(out_ptr)[1],
                        reinterpret_cast<const uint16_t*>(out_ptr)[2],
                        reinterpret_cast<const uint16_t*>(out_ptr)[3],
                        reinterpret_cast<const uint16_t*>(out_ptr)[4],
                        reinterpret_cast<const uint16_t*>(out_ptr)[5],
                        reinterpret_cast<const uint16_t*>(out_ptr)[6],
                        reinterpret_cast<const uint16_t*>(out_ptr)[7]);
                    fflush(stderr);
                }
                // === END DIAG ===

                struct ggml_tensor * out_tensor =
                    cgraph->nodes[group.heads[0].pv_idx];
                const char * out_base = (const char *)out_tensor->data;

                for (int64_t g = 0; g < group_size; g++) {
                    int64_t q_head = batched ? (kv_h * q_heads_per_kv + g) : g;
                    if (out_tensor->ne[2] > 1) {
                        memcpy((char *)out_base + q_head * out_tensor->nb[2],
                               out_ptr + g * out_head_bytes, out_head_bytes);
                    } else {
                        struct ggml_tensor * h_out =
                            cgraph->nodes[group.heads[g].pv_idx];
                        memcpy(h_out->data, out_ptr + g * out_head_bytes,
                               out_head_bytes);
                    }
                }
            }
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna: FlowKV per-head exception: %s\n", e.what());
            all_ok = false;
        }
    } // for kv_h

    return all_ok;
}

// ============================================================================
// Decode GEMV batcher — collects standalone MUL_MAT M=1 dispatches and
// flushes them as a single xrt::runlist per (K,N) shape group.
//
// In decode mode, each transformer layer has an O projection GEMV that is
// dispatched individually. With 32 layers, that's 32 separate xrt::run
// submissions (~1-5ms each). The batcher collects these and dispatches
// them as one runlist per shape group, reducing host-side overhead.
//
// Gate: XDNA_ENABLE_DECODE_BATCH=1 (requires XDNA_ENABLE_GEMV=1 for
// standalone GEMV dispatch).
// ============================================================================

struct xdna_decode_batcher {
    struct batch_item {
        xdna_kernel_entry * entry;  // kernel for this (K,N) shape
        struct ggml_tensor * dst;   // MUL_MAT output tensor (has src[0]=weight, src[1]=input)
        int node_idx;               // cgraph index (for debug logging)
    };

    std::vector<batch_item> items;
    bool enabled;

    xdna_decode_batcher() : enabled(xdna_env_enabled("XDNA_ENABLE_DECODE_BATCH")) {}

    bool empty() const { return items.empty(); }
    bool is_enabled() const { return enabled; }

    void add(xdna_kernel_entry * entry, struct ggml_tensor * dst, int idx) {
        if (!enabled) return;
        items.push_back({entry, dst, idx});
    }

    // Dispatch all collected GEMVs as xrt::runlists, one per (K,N) shape group.
    // Falls back to individual dispatch on failure.
    void flush(ggml_backend_xdna_context * ctx) {
        if (items.empty()) return;

        static const bool prof = getenv("XDNA_DEBUG") != NULL;
        using clk = std::chrono::steady_clock;

        // Group items by (K,N) shape — same shape shares kernel/hw_ctx.
        struct shape_key {
            int64_t K, N;
            bool operator==(const shape_key & o) const { return K == o.K && N == o.N; }
        };
        struct shape_hash {
            size_t operator()(const shape_key & k) const {
                size_t h1 = std::hash<int64_t>{}(k.K);
                size_t h2 = std::hash<int64_t>{}(k.N);
                return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
            }
        };
        std::unordered_map<shape_key, std::vector<batch_item *>, shape_hash> groups;
        for (auto & item : items) {
            int64_t K = item.dst->src[0]->ne[0];
            int64_t N = item.dst->src[0]->ne[1];
            groups[{K, N}].push_back(&item);
        }

        auto t_flush_s = clk::now();
        int n_dispatched = 0;

        for (auto & [key, group_entries] : groups) {
            const int batch_n = (int)group_entries.size();
            xdna_kernel_entry * entry = group_entries[0]->entry;

            // Single item: dispatch via existing direct path (less overhead).
            if (batch_n == 1) {
                try {
                    ggml_backend_xdna_mul_mat_gemv(ctx, group_entries[0]->dst);
                    n_dispatched++;
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("ggml-xdna: decode_batch single dispatch failed: %s\n", e.what());
                }
                continue;
            }

            // Multiple same-shape GEMVs: build one xrt::runlist.
            try {
                // Per-item BOs — must outlive rl.execute()/rl.wait().
                std::vector<xrt::bo> in_bos;
                std::vector<xrt::bo> out_bos;
                std::vector<xrt::bo *> w_ptrs;
                in_bos.reserve(batch_n);
                out_bos.reserve(batch_n);
                w_ptrs.reserve(batch_n);

                auto t_dma_s = clk::now();

                for (int j = 0; j < batch_n; j++) {
                    auto * item = group_entries[j];
                    const ggml_tensor * src0 = item->dst->src[0];  // weight
                    const ggml_tensor * src1 = item->dst->src[1];  // input
                    const int64_t K = src0->ne[0];
                    const int64_t N = src0->ne[1];
                    const size_t mat_elems = (size_t)(N * K);

                    // Weight BO from cache (immutable after first load).
                    xrt::bo * w_bo = nullptr;
                    {
                        std::lock_guard<std::mutex> lock(*entry->b_bo_mutex);
                        auto it = entry->b_bo_cache.find(src0->data);
                        if (it == entry->b_bo_cache.end()) {
                            size_t mat_bytes = mat_elems * sizeof(uint16_t);
                            xrt::bo new_w(ctx->device, mat_bytes,
                                          xrt::bo::flags::host_only,
                                          entry->kernel.group_id(3));
                            if (src0->type == GGML_TYPE_F32) {
                                f32_to_bf16((const float *)src0->data,
                                            (uint16_t *)new_w.map<void*>(), mat_elems);
                            } else {
                                memcpy(new_w.map<void*>(), src0->data, mat_bytes);
                            }
                            new_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                            auto [ins, _] = entry->b_bo_cache.emplace(src0->data, std::move(new_w));
                            w_bo = &ins->second;
                            if (prof) {
                                fprintf(stderr,
                                    "ggml-xdna: decode_batch warm weight K=%lld N=%lld %s (%zu cached)\n",
                                    (long long)K, (long long)N,
                                    src0->name[0] ? src0->name : "?",
                                    entry->b_bo_cache.size());
                                fflush(stderr);
                            }
                        } else {
                            w_bo = &it->second;
                        }
                    }
                    w_ptrs.push_back(w_bo);

                    // Per-item input BO.
                    size_t vec_bytes = (size_t)K * sizeof(uint16_t);
                    in_bos.emplace_back(ctx->device, vec_bytes,
                                        xrt::bo::flags::host_only,
                                        entry->kernel.group_id(4));
                    if (src1->type == GGML_TYPE_F32) {
                        f32_to_bf16((const float *)src1->data,
                                    (uint16_t *)in_bos.back().map<void*>(),
                                    (size_t)K);
                    } else {
                        memcpy(in_bos.back().map<void*>(), src1->data, vec_bytes);
                    }
                    in_bos.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);

                    // Per-item output BO.
                    size_t out_bytes = (size_t)N * sizeof(uint16_t);
                    out_bos.emplace_back(ctx->device, out_bytes,
                                         xrt::bo::flags::host_only,
                                         entry->kernel.group_id(5));
                }
                auto t_dma_e = clk::now();

                // Build runlist.
                auto t_build_s = clk::now();
                xrt::runlist rl(entry->hw_ctx);
                for (int j = 0; j < batch_n; j++) {
                    xrt::run r(entry->kernel);
                    r.set_arg(0, 3u);  // GEMV opcode
                    r.set_arg(1, entry->insts_bo);
                    r.set_arg(2, (uint32_t)entry->insts.size());
                    r.set_arg(3, *w_ptrs[j]);
                    r.set_arg(4, in_bos[j]);
                    r.set_arg(5, out_bos[j]);
                    rl.add(r);
                }
                auto t_build_e = clk::now();

                // Execute runlist (all GEMVs in one firmware submission).
                auto t_exec_s = clk::now();
                rl.execute();
                auto t_exec_e = clk::now();
                rl.wait();
                auto t_wait_e = clk::now();

                // Sync outputs back and convert bf16 → f32.
                auto t_out_s = clk::now();
                for (int j = 0; j < batch_n; j++) {
                    const int64_t N = group_entries[j]->dst->src[0]->ne[1];
                    out_bos[j].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    bf16_to_f32((const uint16_t *)out_bos[j].map<void*>(),
                                (float *)group_entries[j]->dst->data,
                                (size_t)N);
                }
                auto t_out_e = clk::now();

                n_dispatched += batch_n;

                if (prof) {
                    auto us = [](clk::time_point a, clk::time_point b) {
                        return (long long)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
                    };
                    fprintf(stderr,
                        "ggml-xdna: decode_batch K=%lld N=%lld count=%d "
                        "dma=%lld build=%lld exec=%lld wait=%lld out=%lld total=%lld us\n",
                        (long long)key.K, (long long)key.N, batch_n,
                        us(t_dma_s, t_dma_e),
                        us(t_build_s, t_build_e),
                        us(t_exec_s, t_exec_e),
                        us(t_exec_e, t_wait_e),
                        us(t_out_s, t_out_e),
                        us(t_dma_s, t_out_e));
                    fflush(stderr);
                }
            } catch (const std::exception & e) {
                GGML_LOG_ERROR("ggml-xdna: decode_batch runlist failed "
                               "(K=%lld N=%lld count=%d): %s, falling back to individual dispatch\n",
                               (long long)key.K, (long long)key.N, batch_n, e.what());
                // Fallback: dispatch each item individually.
                for (int j = 0; j < batch_n; j++) {
                    try {
                        ggml_backend_xdna_mul_mat_gemv(ctx, group_entries[j]->dst);
                        n_dispatched++;
                    } catch (const std::exception & e2) {
                        GGML_LOG_ERROR("ggml-xdna: decode_batch fallback also failed: %s\n", e2.what());
                    }
                }
            }
        }

        if (prof && n_dispatched > 0) {
            auto t_flush_e = clk::now();
            auto flush_us = std::chrono::duration_cast<std::chrono::microseconds>(
                t_flush_e - t_flush_s).count();
            fprintf(stderr,
                "ggml-xdna: decode_batch flush: %d GEMVs in %zu runlists, "
                "%lld us total, %d dispatches saved\n",
                n_dispatched, groups.size(), (long long)flush_us,
                n_dispatched - (int)groups.size());
            fflush(stderr);
        }

        clear();
    }

    void clear() { items.clear(); }
};

// Pre-scan the graph for standalone decode GEMVs (MUL_MAT with M==1) that
// are shape-dispatchable and not already consumed by QKV or SwiGLU matchers.
// Returns node indices eligible for decode_batch collection.
static std::unordered_set<int> xdna_plan_decode_batch(
    const struct ggml_cgraph * cgraph,
    const xdna_qkv_plan & qkv_plan)
{
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    std::unordered_set<int> batchable;

    // Indices already consumed by QKV.
    std::unordered_set<int> consumed = qkv_plan.skip_indices;
    for (auto & [idx, _] : qkv_plan.triple_at) {
        consumed.insert(idx);
    }

    int n_decode_gemv = 0;
    int n_attn_reject = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        const int64_t M = node->src[1]->ne[1];
        if (M != 1) continue;
        n_decode_gemv++;
        const int64_t K = node->src[0]->ne[0];
        const int64_t N = node->src[0]->ne[1];
        if (!xdna_shape_dispatchable_gemv(K, N)) continue;
        if (consumed.count(i)) continue;

        // Attention guard: skip MUL_MAT nodes that are near SOFT_MAX.
        // Case 1 (forward): Q@K^T before SOFT_MAX
        // Case 2 (backward): scores@V after SOFT_MAX
        // Both have permuted/interleaved data that the NPU GEMV kernel
        // cannot handle correctly.
        bool is_attn = false;
        // Forward scan: Q@K^T → [SCALE] [ADD] → SOFT_MAX
        for (int j = i + 1; j < cgraph->n_nodes && j < i + 8; j++) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_SOFT_MAX) { is_attn = true; break; }
            if (op == GGML_OP_SCALE || op == GGML_OP_ADD ||
                op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                op == GGML_OP_PERMUTE || op == GGML_OP_CONT) continue;
            break;
        }
        // Backward scan: SOFT_MAX → [VIEW/RESHAPE/PERMUTE] → scores@V
        if (!is_attn) {
            for (int j = i - 1; j >= 0 && j > i - 8; j--) {
                enum ggml_op op = cgraph->nodes[j]->op;
                if (op == GGML_OP_SOFT_MAX) { is_attn = true; break; }
                if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                    op == GGML_OP_PERMUTE || op == GGML_OP_CONT) continue;
                break;
            }
        }
        if (is_attn) {
            n_attn_reject++;
            if (dbg) {
                fprintf(stderr,
                    "ggml-xdna: decode_batch skip [%d] attn MUL_MAT K=%lld N=%lld\n",
                    i, (long long)K, (long long)N);
                fflush(stderr);
            }
            continue;
        }

        batchable.insert(i);
    }

    if (dbg) {
        fprintf(stderr,
            "ggml-xdna: decode_batch plan: %d decode GEMVs, %zu batchable "
            "(QKV consumed %zu indices, attn reject %d)\n",
            n_decode_gemv, batchable.size(), consumed.size(), n_attn_reject);
        fflush(stderr);
    }

    return batchable;
}

// Matched SwiGLU pattern. Populated by xdna_try_match_swiglu.
struct xdna_swiglu_match {
    struct ggml_tensor * gate_mm;    // node[i]   — MUL_MAT(gate_w, input)
    struct ggml_tensor * up_mm;      // node[i+1] — MUL_MAT(up_w,   input)
    struct ggml_tensor * glu;        // node[i+2] — GLU(gate_mm, up_mm)
    struct ggml_tensor * down_mm;    // node[i+3] — MUL_MAT(down_w, glu)
    const struct ggml_tensor * gate_w;
    const struct ggml_tensor * up_w;
    const struct ggml_tensor * down_w;
    const struct ggml_tensor * input;
    // When true the three weights are Q8_0 and the int8 dispatch path
    // (ggml_backend_xdna_mul_mat_swiglu_int8) should be used. Bf16 matches
    // leave this false for the existing bf16 dispatcher.
    bool is_int8;
    // Number of AIE columns to use. Prefer 8 (full NPU2), fall back to 4
    // when hidden_dim/embedding_dim don't tile at 8 cols.
    int num_cols;
};

// Attempt to match a 4-node SwiGLU pattern starting at cgraph->nodes[i].
// Strict: requires the standard ffn_norm→gate/up→swiglu_split→down shape
// llama.cpp emits via ggml_swiglu_split in llama-graph.cpp:1141.
static bool xdna_try_match_swiglu(const struct ggml_cgraph * cgraph, int i,
                                  xdna_swiglu_match * out) {
    // Under XDNA_DEBUG, log why early match attempts fail — capped so a long
    // session doesn't drown the log. Useful when llama.cpp cgraph emission
    // changes and the matcher suddenly misses.
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    static std::atomic<int> dbg_remaining{dbg ? 16 : 0};
    auto dbg_ok = [&]() { return dbg && dbg_remaining.fetch_sub(1) > 0; };
    #define SWIGLU_REJECT(reason) do { \
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: swiglu reject @%d: %s\n", i, reason); \
        return false; \
    } while (0)

    if (i + 3 >= cgraph->n_nodes) return false;

    struct ggml_tensor * n0 = cgraph->nodes[i];
    struct ggml_tensor * n1 = cgraph->nodes[i + 1];
    struct ggml_tensor * n2 = cgraph->nodes[i + 2];
    struct ggml_tensor * n3 = cgraph->nodes[i + 3];

    if (n0->op != GGML_OP_MUL_MAT) return false;
    if (n1->op != GGML_OP_MUL_MAT) return false;
    if (n2->op != GGML_OP_GLU)     return false;
    if (n3->op != GGML_OP_MUL_MAT) return false;

    if (ggml_get_glu_op(n2) != GGML_GLU_OP_SWIGLU) SWIGLU_REJECT("not SWIGLU glu_op");
    if (ggml_get_op_params_i32(n2, 1) != 0)        SWIGLU_REJECT("swapped flag set");

    struct ggml_tensor * gate_mm = nullptr;
    struct ggml_tensor * up_mm   = nullptr;
    if (n2->src[0] == n0 && n2->src[1] == n1) {
        gate_mm = n0; up_mm = n1;
    } else if (n2->src[0] == n1 && n2->src[1] == n0) {
        gate_mm = n1; up_mm = n0;
    } else {
        SWIGLU_REJECT("GLU srcs don't link to n0/n1");
    }

    if (n3->src[1] != n2) SWIGLU_REJECT("down_mm src[1] != GLU");

    const struct ggml_tensor * gate_w = gate_mm->src[0];
    const struct ggml_tensor * up_w   = up_mm->src[0];
    const struct ggml_tensor * down_w = n3->src[0];
    const struct ggml_tensor * input  = gate_mm->src[1];

    if (up_mm->src[1] != input) SWIGLU_REJECT("gate/up don't share input");

    if (gate_w->ne[0] != up_w->ne[0]) SWIGLU_REJECT("gate/up ne[0] mismatch");
    if (gate_w->ne[1] != up_w->ne[1]) SWIGLU_REJECT("gate/up ne[1] mismatch");
    if (gate_w->type  != up_w->type)  SWIGLU_REJECT("gate/up type mismatch");

    if (input->type != GGML_TYPE_F32) {
        if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: input type=%d (want F32)\n",
                         i, input->type);
        return false;
    }
    // Decide which precision path this pattern is eligible for. Q8_0 is only
    // accepted when the master SwiGLU gate and the INT8 opt-in flag are both
    // set; otherwise we keep the existing bf16/f32 behaviour untouched.
    static const bool int8_enabled = xdna_env_enabled("XDNA_ENABLE_SWIGLU_INT8");
    // Tblock W8A16 has its own Q8_0 upload path (dequant → fp32 → packer).
    // When tblock+W8A16 is active we accept Q8_0 weights here so the
    // tblock matcher's swiglu sub-match succeeds; is_int8 is still set so
    // the tblock consumer can distinguish "Q8_0 weights need the uint8
    // upload branch" from "bf16/f16 weights take the bf16 upload branch".
    static const bool tblock_w8a16_gate =
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED")) &&
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED_W8A16"));
    const bool all_q8_0 = (gate_w->type == GGML_TYPE_Q8_0)
                       && (up_w->type   == GGML_TYPE_Q8_0)
                       && (down_w->type == GGML_TYPE_Q8_0);
    // IRON-windows does not compile INT8 kernels — force-disable.
    const bool allow_int8 = (int8_enabled || tblock_w8a16_gate) && all_q8_0
#ifdef _WIN32
                            && false
#endif
                            ;

    const struct ggml_tensor * ws[3] = { gate_w, up_w, down_w };
    const char * ws_names[3] = { "gate_w", "up_w", "down_w" };
    for (int wi = 0; wi < 3; wi++) {
        const struct ggml_tensor * w = ws[wi];
        const bool bf16_typed = (w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F16);
        const bool int8_typed = (w->type == GGML_TYPE_Q8_0) && allow_int8;
        if (!bf16_typed && !int8_typed) {
            if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: %s type=%d\n",
                             i, ws_names[wi], w->type);
            return false;
        }
        if (!ggml_is_contiguous(w)) {
            if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: %s non-contiguous\n",
                             i, ws_names[wi]);
            return false;
        }
        if (w->ne[2] != 1 || w->ne[3] != 1) {
            if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: %s ne[2,3]=%lld,%lld\n",
                             i, ws_names[wi], (long long)w->ne[2], (long long)w->ne[3]);
            return false;
        }
    }
    if (!ggml_is_contiguous(input)) SWIGLU_REJECT("input non-contiguous");
    if (input->ne[2] != 1 || input->ne[3] != 1) SWIGLU_REJECT("input ne[2,3] != 1");

    const int64_t embedding_dim = input->ne[0];
    const int64_t hidden_dim    = gate_w->ne[1];
    if (gate_w->ne[0] != embedding_dim) {
        if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: gate_w[0]=%lld != input[0]=%lld\n",
                         i, (long long)gate_w->ne[0], (long long)embedding_dim);
        return false;
    }
    if (down_w->ne[0] != hidden_dim) {
        if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: down_w[0]=%lld != hidden=%lld\n",
                         i, (long long)down_w->ne[0], (long long)hidden_dim);
        return false;
    }
    if (down_w->ne[1] != embedding_dim) {
        if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: down_w[1]=%lld != embedding=%lld\n",
                         i, (long long)down_w->ne[1], (long long)embedding_dim);
        return false;
    }

    const int64_t M = input->ne[1];

    int num_cols = 4;
    const char * cols_env = getenv("GGML_XDNA_NUM_COLS");
    if (cols_env) num_cols = atoi(cols_env);

    if (M == 1) {
        if (allow_int8) {
            if (!xdna_shape_dispatchable_swiglu_decode_int8(
                    embedding_dim, hidden_dim, num_cols, /*group_size=*/32))
                SWIGLU_REJECT("decode-int8 shape not dispatchable");
        } else {
            if (!xdna_shape_dispatchable_swiglu_decode(embedding_dim, hidden_dim, num_cols))
                SWIGLU_REJECT("decode shape not dispatchable");
        }
    } else if (M >= 1) {
        // Prefill path. M is rounded up to the next multiple of 64
        // requires tile_m >= 16 → M % 64 == 0). The M >= 32 floor keeps the
        // padding waste ≤ 2× (padded_M is always 64 for M ≤ 64). Below 32 the
        // kernel is dominated by per-submit XRT overhead (~11ms/layer) on
        // compute the user didn't ask for — catastrophic at spec-decode
        // verify batches (M=2..8) where the overhead is paid ~28 times per
        // token for zero net throughput. Keep this aligned with
        // xdna_shape_dispatchable()'s M >= 32 floor for standalone GEMM.
        static const bool prefill_enabled =
            (xdna_env_enabled("XDNA_ENABLE_SWIGLU_PREFILL")) ||
            (xdna_env_enabled("XDNA_ENABLE_TRANSFORMER_BLOCK")) ||
            (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED"));
        if (!prefill_enabled) SWIGLU_REJECT("prefill disabled (set XDNA_ENABLE_SWIGLU_PREFILL=1 or XDNA_ENABLE_TRANSFORMER_BLOCK=1 or XDNA_ENABLE_TBLOCK_FUSED=1 to opt in)");
        const int64_t padded_M = ((M + 63) / 64) * 64;
        if (allow_int8) {
            // Tblock+W8A16 is itself a prefill-int8 opt-in; don't gate it
            // behind XDNA_ENABLE_SWIGLU_PREFILL_INT8 (which is for the
            // standalone int8 swiglu dispatch path).
            static const bool prefill_int8_ok =
                (xdna_env_enabled("XDNA_ENABLE_SWIGLU_PREFILL_INT8")) ||
                tblock_w8a16_gate;
            if (!prefill_int8_ok)
                SWIGLU_REJECT("prefill-int8 disabled (set XDNA_ENABLE_SWIGLU_PREFILL_INT8=1)");
            int tm = xdna_pick_swiglu_prefill_tile_m(padded_M);
            if (tm < 16) SWIGLU_REJECT("prefill-int8 tile_m < 16");
            if (!xdna_shape_dispatchable_swiglu_prefill(padded_M, embedding_dim, hidden_dim, num_cols))
                SWIGLU_REJECT("prefill-int8 shape not dispatchable");
        } else {
            if (!xdna_shape_dispatchable_swiglu_prefill(padded_M, embedding_dim, hidden_dim, num_cols)) {
                if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: prefill M=%lld(pad=%lld) K=%lld N=%lld not dispatchable\n",
                                 i, (long long)M, (long long)padded_M, (long long)embedding_dim, (long long)hidden_dim);
                return false;
            }
        }
    } else {
        if (dbg) fprintf(stderr, "ggml-xdna: swiglu reject @%d: M=%lld below prefill floor (M>=1 required)\n",
                         i, (long long)M);
        return false;
    }
    #undef SWIGLU_REJECT

    out->gate_mm = gate_mm;
    out->up_mm   = up_mm;
    out->glu     = n2;
    out->down_mm = n3;
    out->gate_w  = gate_w;
    out->up_w    = up_w;
    out->down_w  = down_w;
    out->input   = input;
    out->is_int8 = allow_int8;
    out->num_cols = num_cols;
    return true;
}

// ============================================================================
// Attention-prefill matcher (Stage 4b.3 Phase A — scaffolding only).
//
// Purpose: prove that the multi-node pattern matcher reliably fires on a real
// Llama-3.2-1B graph, with aggressive offload_op keeping the whole attention
// block in the XDNA scheduler segment.
//
// Phase A does NOT dispatch. On a successful match we only log; the matched
// nodes still fall through to CPU via the normal per-node loop. This keeps
// the model producing correct output while we observe matches.
//
// The pattern we look for (per layer, Llama-3.2-1B):
//   (a) RMS_NORM(inpL)
//   (b) MUL(rms_norm_out, attn_norm_weight)          ← gain multiply
//   (c) MUL_MAT(wq, attn_norm_out)                    ← Q proj
//   (d) MUL_MAT(wk, attn_norm_out)                    ← K proj
//   (e) MUL_MAT(wv, attn_norm_out)                    ← V proj
//       [VIEW/RESHAPE/CONT intermediates — tolerated and skipped]
//   (f) ROPE on Q
//   (g) ROPE on K
//       [SET_ROWS for KV-cache writes — tolerated and skipped]
//   (h) FLASH_ATTN_EXT(Q_rope, K_cached, V_cached, mask)
//       [VIEW/RESHAPE/CONT on attn output — tolerated]
//   (i) MUL_MAT(wo, attn_out)                         ← O proj
//   (j) ADD(o_proj, inpL)                             ← residual
//
// Expanded-attention (no FLASH_ATTN_EXT, i.e. MUL_MAT(K,Q)→SCALE→ADD(mask)
// →SOFT_MAX→MUL_MAT(V,softmax)) is not supported in Phase A — we log and
// bail. Llama.cpp emits FLASH_ATTN_EXT by default.
// ============================================================================

struct xdna_attention_match {
    int rms_norm_idx;
    int attn_norm_mul_idx;
    int q_proj_idx;
    int k_proj_idx;
    int v_proj_idx;
    int q_rope_idx;
    int k_rope_idx;
    int attn_core_idx;       // FLASH_ATTN_EXT index; -1 for expanded path (unsupported)
    int o_proj_idx;
    int residual_add_idx;

    // Key tensor pointers (for future dispatch)
    struct ggml_tensor * inpL;
    struct ggml_tensor * gain;
    struct ggml_tensor * wq;
    struct ggml_tensor * wk;
    struct ggml_tensor * wv;
    struct ggml_tensor * wo;
    struct ggml_tensor * inp_pos;
    struct ggml_tensor * attn_rope_freqs;
    struct ggml_tensor * q_rope_node;  // for op_params: theta_base, mode, n_dims, scales

    // Shape params (for future xclbin selection)
    int64_t seq_len;
    int64_t embed_dim;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t head_dim;

    // IRON RoPE method_type derived from ggml RoPE op_params[2]:
    //   ggml mode 0 (NORMAL/adjacent-pair) -> 1 (INTERLEAVED)
    //   ggml mode 2 (NEOX/half-split)      -> 0 (TWO_HALVES)
    // Set by xdna_try_match_attention_prefill from the q_rope node.
    int rope_method_type;
};

// Walk backwards through VIEW/RESHAPE/CONT/PERMUTE/TRANSPOSE/CPY/DUP nodes to
// find the underlying "real" tensor a later op was actually built from. Used
// for inpL residual identification and for tracking the attn_norm_out
// activation as it is re-shaped through Q/K/V projections.
static struct ggml_tensor * xdna_strip_view(struct ggml_tensor * t) {
    while (t && t->src[0] != nullptr) {
        switch (t->op) {
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_CONT:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CPY:
            case GGML_OP_DUP:
                t = t->src[0];
                continue;
            default:
                return t;
        }
    }
    return t;
}

// Attempt to match a full attention prefill block starting at cgraph->nodes[i]
// (which must be a RMS_NORM). On success, populates *out with tensor pointers
// and node indices and returns true. Does NOT consume nodes — caller is
// responsible for deciding what to do with the match.
static bool xdna_try_match_attention_prefill(const struct ggml_cgraph * cgraph, int start_idx,
                                             xdna_attention_match * out) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    // Bounded reject-reason log (per-process, reset conceptually per graph
    // via the counter in graph_compute — here we just cap total spam).
    static std::atomic<int> dbg_remaining{dbg ? 24 : 0};
    auto dbg_ok = [&]() { return dbg && dbg_remaining.fetch_sub(1) > 0; };
    #define ATTN_REJECT(reason) do { \
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: attn_prefill reject @%d: %s\n", start_idx, reason); \
        return false; \
    } while (0)

    const int n = cgraph->n_nodes;
    if (start_idx >= n) return false;

    struct ggml_tensor * n_rms = cgraph->nodes[start_idx];
    if (n_rms->op != GGML_OP_RMS_NORM) return false;

    struct ggml_tensor * inpL = n_rms->src[0];
    if (!inpL) ATTN_REJECT("RMS_NORM has no src[0]");

    // Step 1: MUL(rms_norm_out, gain) — the attn_norm gain. Scan a small
    // window forward (tolerate a view or two between RMS_NORM and MUL).
    int attn_norm_mul_idx = -1;
    struct ggml_tensor * attn_norm_out = nullptr;
    struct ggml_tensor * gain = nullptr;
    for (int j = start_idx + 1; j < std::min(start_idx + 6, n); j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_MUL) continue;
        // One of the two sources must derive from n_rms (possibly through a
        // view). The other is the gain weight.
        struct ggml_tensor * s0 = nj->src[0];
        struct ggml_tensor * s1 = nj->src[1];
        struct ggml_tensor * s0r = xdna_strip_view(s0);
        struct ggml_tensor * s1r = xdna_strip_view(s1);
        if (s0r == n_rms) {
            attn_norm_mul_idx = j;
            attn_norm_out = nj;
            gain = s1;
            break;
        } else if (s1r == n_rms) {
            attn_norm_mul_idx = j;
            attn_norm_out = nj;
            gain = s0;
            break;
        }
    }
    if (attn_norm_mul_idx < 0) ATTN_REJECT("no MUL(rms_norm, gain) follower");

    // Step 2: three MUL_MATs off attn_norm_out (Q, K, V). They can appear in
    // any order and there may be VIEWs separating them. Walk forward until we
    // have three whose src[1] strips back to attn_norm_out.
    int q_idx = -1, k_idx = -1, v_idx = -1;
    int mm_found = 0;
    int mm_idxs[3] = {-1, -1, -1};
    struct ggml_tensor * mm_nodes[3] = {nullptr, nullptr, nullptr};
    int scan_limit = std::min(attn_norm_mul_idx + 24, n);
    for (int j = attn_norm_mul_idx + 1; j < scan_limit && mm_found < 3; j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_MUL_MAT) continue;
        struct ggml_tensor * act = xdna_strip_view(nj->src[1]);
        if (act != attn_norm_out) continue;
        mm_idxs[mm_found] = j;
        mm_nodes[mm_found] = nj;
        mm_found++;
    }
    if (mm_found < 3) ATTN_REJECT("fewer than 3 MUL_MATs share attn_norm_out");

    // Identify Q by output feature count: Q has the largest N (embed_dim =
    // num_heads * head_dim); K and V are equal (num_kv_heads * head_dim).
    // Disambiguating K vs V by cgraph index is unreliable (llama.cpp emits
    // Q, V, K order on some archs) — instead we identify K by "has an inbound
    // ROPE", V by elimination, since K gets RoPE'd and V does not.
    int big_slot = 0;
    for (int k = 1; k < 3; k++) {
        if (mm_nodes[k]->ne[0] > mm_nodes[big_slot]->ne[0]) big_slot = k;
    }
    q_idx = mm_idxs[big_slot];
    struct ggml_tensor * q_mm = mm_nodes[big_slot];
    int kv_slots[2]; int kv_fill = 0;
    for (int k = 0; k < 3; k++) if (k != big_slot) kv_slots[kv_fill++] = k;
    struct ggml_tensor * kv_cand[2] = {mm_nodes[kv_slots[0]], mm_nodes[kv_slots[1]]};
    int               kv_cand_idx[2] = {mm_idxs[kv_slots[0]], mm_idxs[kv_slots[1]]};

    if (kv_cand[0]->ne[0] != kv_cand[1]->ne[0]) ATTN_REJECT("K/V projection N mismatch");
    if (q_mm->ne[0] % kv_cand[0]->ne[0] != 0)   ATTN_REJECT("Q/KV head ratio not integer");

    // Step 3: find ROPE on Q and ROPE on K. Follow graph forward; tolerate
    // intervening VIEW/RESHAPE/CONT nodes. Llama-3.2 cgraph emits Q_ROPE
    // (and its reshape) BEFORE K_mm/V_mm, so scan from right after the first
    // MUL_MAT.
    int q_rope_idx = -1, k_rope_idx = -1;
    struct ggml_tensor * q_rope = nullptr, * k_rope = nullptr;
    struct ggml_tensor * inp_pos = nullptr, * rope_freqs = nullptr;
    int k_cand_slot = -1;  // 0 or 1 — which of kv_cand has an inbound ROPE (= K)
    int rope_scan_start = std::min(q_idx, std::min(kv_cand_idx[0], kv_cand_idx[1])) + 1;
    int rope_scan_limit = std::min(std::max(q_idx, std::max(kv_cand_idx[0], kv_cand_idx[1])) + 40, n);
    for (int j = rope_scan_start; j < rope_scan_limit; j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_ROPE) continue;
        struct ggml_tensor * src0 = xdna_strip_view(nj->src[0]);
        struct ggml_tensor * anc = src0;
        for (int steps = 0; anc && steps < 6; steps++) {
            if (anc == q_mm || anc == kv_cand[0] || anc == kv_cand[1]) break;
            if (anc->src[0] == nullptr) break;
            anc = xdna_strip_view(anc->src[0]);
        }
        if (anc == q_mm && q_rope_idx < 0) {
            q_rope_idx = j; q_rope = nj;
        } else if (k_rope_idx < 0) {
            if (anc == kv_cand[0]) { k_rope_idx = j; k_rope = nj; k_cand_slot = 0; }
            else if (anc == kv_cand[1]) { k_rope_idx = j; k_rope = nj; k_cand_slot = 1; }
        }
        if (nj->src[1] && !inp_pos)     inp_pos    = nj->src[1];
        if (nj->src[2] && !rope_freqs)  rope_freqs = nj->src[2];
        if (q_rope_idx >= 0 && k_rope_idx >= 0) break;
    }

    // Now finalize K / V assignment: K is the KV candidate that had a ROPE,
    // V is the other.
    int v_cand_slot = (k_cand_slot == 0) ? 1 : 0;
    k_idx = (k_cand_slot >= 0) ? kv_cand_idx[k_cand_slot] : -1;
    v_idx = (k_cand_slot >= 0) ? kv_cand_idx[v_cand_slot] : -1;
    struct ggml_tensor * k_mm = (k_cand_slot >= 0) ? kv_cand[k_cand_slot] : nullptr;
    struct ggml_tensor * v_mm = (k_cand_slot >= 0) ? kv_cand[v_cand_slot] : nullptr;

    struct ggml_tensor * wq = q_mm->src[0];
    struct ggml_tensor * wk = k_mm ? k_mm->src[0] : nullptr;
    struct ggml_tensor * wv = v_mm ? v_mm->src[0] : nullptr;
    if (!wq || !wk || !wv) ATTN_REJECT("Q/K/V weight tensors null (no ROPE identified K)");

    // --- DIAGNOSTIC: dump cgraph neighborhood when ROPE(Q) or ROPE(K) is not
    // found. Gated on XDNA_DEBUG, capped to 3 attention blocks per process.
    if (dbg && (q_rope_idx < 0 || k_rope_idx < 0)) {
        static std::atomic<int> attn_dbg_remaining{3};
        if (attn_dbg_remaining.fetch_sub(1) > 0) {
            auto shape_str = [](const struct ggml_tensor * t, char * buf, size_t bsz) {
                snprintf(buf, bsz, "[%lld,%lld,%lld,%lld]",
                         (long long)t->ne[0], (long long)t->ne[1],
                         (long long)t->ne[2], (long long)t->ne[3]);
            };
            char qs[64], ks[64], vs[64];
            shape_str(q_mm, qs, sizeof(qs));
            shape_str(k_mm, ks, sizeof(ks));
            shape_str(v_mm, vs, sizeof(vs));
            fprintf(stderr,
                    "ggml-xdna: attn_prefill ROPE-search dump @start=%d "
                    "q_rope_idx=%d k_rope_idx=%d\n",
                    start_idx, q_rope_idx, k_rope_idx);
            fprintf(stderr,
                    "  q_mm=%p op=%s ne=%s (idx=%d)\n"
                    "  k_mm=%p op=%s ne=%s (idx=%d)\n"
                    "  v_mm=%p op=%s ne=%s (idx=%d)\n"
                    "  scan window: [%d, %d)\n",
                    (void*)q_mm, ggml_op_name(q_mm->op), qs, q_idx,
                    (void*)k_mm, ggml_op_name(k_mm->op), ks, k_idx,
                    (void*)v_mm, ggml_op_name(v_mm->op), vs, v_idx,
                    rope_scan_start, rope_scan_limit);
            for (int j = rope_scan_start; j < rope_scan_limit; j++) {
                struct ggml_tensor * nj = cgraph->nodes[j];
                if (nj->op != GGML_OP_ROPE) continue;
                struct ggml_tensor * s0 = nj->src[0];
                struct ggml_tensor * s0r = xdna_strip_view(s0);
                char ss[64], rs[64];
                shape_str(s0, ss, sizeof(ss));
                shape_str(s0r, rs, sizeof(rs));
                fprintf(stderr,
                        "  ROPE@%d: src0=%p op=%s ne=%s -> strip_view=%p op=%s ne=%s",
                        j, (void*)s0, ggml_op_name(s0->op), ss,
                        (void*)s0r, ggml_op_name(s0r->op), rs);
                if (s0r == q_mm)       fprintf(stderr, " [matches q_mm]\n");
                else if (s0r == k_mm)  fprintf(stderr, " [matches k_mm]\n");
                else {
                    fprintf(stderr, " [no direct match, anc chain:]\n");
                    struct ggml_tensor * anc = s0r;
                    for (int steps = 0; anc && steps < 6; steps++) {
                        char as[64];
                        shape_str(anc, as, sizeof(as));
                        const char * tag = "";
                        if (anc == q_mm) tag = " == q_mm";
                        else if (anc == k_mm) tag = " == k_mm";
                        else if (anc == v_mm) tag = " == v_mm";
                        fprintf(stderr,
                                "    step=%d anc=%p op=%s ne=%s%s\n",
                                steps, (void*)anc, ggml_op_name(anc->op), as, tag);
                        if (anc == q_mm || anc == k_mm) break;
                        if (anc->src[0] == nullptr) break;
                        anc = xdna_strip_view(anc->src[0]);
                    }
                }
            }
        }
    }
    // --- END DIAGNOSTIC ---

    if (q_rope_idx < 0) ATTN_REJECT("no ROPE(Q)");
    if (k_rope_idx < 0) ATTN_REJECT("no ROPE(K)");

    // Step 4: FLASH_ATTN_EXT. Scan forward tolerating VIEW/RESHAPE/CONT/
    // SET_ROWS/COPY (KV-cache writes).
    int attn_core_idx = -1;
    struct ggml_tensor * attn_core = nullptr;
    int fa_start = std::max(q_rope_idx, k_rope_idx) + 1;
    int fa_scan_limit = std::min(fa_start + 40, n);
    for (int j = fa_start; j < fa_scan_limit; j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op == GGML_OP_FLASH_ATTN_EXT) {
            attn_core_idx = j; attn_core = nj; break;
        }
        // If we hit a SOFT_MAX before FA, this is an expanded attention
        // graph — unsupported in Phase A.
        if (nj->op == GGML_OP_SOFT_MAX) {
            ATTN_REJECT("expanded attention pattern (SOFT_MAX seen before FLASH_ATTN_EXT) — unsupported in phase A");
        }
    }
    if (attn_core_idx < 0) ATTN_REJECT("no FLASH_ATTN_EXT");

    // Step 5: O projection (MUL_MAT(wo, attn_out)) and residual ADD.
    int o_proj_idx = -1;
    struct ggml_tensor * o_mm = nullptr;
    int o_scan_limit = std::min(attn_core_idx + 16, n);
    for (int j = attn_core_idx + 1; j < o_scan_limit; j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_MUL_MAT) continue;
        struct ggml_tensor * act = xdna_strip_view(nj->src[1]);
        // act should descend from attn_core (FA output).
        struct ggml_tensor * anc = act;
        bool links = false;
        for (int steps = 0; anc && steps < 6; steps++) {
            if (anc == attn_core) { links = true; break; }
            if (anc->src[0] == nullptr) break;
            anc = xdna_strip_view(anc->src[0]);
        }
        if (!links) continue;
        o_proj_idx = j; o_mm = nj; break;
    }
    if (o_proj_idx < 0) ATTN_REJECT("no O MUL_MAT off FA output");
    struct ggml_tensor * wo = o_mm->src[0];
    if (!wo) ATTN_REJECT("O weight null");

    // Residual ADD.
    int residual_add_idx = -1;
    int add_scan_limit = std::min(o_proj_idx + 8, n);
    for (int j = o_proj_idx + 1; j < add_scan_limit; j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_ADD) continue;
        struct ggml_tensor * s0r = xdna_strip_view(nj->src[0]);
        struct ggml_tensor * s1r = xdna_strip_view(nj->src[1]);
        struct ggml_tensor * o_stripped = xdna_strip_view(o_mm);
        struct ggml_tensor * inpL_stripped = xdna_strip_view(inpL);
        bool have_o = (s0r == o_stripped || s0r == o_mm) || (s1r == o_stripped || s1r == o_mm);
        bool have_inpL = (s0r == inpL_stripped || s0r == inpL)
                      || (s1r == inpL_stripped || s1r == inpL);
        if (have_o && have_inpL) {
            residual_add_idx = j;
            break;
        }
    }
    if (residual_add_idx < 0) ATTN_REJECT("no residual ADD(o_proj, inpL)");

    // Shape params.
    const int64_t embed_dim   = inpL->ne[0];
    const int64_t seq_len     = inpL->ne[1];
    const int64_t q_n         = q_mm->ne[0];  // num_heads * head_dim
    const int64_t kv_n        = k_mm->ne[0];  // num_kv_heads * head_dim
    // Head-dim inference: FLASH_ATTN_EXT params don't expose it directly,
    // but ne[0] of the FA output equals embed_dim; q_rope reshape gives
    // head_dim as ne[0] post-reshape. We recover head_dim from q_mm/wq
    // assuming q_n = num_heads * head_dim and num_heads derived from
    // attn_core shape. Defensive fallback: query FA node's src[0] (Q input
    // after rope+reshape) for ne[0].
    int64_t head_dim = 0;
    if (attn_core->src[0]) head_dim = attn_core->src[0]->ne[0];
    if (head_dim <= 0) ATTN_REJECT("could not infer head_dim");
    if (q_n % head_dim != 0) ATTN_REJECT("q_n not divisible by head_dim");
    if (kv_n % head_dim != 0) ATTN_REJECT("kv_n not divisible by head_dim");
    const int64_t num_heads    = q_n / head_dim;
    const int64_t num_kv_heads = kv_n / head_dim;

    out->rms_norm_idx        = start_idx;
    out->attn_norm_mul_idx   = attn_norm_mul_idx;
    out->q_proj_idx          = q_idx;
    out->k_proj_idx          = k_idx;
    out->v_proj_idx          = v_idx;
    out->q_rope_idx          = q_rope_idx;
    out->k_rope_idx          = k_rope_idx;
    out->attn_core_idx       = attn_core_idx;
    out->o_proj_idx          = o_proj_idx;
    out->residual_add_idx    = residual_add_idx;

    out->inpL               = inpL;
    out->gain               = gain;
    out->wq                 = wq;
    out->wk                 = wk;
    out->wv                 = wv;
    out->wo                 = wo;
    out->inp_pos            = inp_pos;
    out->attn_rope_freqs    = rope_freqs;
    out->q_rope_node        = q_rope;

    out->seq_len            = seq_len;
    out->embed_dim          = embed_dim;
    out->num_heads          = num_heads;
    out->num_kv_heads       = num_kv_heads;
    out->head_dim           = head_dim;

    // Translate ggml RoPE mode -> IRON RoPE method_type. Both Q and K rope
    // nodes share the same mode in any sane attention block; we read from Q.
    {
        const int32_t * rp = (const int32_t *)q_rope->op_params;
        const int ggml_mode = rp[2];
        if (ggml_mode == 0) {
            out->rope_method_type = 1;  // ggml NORMAL (adjacent-pair) -> INTERLEAVED
        } else if (ggml_mode == 2) {
            out->rope_method_type = 0;  // ggml NEOX (half-split) -> TWO_HALVES
        } else {
            if (dbg_ok()) fprintf(stderr,
                "ggml-xdna: attn_prefill reject @%d: unsupported RoPE mode %d "
                "(only 0=NORMAL and 2=NEOX supported)\n",
                start_idx, ggml_mode);
            return false;
        }
    }

    #undef ATTN_REJECT
    return true;
}

// ============================================================================
// Attention-prefill Phase B — dispatch the matched RMSNorm+QKV+RoPE+MHA+O+Add
// block onto our chained 11-kernel xclbin. Mirrors the ggml_backend_xdna_
// mul_mat_swiglu template (weight BO caching by tensor->data, persistent per-
// call intermediate BOs, xrt::runlist for atomic chained submit).
// ============================================================================

// Seq-len buckets — must match ATTENTION_PREFILL_SEQ_BUCKETS in compile.py.
static constexpr int64_t XDNA_ATTN_PREFILL_BUCKETS[] = {128, 256, 512, 768, 1024, 1536, 2048, 4096};
static constexpr int XDNA_ATTN_PREFILL_NUM_BUCKETS =
    sizeof(XDNA_ATTN_PREFILL_BUCKETS) / sizeof(XDNA_ATTN_PREFILL_BUCKETS[0]);

static int64_t xdna_select_attention_prefill_bucket(int64_t seq_len) {
    if (seq_len <= 0) return -1;
    for (int i = 0; i < XDNA_ATTN_PREFILL_NUM_BUCKETS; i++) {
        if (seq_len <= XDNA_ATTN_PREFILL_BUCKETS[i]) return XDNA_ATTN_PREFILL_BUCKETS[i];
    }
    return -1;
}

// Mirror attention_prefill_cache_key in compile.py. The compile-side bucket is
// already applied by the caller, so we just hash the same fields. The C++ key
// is a human-readable string (not a SHA-16 hash) for easier on-disk inspection
// — compile.py uses the hash. We let ensure_attention_prefill_compiled pass
// the shape params to compile.py, which will compute its own hash cache-dir.
//
// But: our ctx->cache_dir/<key> bundle layout requires our key to MATCH the
// hash compile.py writes under get_cache_dir()/<hash>/... So we compute the
// same SHA-16 hex that compile.py produces via json.dumps(sort_keys=True).
static std::string make_attention_prefill_cache_key(int64_t seq_bucket,
                                                    int64_t embed_dim,
                                                    int64_t num_heads,
                                                    int64_t num_kv_heads,
                                                    int64_t head_dim,
                                                    int rope_method_type) {
    // We don't try to match compile.py's SHA — instead we pass --out <our_dir>
    // to compile.py so the bundle layout follows our human-readable key.
    // rope_method_type is included so the two RoPE rotations get distinct
    // bundles and never collide on disk.
    char buf[192];
    snprintf(buf, sizeof(buf),
             "attn_prefill_S%lld_E%lld_H%lld_KV%lld_D%lld_bf16_rope%d",
             (long long)seq_bucket, (long long)embed_dim,
             (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
             rope_method_type);
    return std::string(buf);
}

// Check that the full attention-prefill bundle is present on disk.
static bool attention_prefill_bundle_present(const std::string & bundle_dir) {
    std::ifstream xf(bundle_dir + "/combined.xclbin");
    if (!xf.good()) return false;
    for (int s = 0; s < XDNA_ATTN_NUM_SLOTS; s++) {
        std::ifstream f(bundle_dir + "/attn_prefill_" + XDNA_ATTN_PREFILL_INSTS_TAGS[s] + ".insts");
        if (!f.good()) return false;
    }
    return true;
}

static bool ensure_attention_prefill_compiled(ggml_backend_xdna_context * ctx,
                                              const std::string & cache_key,
                                              int64_t seq_bucket,
                                              int64_t embed_dim,
                                              int64_t num_heads,
                                              int64_t num_kv_heads,
                                              int64_t head_dim,
                                              int rope_method_type) {
    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    if (attention_prefill_bundle_present(bundle_dir)) return true;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet attention-prefill --seq-len %lld --embed-dim %lld "
             "--num-heads %lld --num-kv-heads %lld --head-dim %lld "
             "--rope-method-type %d "
             "--out %s%s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)seq_bucket, (long long)embed_dim,
             (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
             rope_method_type,
             bundle_dir.c_str(), xdna_null_redirect());
    fprintf(stderr, "ggml-xdna: compiling attention-prefill S=%lld E=%lld H=%lld KV=%lld d=%lld "
                  "(first run, will be cached)...\n",
                  (long long)seq_bucket, (long long)embed_dim,
                  (long long)num_heads, (long long)num_kv_heads, (long long)head_dim);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: attention-prefill compile failed (exit %d)\n", ret);
        return false;
    }
    if (!attention_prefill_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: attention-prefill compile ran but bundle missing in %s\n",
                       bundle_dir.c_str());
        return false;
    }
    fprintf(stderr, "ggml-xdna: attention-prefill compile complete, cached at %s\n",
                  bundle_dir.c_str());
    return true;
}

static xdna_attention_prefill_entry * get_or_load_attention_prefill_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t seq_bucket, int64_t embed_dim,
        int64_t num_heads, int64_t num_kv_heads, int64_t head_dim) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);
    auto it = ctx->attention_prefill_cache.find(cache_key);
    if (it != ctx->attention_prefill_cache.end()) return &it->second;

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    try {
        xdna_attention_prefill_entry entry;
        entry.op_kind        = XDNA_OP_ATTENTION_PREFILL;
        entry.seq_len_padded = seq_bucket;
        entry.embed_dim      = embed_dim;
        entry.num_heads      = num_heads;
        entry.num_kv_heads   = num_kv_heads;
        entry.head_dim       = head_dim;
        entry.cache_key      = cache_key;

        entry.xclbin = xrt::xclbin(bundle_dir + "/combined.xclbin");
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);

        for (int s = 0; s < XDNA_ATTN_NUM_SLOTS; s++) {
            entry.kernels[s] = xrt::kernel(entry.hw_ctx, XDNA_ATTN_PREFILL_KERNEL_NAMES[s]);
            const std::string insts_path =
                bundle_dir + "/attn_prefill_" + XDNA_ATTN_PREFILL_INSTS_TAGS[s] + ".insts";
            entry.insts_data[s] = read_binary_file(insts_path);
            if (entry.insts_data[s].empty()) {
                GGML_LOG_ERROR("ggml-xdna: failed to read attention-prefill insts: %s\n",
                               insts_path.c_str());
                return nullptr;
            }
            entry.insts_bo[s] = xrt::bo(ctx->device, entry.insts_data[s].size(),
                                         xrt::bo::flags::cacheable,
                                         entry.kernels[s].group_id(1));
            entry.insts_bo[s].write(entry.insts_data[s].data());
            entry.insts_bo[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        auto [ins, _] = ctx->attention_prefill_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded attention-prefill bundle %s\n", cache_key.c_str());
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load attention-prefill bundle: %s\n", e.what());
        return nullptr;
    }
}

// Repack + upload a bf16 weight BO for a GEMM sub-kernel. Cached by the ggml
// weight tensor's data pointer. IRON GEMM wants [K,N] row-major — ggml stores
// weights as [K,N] with ne[0]=K, ne[1]=N row-major (same as the stored
// memory layout). Mirrors swiglu_warm_weight GEMM branch exactly.
// Caller holds entry->weights_mutex.
static xrt::bo * attn_prefill_warm_gemm_weight(
        ggml_backend_xdna_context * ctx,
        xdna_attention_prefill_entry * entry,
        std::unordered_map<const void *, xrt::bo> & cache,
        const struct ggml_tensor * weight,
        int kernel_slot,
        int arg_group_id,
        const char * slot_name) {
    auto it = cache.find(weight->data);
    if (it != cache.end()) return &it->second;

    // ggml stores MUL_MAT weight with ne[0]=K, ne[1]=N (weight is [N,K] in
    // logical sense — first dim is K because ggml transposes). The IRON GEMM
    // B-buffer expects [K,N] row-major. Same transpose as swiglu_warm_weight.
    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];
    const size_t  n_elems = (size_t)K * (size_t)N;
    const size_t  n_bytes = n_elems * sizeof(uint16_t);

    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[kernel_slot].group_id(arg_group_id));
        uint16_t * dst_bf16 = (uint16_t *)new_bo.map<void*>();

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
            // GGML_TYPE_BF16 (treated as 2 bytes/elem) — direct transpose.
            const uint16_t * src = (const uint16_t *)weight->data;
            for (int64_t k = 0; k < K; k++) {
                for (int64_t n = 0; n < N; n++) {
                    dst_bf16[k * N + n] = src[n * K + k];
                }
            }
        }
        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(weight->data, std::move(new_bo));
        fprintf(stderr,
                "ggml-xdna: warm attn-prefill %s K=%lld N=%lld weight=%s (%zu cached)\n",
                slot_name, (long long)K, (long long)N, weight->name, cache.size());
        fflush(stderr);
        return &ins->second;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm attn-prefill %s weight: %s\n",
                       slot_name, e.what());
        return nullptr;
    }
}

// Upload the RMSNorm gain vector (embed_dim bf16). No transpose — just
// per-element convert F32/F16 → bf16. Cached by tensor->data ptr.
// Caller holds entry->weights_mutex.
static xrt::bo * attn_prefill_warm_gain(
        ggml_backend_xdna_context * ctx,
        xdna_attention_prefill_entry * entry,
        const struct ggml_tensor * gain) {
    auto & cache = entry->gain_bo_cache;
    auto it = cache.find(gain->data);
    if (it != cache.end()) return &it->second;

    const int64_t n = entry->embed_dim;
    const size_t  n_bytes = (size_t)n * sizeof(uint16_t);
    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[XDNA_ATTN_RMS_NORM].group_id(4));
        uint16_t * dst = (uint16_t *)new_bo.map<void*>();

        if (gain->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)gain->data, dst, (size_t)n);
        } else if (gain->type == GGML_TYPE_F16) {
            const uint16_t * src = (const uint16_t *)gain->data;
            for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_bf16(src[i]);
        } else {
            memcpy(dst, gain->data, n_bytes);
        }
        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(gain->data, std::move(new_bo));
        fprintf(stderr, "ggml-xdna: warm attn-prefill gain N=%lld weight=%s (%zu cached)\n",
                (long long)n, gain->name, cache.size());
        fflush(stderr);
        return &ins->second;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm attn-prefill gain: %s\n", e.what());
        return nullptr;
    }
}

// ----------------------------------------------------------------------------
// Bulk attention-prefill weight pre-warm.
//
// The per-dispatch path above uploads one weight BO at a time, serialized with
// kernel dispatches. On Llama-3.2-1B that's 4 GEMM weights + 1 gain × 16 layers
// × ~44ms/weight ≈ ~700ms of wall time on the critical path per prefill.
//
// attn_prefill_bulk_prewarm() scans the cgraph for ALL attention-prefill
// matches up-front, resolves each to its entry's cache, and uploads every
// missing weight+gain BO in parallel using std::async. Subsequent dispatch-path
// calls to attn_prefill_warm_gemm_weight/attn_prefill_warm_gain find every
// weight cached and return immediately.
//
// Guarded per-cgraph on ctx->attn_prewarmed_cgraphs so we only scan once.
// ----------------------------------------------------------------------------
struct xdna_attn_prewarm_task {
    xdna_attention_prefill_entry * entry;
    std::unordered_map<const void *, xrt::bo> * cache;
    const struct ggml_tensor * weight;
    int kernel_slot;
    int arg_group_id;
    const char * slot_name;
    bool is_gain;  // gain has its own code path (no transpose)
};

static void attn_prefill_bulk_prewarm(ggml_backend_xdna_context * ctx,
                                      const struct ggml_cgraph * cgraph) {
    // Attention-prefill is only meaningful on a first-pass scan per cgraph ptr.
    // Scan the graph for every RMS_NORM start-index that matches a full attn
    // block with seq_len >= 256 (same gate as the dispatch site).
    std::vector<xdna_attention_match> matches;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_RMS_NORM) continue;
        xdna_attention_match am{};
        if (xdna_try_match_attention_prefill(cgraph, i, &am) && am.seq_len >= 256) {
            matches.push_back(am);
        }
    }
    if (matches.empty()) return;

    // Resolve each match to its entry. Different layers of one model share the
    // same shape → one entry covers them all, but we still build a task per
    // (entry, weight) pair because each weight has a distinct data pointer.
    struct task_t {
        std::function<void()> run;
    };
    std::vector<task_t> tasks;
    tasks.reserve(matches.size() * 5);

    // Dedupe (entry, weight->data) so we don't schedule the same upload twice
    // across matches — paranoia (all 16 Llama layers have distinct weight ptrs,
    // but tied-weight models could collide).
    std::unordered_set<uint64_t> seen;
    auto key_of = [](const xdna_attention_prefill_entry * e, const void * p) {
        return ((uint64_t)(uintptr_t)e) ^ (uint64_t)(uintptr_t)p;
    };

    for (const auto & m : matches) {
        const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
        if (seq_bucket < 0) continue;
        if (m.head_dim != 64) continue;
        if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) continue;
        if ((m.num_kv_heads * m.head_dim) % 64 != 0) continue;
        if (m.num_heads % m.num_kv_heads != 0) continue;

        std::string cache_key = make_attention_prefill_cache_key(
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
            m.rope_method_type);

        // Make sure the xclbin bundle is compiled + loaded into ctx->
        // attention_prefill_cache. This acquires ctx->cache_mutex internally.
        if (!ensure_attention_prefill_compiled(
                ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
                m.num_kv_heads, m.head_dim, m.rope_method_type)) {
            continue;
        }
        xdna_attention_prefill_entry * entry = get_or_load_attention_prefill_kernel(
            ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
            m.num_kv_heads, m.head_dim);
        if (!entry) continue;

        struct slot_desc {
            std::unordered_map<const void *, xrt::bo> * cache;
            const struct ggml_tensor * weight;
            int kernel_slot;
            int arg_group_id;
            const char * slot_name;
            bool is_gain;
        };
        const slot_desc slots[] = {
            { &entry->gain_bo_cache, m.gain, XDNA_ATTN_RMS_NORM, 4, "gain", true  },
            { &entry->w_q_bo_cache,  m.wq,   XDNA_ATTN_GEMM_Q,   4, "w_q",  false },
            { &entry->w_k_bo_cache,  m.wk,   XDNA_ATTN_GEMM_KV,  4, "w_k",  false },
            { &entry->w_v_bo_cache,  m.wv,   XDNA_ATTN_GEMM_KV,  4, "w_v",  false },
            { &entry->w_o_bo_cache,  m.wo,   XDNA_ATTN_GEMM_O,   4, "w_o",  false },
        };
        for (const auto & s : slots) {
            if (!s.weight || !s.weight->data) continue;
            uint64_t k = key_of(entry, s.weight->data);
            if (!seen.insert(k).second) continue;
            // Skip if already cached (can happen across repeat prefill calls
            // if our per-cgraph guard missed — e.g. same cgraph ptr reused).
            {
                std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                if (s.cache->count(s.weight->data)) continue;
            }
            task_t t;
            auto cache_ptr    = s.cache;
            auto weight_ptr   = s.weight;
            int  kslot        = s.kernel_slot;
            int  arg_group    = s.arg_group_id;
            const char * name = s.slot_name;
            bool gain         = s.is_gain;
            t.run = [ctx, entry, cache_ptr, weight_ptr, kslot, arg_group, name, gain]() {
                // Build BO + do the expensive transpose/convert/sync WITHOUT
                // holding entry->weights_mutex — every task gets a distinct
                // weight->data ptr (deduped earlier), so the BO construction
                // and sync are mutually independent and XRT is thread-safe
                // for distinct BOs. Lock only at the very end for the brief
                // cache.emplace, with a re-check in case another worker won.
                if (gain) {
                    const int64_t n = entry->embed_dim;
                    const size_t  n_bytes = (size_t)n * sizeof(uint16_t);
                    try {
                        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                                       entry->kernels[XDNA_ATTN_RMS_NORM].group_id(4));
                        uint16_t * dst = (uint16_t *)new_bo.map<void*>();
                        if (weight_ptr->type == GGML_TYPE_F32) {
                            f32_to_bf16((const float *)weight_ptr->data, dst, (size_t)n);
                        } else if (weight_ptr->type == GGML_TYPE_F16) {
                            const uint16_t * src = (const uint16_t *)weight_ptr->data;
                            for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_bf16(src[i]);
                        } else {
                            memcpy(dst, weight_ptr->data, n_bytes);
                        }
                        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                        if (entry->gain_bo_cache.count(weight_ptr->data) == 0) {
                            entry->gain_bo_cache.emplace(weight_ptr->data, std::move(new_bo));
                        }
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("ggml-xdna: prewarm gain failed: %s\n", e.what());
                    }
                } else {
                    const int64_t K = weight_ptr->ne[0];
                    const int64_t N = weight_ptr->ne[1];
                    const size_t  n_elems = (size_t)K * (size_t)N;
                    const size_t  n_bytes = n_elems * sizeof(uint16_t);
                    try {
                        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                                       entry->kernels[kslot].group_id(arg_group));
                        uint16_t * dst_bf16 = (uint16_t *)new_bo.map<void*>();
                        if (weight_ptr->type == GGML_TYPE_F32) {
                            const float * src_f32 = (const float *)weight_ptr->data;
                            for (int64_t k = 0; k < K; k++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    float val = src_f32[nn * K + k];
                                    uint32_t bits;
                                    memcpy(&bits, &val, sizeof(bits));
                                    bits += (0x7FFF + ((bits >> 16) & 1));
                                    dst_bf16[k * N + nn] = (uint16_t)(bits >> 16);
                                }
                            }
                        } else if (weight_ptr->type == GGML_TYPE_F16) {
                            const uint16_t * src_f16 = (const uint16_t *)weight_ptr->data;
                            for (int64_t k = 0; k < K; k++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    dst_bf16[k * N + nn] = fp16_to_bf16(src_f16[nn * K + k]);
                                }
                            }
                        } else {
                            const uint16_t * src = (const uint16_t *)weight_ptr->data;
                            for (int64_t k = 0; k < K; k++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    dst_bf16[k * N + nn] = src[nn * K + k];
                                }
                            }
                        }
                        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                        if (cache_ptr->count(weight_ptr->data) == 0) {
                            cache_ptr->emplace(weight_ptr->data, std::move(new_bo));
                        }
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("ggml-xdna: prewarm %s failed: %s\n", name, e.what());
                    }
                }
            };
            tasks.push_back(std::move(t));
        }
    }

    if (tasks.empty()) return;

    // Spawn up to hardware_concurrency() workers and feed tasks through a
    // shared atomic index. Keep it simple: one wave of futures, each worker
    // pulls until the queue is drained.
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned nworkers = (unsigned)std::min<size_t>(hw, tasks.size());

    std::atomic<size_t> next{0};
    std::vector<std::future<void>> futs;
    futs.reserve(nworkers);
    for (unsigned w = 0; w < nworkers; w++) {
        futs.push_back(std::async(std::launch::async, [&tasks, &next]() {
            for (;;) {
                size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= tasks.size()) break;
                try {
                    tasks[idx].run();
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("ggml-xdna: attn-prefill bulk prewarm task failed: %s\n",
                                   e.what());
                }
            }
        }));
    }
    for (auto & f : futs) f.wait();

    static const bool ap_dbg = getenv("XDNA_DEBUG") != NULL;
    if (ap_dbg) {
        fprintf(stderr,
                "ggml-xdna: attn-prefill bulk pre-warm done: %zu matches, "
                "%zu uploads, %u workers\n",
                matches.size(), tasks.size(), nworkers);
        fflush(stderr);
    }
}

// Precompute the RoPE angle LUT: shape (seq_len_padded, head_dim) bf16, with
// even cols = cos(pos * inv_freq), odd cols = sin(pos * inv_freq). Matches
// iron/operators/attention_block_prefill/reference.py::_compute_rope_angles AND
// ggml's ggml_rope_cache_init (cache[2k]=cos, cache[2k+1]=sin).
//
// rope_node (if non-null) is the actual GGML_OP_ROPE tensor whose op_params
// drive the angle schedule. We pull:
//   op_params[1] = n_dims   (rotary dims; must equal head_dim for our path)
//   op_params[2] = mode     (GGML_ROPE_TYPE_NEOX=2 for Llama; NORMAL=0 unsupported)
//   op_params[5] = freq_base (theta_base — Llama-3 uses 500000, not 10000)
//   op_params[6] = freq_scale   (must be 1.0)
//   op_params[7] = ext_factor   (must be 0.0)
//   op_params[8] = attn_factor  (must be 1.0)
// and src[2] (freq_factors F32, length >= head_dim/2) if present — per-dim
// divisor on theta (NOT the inv_freq itself; ff=1.0 when absent).
//
// pos_tensor (preferred src[1] of the ROPE node) is a int32 tensor of token
// positions. If rope_node->src[1] is available we use it; else fallback to
// 0..seq_len-1.
//
// Output layout: dst[row*head_dim + 2*i]   = cos(pos[row] * inv_freq[i])
//                dst[row*head_dim + 2*i+1] = sin(pos[row] * inv_freq[i])
// Returns false on any inconsistency the caller should fall back to CPU for.
static bool xdna_compute_rope_angles_bf16(
        const struct ggml_tensor * rope_node,
        int64_t seq_len, int64_t seq_len_padded,
        int64_t head_dim, uint16_t * dst_bf16) {
    if (head_dim <= 0 || head_dim % 2 != 0) return false;
    const int64_t half = head_dim / 2;

    // Defaults — standard RoPE, no YaRN / no rescaling.
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    int   n_dims      = (int)head_dim;
    int   mode        = 2;  // ggml RoPE mode — does not affect angle LUT layout.
    const float * freq_factors = nullptr;
    const struct ggml_tensor * pos_tensor = nullptr;

    if (rope_node) {
        const int32_t * p = (const int32_t *)rope_node->op_params;
        n_dims = p[1];
        mode   = p[2];
        memcpy(&freq_base,   p + 5, sizeof(float));
        memcpy(&freq_scale,  p + 6, sizeof(float));
        memcpy(&ext_factor,  p + 7, sizeof(float));
        memcpy(&attn_factor, p + 8, sizeof(float));

        // mode is dispatched at the IRON RoPE kernel level (method_type 0/1
        // selected at compile time). The angle LUT layout (cos/sin interleaved
        // per row) is identical for both, so we don't gate on mode here.
        (void)mode;

        // Partial rotary (n_dims < head_dim): ggml leaves the tail un-rotated
        // but IRON RoPE rotates the full head_dim. Safe bail-out.
        if (n_dims != head_dim) return false;

        // YaRN / linear-interp / attn-scale: IRON's LUT can't express these.
        if (ext_factor != 0.0f) return false;
        if (std::fabs(freq_scale  - 1.0f) > 1e-6f) return false;
        if (std::fabs(attn_factor - 1.0f) > 1e-6f) return false;

        if (rope_node->src[1] && rope_node->src[1]->type == GGML_TYPE_I32) {
            pos_tensor = rope_node->src[1];
        }
        if (rope_node->src[2]
            && rope_node->src[2]->type == GGML_TYPE_F32
            && rope_node->src[2]->ne[0] >= half) {
            freq_factors = (const float *)rope_node->src[2]->data;
        }
    }

    // Build inv_freq[i]: theta_base^(-2i/n_dims) / freq_factors[i] (if present).
    std::vector<float> inv_freq(half);
    const float theta_scale = std::pow(freq_base, -2.0f / (float)n_dims);
    float theta_i = 1.0f;  // theta_base^(-0/n_dims)
    for (int64_t i = 0; i < half; i++) {
        const float ff = freq_factors ? freq_factors[i] : 1.0f;
        inv_freq[i] = theta_i / ff;
        theta_i *= theta_scale;
    }

    // Resolve pos[row] for row in [0, seq_len). Beyond seq_len, zero.
    std::vector<int32_t> positions(seq_len_padded, 0);
    if (pos_tensor && pos_tensor->ne[0] >= seq_len) {
        const int32_t * src = (const int32_t *)pos_tensor->data;
        for (int64_t i = 0; i < seq_len; i++) positions[i] = src[i];
    } else {
        // Contiguous positions 0..seq_len-1 if ggml didn't hand us an inp_pos.
        for (int64_t i = 0; i < seq_len; i++) positions[i] = (int32_t)i;
    }

    // Fill the LUT. Rows beyond seq_len are left as cos(0)=1, sin(0)=0.
    for (int64_t row = 0; row < seq_len_padded; row++) {
        const float pos_f = (float)positions[row];
        for (int64_t i = 0; i < half; i++) {
            const float angle = pos_f * inv_freq[i];
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            uint32_t cb, sb;
            memcpy(&cb, &c, sizeof(cb));
            memcpy(&sb, &s, sizeof(sb));
            cb += (0x7FFF + ((cb >> 16) & 1));
            sb += (0x7FFF + ((sb >> 16) & 1));
            dst_bf16[row * head_dim + 2*i]     = (uint16_t)(cb >> 16);
            dst_bf16[row * head_dim + 2*i + 1] = (uint16_t)(sb >> 16);
        }
    }
    return true;
}

// Full attention-prefill dispatch. Returns true on success; false lets the
// caller fall back to CPU for this block (Phase B uses a "strict skip on
// success, no change on fail" policy — see graph_compute hook).
// ============================================================================
// Attention-prefill per-phase profiling (gated on XDNA_ATTN_PREFILL_PROFILE=1).
// All buffers and helpers are no-cost when the env var is unset.
//   - Phase samples are aggregated per call to ggml_backend_xdna_graph_compute,
//     reset at entry and printed at exit.
//   - Each sample is in microseconds (int64_t).
// ============================================================================
enum xdna_attn_prof_phase {
    XDNA_AP_ROPE_PRECOMPUTE = 0,
    XDNA_AP_WEIGHT_WARM,
    XDNA_AP_BO_SYNC_TO_DEV,
    XDNA_AP_RL_BUILD,
    XDNA_AP_RL_EXEC,
    XDNA_AP_RL_WAIT,
    XDNA_AP_BO_SYNC_FROM_DEV,
    XDNA_AP_TOTAL_PER_LAYER,
    XDNA_AP_NUM_PHASES,
};

static const char * xdna_attn_prof_phase_label(xdna_attn_prof_phase p) {
    switch (p) {
        case XDNA_AP_ROPE_PRECOMPUTE:  return "rope_precompute  ";
        case XDNA_AP_WEIGHT_WARM:      return "weight_warm      ";
        case XDNA_AP_BO_SYNC_TO_DEV:   return "bo_sync_to_dev   ";
        case XDNA_AP_RL_BUILD:         return "rl_build         ";
        case XDNA_AP_RL_EXEC:          return "rl_exec          ";
        case XDNA_AP_RL_WAIT:          return "rl_wait          ";
        case XDNA_AP_BO_SYNC_FROM_DEV: return "bo_sync_from_dev ";
        case XDNA_AP_TOTAL_PER_LAYER:  return "total_per_layer  ";
        default:                       return "unknown          ";
    }
}

static std::vector<int64_t> g_xdna_attn_prof_samples[XDNA_AP_NUM_PHASES];

static inline bool xdna_attn_prof_enabled() {
    static const bool e = getenv("XDNA_ATTN_PREFILL_PROFILE") != NULL;
    return e;
}

static inline void xdna_attn_prof_reset() {
    if (!xdna_attn_prof_enabled()) return;
    for (int i = 0; i < XDNA_AP_NUM_PHASES; i++) {
        g_xdna_attn_prof_samples[i].clear();
    }
}

static inline void xdna_attn_prof_record(xdna_attn_prof_phase p, int64_t us) {
    g_xdna_attn_prof_samples[p].push_back(us);
}

static void xdna_attn_prof_print() {
    if (!xdna_attn_prof_enabled()) return;
    const size_t n = g_xdna_attn_prof_samples[XDNA_AP_TOTAL_PER_LAYER].size();
    if (n == 0) return;

    fprintf(stderr, "[xdna attn-prefill profile] N=%zu dispatches\n", n);
    for (int p = 0; p < XDNA_AP_NUM_PHASES; p++) {
        auto & v = g_xdna_attn_prof_samples[p];
        if (v.empty()) continue;
        std::vector<int64_t> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        const int64_t med = sorted[sorted.size() / 2];
        const int64_t mn  = sorted.front();
        const int64_t mx  = sorted.back();
        if ((xdna_attn_prof_phase)p == XDNA_AP_TOTAL_PER_LAYER) {
            fprintf(stderr, "  %s: median=%6.2f ms\n",
                    xdna_attn_prof_phase_label((xdna_attn_prof_phase)p),
                    med / 1000.0);
        } else {
            fprintf(stderr, "  %s: median=%6.2f ms   (min=%6.2f  max=%6.2f)\n",
                    xdna_attn_prof_phase_label((xdna_attn_prof_phase)p),
                    med / 1000.0, mn / 1000.0, mx / 1000.0);
        }
    }
    fflush(stderr);
}

static bool ggml_backend_xdna_attention_prefill(ggml_backend_xdna_context * ctx,
                                                const xdna_attention_match & m,
                                                struct ggml_cgraph * cgraph) {
    if (!ctx->device_valid) return false;
    if (m.head_dim != 64) return false;
    if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) return false;
    if ((m.num_kv_heads * m.head_dim) % 64 != 0) return false;
    if (m.num_heads % m.num_kv_heads != 0) return false;

    const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
    if (seq_bucket < 0) return false;

    std::string cache_key = make_attention_prefill_cache_key(
        seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
        m.rope_method_type);

    if (!ensure_attention_prefill_compiled(ctx, cache_key, seq_bucket,
                                           m.embed_dim, m.num_heads,
                                           m.num_kv_heads, m.head_dim,
                                           m.rope_method_type)) {
        return false;
    }

    xdna_attention_prefill_entry * entry = get_or_load_attention_prefill_kernel(
        ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim);
    if (!entry) return false;

    const int64_t S      = m.seq_len;
    const int64_t S_pad  = seq_bucket;       // MHA seq_padding == bucket for our bucket set (all multiples of 64)
    const int64_t E      = m.embed_dim;
    const int64_t H      = m.num_heads;
    const int64_t KV     = m.num_kv_heads;
    const int64_t d      = m.head_dim;
    const int64_t H_d    = H * d;
    const int64_t KV_d   = KV * d;
    const int64_t xE_pad = S_pad * E;
    const int64_t qHD_pad = S_pad * H_d;
    const int64_t kvHD_pad = S_pad * KV_d;
    const int64_t mha_buf = H * d * S_pad;     // MHA arg_spec buffer_size (shared by Q/K/V/O)

    const size_t bytes_xE     = (size_t)xE_pad    * sizeof(uint16_t);
    const size_t bytes_qHD    = (size_t)qHD_pad   * sizeof(uint16_t);
    const size_t bytes_kvHD   = (size_t)kvHD_pad  * sizeof(uint16_t);
    const size_t bytes_mha    = (size_t)mha_buf   * sizeof(uint16_t);
    const size_t bytes_angles = (size_t)S_pad * (size_t)d * sizeof(uint16_t);
    const size_t bytes_actual_xE = (size_t)S * (size_t)E * sizeof(uint16_t);

    // Per-phase profiling clock (gated on XDNA_ATTN_PREFILL_PROFILE=1; cheap when off).
    const bool _ap_prof = xdna_attn_prof_enabled();
    using _ap_clock = std::chrono::steady_clock;
    auto _ap_now = []() { return _ap_clock::now(); };
    auto _ap_us = [](_ap_clock::time_point a, _ap_clock::time_point b) {
        return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const auto _ap_t_dispatch_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};

    try {
        // Lazily allocate all persistent BOs keyed to the sub-kernels' arg-group IDs.
        // GEMM:         (A=3, B=4, C=5)   — matches swiglu_warm_weight prefill mapping.
        // GEMM_Q:       A=x_norm, B=w_q, C=q_proj
        // GEMM_KV:      A=x_norm, B=w_{k,v}, C={k,v}_proj
        // GEMM_O:       A=o_perm, B=w_o, C=o_proj
        // RMS_NORM:     in=3, gain=4, out=5
        // ROPE:         in=3, angles=4, out=5
        // STRIDED_COPY: in=3, out=4  (unary elementwise layout op)
        // MHA:          Q=3, K=4, V=5, O=6
        // ELTWISE_ADD:  in_a=3, in_b=4, out=5
        if (!entry->bo_x_in) {
            entry->bo_x_in = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_RMS_NORM].group_id(3));
        }
        if (!entry->bo_x_norm) {
            entry->bo_x_norm = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_RMS_NORM].group_id(5));
        }
        if (!entry->bo_q_proj) {
            entry->bo_q_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_GEMM_Q].group_id(5));
        }
        if (!entry->bo_k_proj) {
            entry->bo_k_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_GEMM_KV].group_id(5));
        }
        if (!entry->bo_v_proj) {
            entry->bo_v_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_GEMM_KV].group_id(5));
        }
        if (!entry->bo_q_rope) {
            entry->bo_q_rope = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_ROPE_Q].group_id(5));
        }
        if (!entry->bo_k_rope) {
            entry->bo_k_rope = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_ROPE_K].group_id(5));
        }
        // MHA's Q/K/V/O all want a buffer sized H*d*S_pad even though V/K data
        // only fills KV*d*S_pad. Size the permuted BOs to the MHA ask.
        if (!entry->bo_q_perm) {
            entry->bo_q_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_MHA].group_id(3));
        }
        if (!entry->bo_k_perm) {
            entry->bo_k_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_MHA].group_id(4));
        }
        if (!entry->bo_v_perm) {
            entry->bo_v_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_MHA].group_id(5));
        }
        if (!entry->bo_attn_out) {
            entry->bo_attn_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_MHA].group_id(6));
        }
        if (!entry->bo_o_perm) {
            entry->bo_o_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_PERM_O].group_id(4));
        }
        if (!entry->bo_o_proj) {
            entry->bo_o_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_GEMM_O].group_id(5));
        }
        if (!entry->bo_output) {
            entry->bo_output = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_ADD].group_id(5));
        }
        if (!entry->bo_angles) {
            entry->bo_angles = std::make_unique<xrt::bo>(
                ctx->device, bytes_angles, xrt::bo::flags::host_only,
                entry->kernels[XDNA_ATTN_ROPE_Q].group_id(4));
        }

        // Warm weights (cached by tensor->data pointer — one upload per layer).
        xrt::bo * gain_bo = nullptr;
        xrt::bo * wq_bo = nullptr;
        xrt::bo * wk_bo = nullptr;
        xrt::bo * wv_bo = nullptr;
        xrt::bo * wo_bo = nullptr;
        const auto _ap_t_ww_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            gain_bo = attn_prefill_warm_gain(ctx, entry, m.gain);
            wq_bo = attn_prefill_warm_gemm_weight(ctx, entry, entry->w_q_bo_cache,
                                                  m.wq, XDNA_ATTN_GEMM_Q, 4, "w_q");
            wk_bo = attn_prefill_warm_gemm_weight(ctx, entry, entry->w_k_bo_cache,
                                                  m.wk, XDNA_ATTN_GEMM_KV, 4, "w_k");
            wv_bo = attn_prefill_warm_gemm_weight(ctx, entry, entry->w_v_bo_cache,
                                                  m.wv, XDNA_ATTN_GEMM_KV, 4, "w_v");
            wo_bo = attn_prefill_warm_gemm_weight(ctx, entry, entry->w_o_bo_cache,
                                                  m.wo, XDNA_ATTN_GEMM_O, 4, "w_o");
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_WEIGHT_WARM, _ap_us(_ap_t_ww_begin, _ap_now()));
        }
        if (!gain_bo || !wq_bo || !wk_bo || !wv_bo || !wo_bo) return false;

        // Precompute RoPE angles for this dispatch's positions.
        const auto _ap_t_rope_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            std::vector<uint16_t> angles(S_pad * d, 0);
            if (!xdna_compute_rope_angles_bf16(m.q_rope_node,
                                               S, S_pad, d, angles.data())) {
                return false;
            }
            memcpy(entry->bo_angles->map<void*>(), angles.data(),
                   angles.size() * sizeof(uint16_t));
            entry->bo_angles->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_ROPE_PRECOMPUTE, _ap_us(_ap_t_rope_begin, _ap_now()));
        }

        // Upload activation x (= inpL): (S, E) row-major. Zero-pad rows S..S_pad-1.
        const auto _ap_t_in_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        if (S_pad > S) memset(entry->bo_x_in->map<void*>(), 0, bytes_xE);
        {
            void * dst_in = entry->bo_x_in->map<void*>();
            if (m.inpL->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)m.inpL->data,
                            (uint16_t *)dst_in, (size_t)S * (size_t)E);
            } else if (m.inpL->type == GGML_TYPE_F16) {
                const uint16_t * src = (const uint16_t *)m.inpL->data;
                uint16_t * dst = (uint16_t *)dst_in;
                for (int64_t i = 0; i < S * E; i++) dst[i] = fp16_to_bf16(src[i]);
            } else {
                // bf16 — direct memcpy.
                memcpy(dst_in, m.inpL->data, bytes_actual_xE);
            }
            entry->bo_x_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_TO_DEV, _ap_us(_ap_t_in_begin, _ap_now()));
        }

        // Build the 13-run runlist (11 distinct kernel slots; gemm_kv fires
        // twice for K and V, perm_kv fires twice for K and V).
        const uint32_t i_rms  = (uint32_t)entry->insts_data[XDNA_ATTN_RMS_NORM].size();
        const uint32_t i_gq   = (uint32_t)entry->insts_data[XDNA_ATTN_GEMM_Q].size();
        const uint32_t i_gkv  = (uint32_t)entry->insts_data[XDNA_ATTN_GEMM_KV].size();
        const uint32_t i_rq   = (uint32_t)entry->insts_data[XDNA_ATTN_ROPE_Q].size();
        const uint32_t i_rk   = (uint32_t)entry->insts_data[XDNA_ATTN_ROPE_K].size();
        const uint32_t i_pq   = (uint32_t)entry->insts_data[XDNA_ATTN_PERM_Q].size();
        const uint32_t i_pkv  = (uint32_t)entry->insts_data[XDNA_ATTN_PERM_KV].size();
        const uint32_t i_mha  = (uint32_t)entry->insts_data[XDNA_ATTN_MHA].size();
        const uint32_t i_po   = (uint32_t)entry->insts_data[XDNA_ATTN_PERM_O].size();
        const uint32_t i_go   = (uint32_t)entry->insts_data[XDNA_ATTN_GEMM_O].size();
        const uint32_t i_add  = (uint32_t)entry->insts_data[XDNA_ATTN_ADD].size();

        const auto _ap_t_rlbuild_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        xrt::runlist rl(entry->hw_ctx);

        // 1. rms_norm(x_in, gain, x_norm) — arg order: (opcode, insts, isize, in, gain, out)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_RMS_NORM]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_RMS_NORM]); r.set_arg(2, i_rms);
            r.set_arg(3, *entry->bo_x_in);
            r.set_arg(4, *gain_bo);
            r.set_arg(5, *entry->bo_x_norm);
            rl.add(r);
        }
        // 2. gemm_q(x_norm, w_q, q_proj)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_GEMM_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_GEMM_Q]); r.set_arg(2, i_gq);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wq_bo);
            r.set_arg(5, *entry->bo_q_proj);
            rl.add(r);
        }
        // 3. gemm_kv(x_norm, w_k, k_proj)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_GEMM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_GEMM_KV]); r.set_arg(2, i_gkv);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wk_bo);
            r.set_arg(5, *entry->bo_k_proj);
            rl.add(r);
        }
        // 4. gemm_kv(x_norm, w_v, v_proj) — same kernel, different buffers
        {
            xrt::run r(entry->kernels[XDNA_ATTN_GEMM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_GEMM_KV]); r.set_arg(2, i_gkv);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wv_bo);
            r.set_arg(5, *entry->bo_v_proj);
            rl.add(r);
        }
        // 5. rope_q(q_proj, angles, q_rope)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_ROPE_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_ROPE_Q]); r.set_arg(2, i_rq);
            r.set_arg(3, *entry->bo_q_proj);
            r.set_arg(4, *entry->bo_angles);
            r.set_arg(5, *entry->bo_q_rope);
            rl.add(r);
        }
        // 6. rope_k(k_proj, angles, k_rope)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_ROPE_K]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_ROPE_K]); r.set_arg(2, i_rk);
            r.set_arg(3, *entry->bo_k_proj);
            r.set_arg(4, *entry->bo_angles);
            r.set_arg(5, *entry->bo_k_rope);
            rl.add(r);
        }
        // 7. perm_q(q_rope, q_perm)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_PERM_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_PERM_Q]); r.set_arg(2, i_pq);
            r.set_arg(3, *entry->bo_q_rope);
            r.set_arg(4, *entry->bo_q_perm);
            rl.add(r);
        }
        // 8. perm_kv(k_rope, k_perm)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_PERM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_PERM_KV]); r.set_arg(2, i_pkv);
            r.set_arg(3, *entry->bo_k_rope);
            r.set_arg(4, *entry->bo_k_perm);
            rl.add(r);
        }
        // 9. perm_kv(v_proj, v_perm) — V skips RoPE
        {
            xrt::run r(entry->kernels[XDNA_ATTN_PERM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_PERM_KV]); r.set_arg(2, i_pkv);
            r.set_arg(3, *entry->bo_v_proj);
            r.set_arg(4, *entry->bo_v_perm);
            rl.add(r);
        }
        // 10. mha(q_perm, k_perm, v_perm, attn_out)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_MHA]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_MHA]); r.set_arg(2, i_mha);
            r.set_arg(3, *entry->bo_q_perm);
            r.set_arg(4, *entry->bo_k_perm);
            r.set_arg(5, *entry->bo_v_perm);
            r.set_arg(6, *entry->bo_attn_out);
            rl.add(r);
        }
        // 11. perm_o(attn_out, o_perm)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_PERM_O]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_PERM_O]); r.set_arg(2, i_po);
            r.set_arg(3, *entry->bo_attn_out);
            r.set_arg(4, *entry->bo_o_perm);
            rl.add(r);
        }
        // 12. gemm_o(o_perm, w_o, o_proj)
        {
            xrt::run r(entry->kernels[XDNA_ATTN_GEMM_O]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_GEMM_O]); r.set_arg(2, i_go);
            r.set_arg(3, *entry->bo_o_perm);
            r.set_arg(4, *wo_bo);
            r.set_arg(5, *entry->bo_o_proj);
            rl.add(r);
        }
        // 13. add(o_proj, x_in, output) — residual
        {
            xrt::run r(entry->kernels[XDNA_ATTN_ADD]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_ATTN_ADD]); r.set_arg(2, i_add);
            r.set_arg(3, *entry->bo_o_proj);
            r.set_arg(4, *entry->bo_x_in);
            r.set_arg(5, *entry->bo_output);
            rl.add(r);
        }

        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_BUILD, _ap_us(_ap_t_rlbuild_begin, _ap_now()));
        }
        const auto _ap_t_exec_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        rl.execute();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_EXEC, _ap_us(_ap_t_exec_begin, _ap_now()));
        }
        const auto _ap_t_wait_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        rl.wait();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_WAIT, _ap_us(_ap_t_wait_begin, _ap_now()));
        }

        // Pull final output back. Write into cgraph->nodes[residual_add_idx]->data
        // (the residual ADD tensor the matcher anchored on).
        const auto _ap_t_out_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        entry->bo_output->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_FROM_DEV, _ap_us(_ap_t_out_begin, _ap_now()));
        }
        struct ggml_tensor * dst = cgraph->nodes[m.residual_add_idx];
        const size_t actual_elems = (size_t)S * (size_t)E;
        if (dst->type == GGML_TYPE_F32) {
            bf16_to_f32((const uint16_t *)entry->bo_output->map<void*>(),
                        (float *)dst->data, actual_elems);
        } else if (dst->type == GGML_TYPE_BF16) {
            memcpy(dst->data, entry->bo_output->map<void*>(),
                   actual_elems * sizeof(uint16_t));
        } else {
            GGML_LOG_ERROR("ggml-xdna: attn-prefill unsupported dst dtype %d\n",
                           (int)dst->type);
            return false;
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_TOTAL_PER_LAYER,
                                  _ap_us(_ap_t_dispatch_begin, _ap_now()));
        }
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: attention-prefill dispatch failed (%s)\n", e.what());
        return false;
    }
}

// ============================================================================
// TransformerBlockPrefill matcher + dispatch — attention block + SwiGLU FFN +
// two residuals chained into a single combined.xclbin (17 kernels). Extends
// xdna_try_match_attention_prefill by walking forward from the attn residual
// ADD through ffn_norm -> gate/up GEMMs -> SwiGLU GLU -> down GEMM -> ffn
// residual ADD. Dispatch path layers the FFN on top of the attention entry.
// ============================================================================

struct xdna_transformer_block_match {
    xdna_attention_match attn;

    int ffn_norm_idx;
    int ffn_norm_mul_idx;
    int gate_mm_idx;
    int up_mm_idx;
    int glu_idx;
    int down_mm_idx;
    int ffn_residual_add_idx;

    struct ggml_tensor * ffn_gain;
    struct ggml_tensor * w_gate;
    struct ggml_tensor * w_up;
    struct ggml_tensor * w_down;

    int64_t ffn_hidden_dim;
};

static bool xdna_try_match_transformer_block_prefill(
        const struct ggml_cgraph * cgraph, int start_idx,
        xdna_transformer_block_match * out) {
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    static std::atomic<int> dbg_remaining{dbg ? 64 : 0};
    auto dbg_ok = [&]() { return dbg && dbg_remaining.fetch_sub(1) > 0; };

    // Quick decode-shape early-reject. The dispatch gates on seq>=256, but
    // this matcher runs per RMS_NORM per graph_compute (incl. all decode
    // tokens). Cheap seq check avoids the ~40-node attn scan on every decode
    // token.
    {
        struct ggml_tensor * rms = cgraph->nodes[start_idx];
        if (rms->op == GGML_OP_RMS_NORM && rms->src[0] &&
            rms->src[0]->ne[1] < 32) {
            return false;
        }
    }

    xdna_attention_match am{};
    if (!xdna_try_match_attention_prefill(cgraph, start_idx, &am)) return false;

    const int n = cgraph->n_nodes;

    struct ggml_tensor * attn_res = cgraph->nodes[am.residual_add_idx];
    int ffn_norm_idx = -1;
    struct ggml_tensor * ffn_norm = nullptr;
    for (int j = am.residual_add_idx + 1; j < std::min(am.residual_add_idx + 8, n); j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_RMS_NORM) continue;
        struct ggml_tensor * s0r = xdna_strip_view(nj->src[0]);
        if (s0r == attn_res) { ffn_norm_idx = j; ffn_norm = nj; break; }
    }
    if (ffn_norm_idx < 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: no ffn_norm after attn residual\n", start_idx);
        return false;
    }

    int ffn_norm_mul_idx = -1;
    struct ggml_tensor * ffn_norm_out = nullptr;
    struct ggml_tensor * ffn_gain = nullptr;
    for (int j = ffn_norm_idx + 1; j < std::min(ffn_norm_idx + 6, n); j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_MUL) continue;
        struct ggml_tensor * s0r = xdna_strip_view(nj->src[0]);
        struct ggml_tensor * s1r = xdna_strip_view(nj->src[1]);
        if (s0r == ffn_norm) { ffn_norm_mul_idx = j; ffn_norm_out = nj; ffn_gain = nj->src[1]; break; }
        if (s1r == ffn_norm) { ffn_norm_mul_idx = j; ffn_norm_out = nj; ffn_gain = nj->src[0]; break; }
    }
    if (ffn_norm_mul_idx < 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: no MUL(ffn_norm, gain)\n", start_idx);
        return false;
    }

    // W8A16 tblock accepts Q8_0 FFN weights via the uint8 upload branch
    // in tblock_fused_upload_w8a16_weight. When W8A16 is off we stick to
    // bf16/f16/f32 weights, which is what trial.is_int8=false signals.
    static const bool tblock_w8a16_gate_tbl =
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED")) &&
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED_W8A16"));
    int swiglu_anchor = -1;
    xdna_swiglu_match sm{};
    for (int j = ffn_norm_mul_idx + 1; j + 3 < std::min(ffn_norm_mul_idx + 32, n); j++) {
        xdna_swiglu_match trial{};
        if (!xdna_try_match_swiglu(cgraph, j, &trial)) continue;
        struct ggml_tensor * inp = (struct ggml_tensor *)trial.input;
        struct ggml_tensor * inp_stripped = xdna_strip_view(inp);
        if (inp_stripped != ffn_norm_out && inp != ffn_norm_out) continue;
        if (trial.is_int8 && !tblock_w8a16_gate_tbl) continue;
        swiglu_anchor = j;
        sm = trial;
        break;
    }
    if (swiglu_anchor < 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: no SwiGLU window after ffn_norm_mul\n", start_idx);
        return false;
    }

    auto find_idx = [&](struct ggml_tensor * t) -> int {
        for (int j = swiglu_anchor; j < std::min(swiglu_anchor + 16, n); j++) {
            if (cgraph->nodes[j] == t) return j;
        }
        return -1;
    };
    int gate_mm_idx = find_idx(sm.gate_mm);
    int up_mm_idx   = find_idx(sm.up_mm);
    int glu_idx     = find_idx(sm.glu);
    int down_mm_idx = find_idx(sm.down_mm);
    if (gate_mm_idx < 0 || up_mm_idx < 0 || glu_idx < 0 || down_mm_idx < 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: swiglu nodes not in cgraph\n", start_idx);
        return false;
    }

    int ffn_residual_add_idx = -1;
    for (int j = down_mm_idx + 1; j < std::min(down_mm_idx + 8, n); j++) {
        struct ggml_tensor * nj = cgraph->nodes[j];
        if (nj->op != GGML_OP_ADD) continue;
        struct ggml_tensor * s0r = xdna_strip_view(nj->src[0]);
        struct ggml_tensor * s1r = xdna_strip_view(nj->src[1]);
        struct ggml_tensor * down_stripped = xdna_strip_view(sm.down_mm);
        struct ggml_tensor * attn_stripped = xdna_strip_view(attn_res);
        bool have_down = (s0r == down_stripped || s1r == down_stripped
                       || s0r == sm.down_mm || s1r == sm.down_mm);
        bool have_attn = (s0r == attn_stripped || s1r == attn_stripped
                       || s0r == attn_res || s1r == attn_res);
        if (have_down && have_attn) { ffn_residual_add_idx = j; break; }
    }
    if (ffn_residual_add_idx < 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: no FFN residual ADD\n", start_idx);
        return false;
    }

    const int64_t ffn_hidden = sm.gate_w->ne[1];
    if (ffn_hidden % 64 != 0) {
        if (dbg_ok()) fprintf(stderr, "ggml-xdna: tblock reject @%d: ffn_hidden=%lld not mod 64\n",
                              start_idx, (long long)ffn_hidden);
        return false;
    }

    out->attn                 = am;
    out->ffn_norm_idx         = ffn_norm_idx;
    out->ffn_norm_mul_idx     = ffn_norm_mul_idx;
    out->gate_mm_idx          = gate_mm_idx;
    out->up_mm_idx            = up_mm_idx;
    out->glu_idx              = glu_idx;
    out->down_mm_idx          = down_mm_idx;
    out->ffn_residual_add_idx = ffn_residual_add_idx;
    out->ffn_gain             = ffn_gain;
    out->w_gate               = (struct ggml_tensor *)sm.gate_w;
    out->w_up                 = (struct ggml_tensor *)sm.up_w;
    out->w_down               = (struct ggml_tensor *)sm.down_w;
    out->ffn_hidden_dim       = ffn_hidden;
    return true;
}

// ----------------------------------------------------------------------------
// TransformerBlockPrefill cache-key + bundle checks + compile/load helpers.
// Mirrors attention-prefill equivalents with the extended shape fingerprint
// (embed_dim + ffn_hidden_dim).
// ----------------------------------------------------------------------------

static std::string make_transformer_block_prefill_cache_key(
        int64_t seq_bucket, int64_t embed_dim, int64_t num_heads,
        int64_t num_kv_heads, int64_t head_dim, int64_t ffn_hidden_dim,
        int rope_method_type) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "tblock_prefill_S%lld_E%lld_H%lld_KV%lld_D%lld_F%lld_bf16_rope%d",
             (long long)seq_bucket, (long long)embed_dim,
             (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
             (long long)ffn_hidden_dim, rope_method_type);
    return std::string(buf);
}

static bool transformer_block_prefill_bundle_present(const std::string & bundle_dir) {
    std::ifstream xf(bundle_dir + "/combined.xclbin");
    if (!xf.good()) return false;
    for (int s = 0; s < XDNA_TBLOCK_NUM_SLOTS; s++) {
        std::ifstream f(bundle_dir + "/tblock_prefill_" + XDNA_TBLOCK_PREFILL_INSTS_TAGS[s] + ".insts");
        if (!f.good()) return false;
    }
    return true;
}

static bool ensure_transformer_block_prefill_compiled(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t seq_bucket, int64_t embed_dim, int64_t num_heads,
        int64_t num_kv_heads, int64_t head_dim, int64_t ffn_hidden_dim,
        int rope_method_type) {
    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    if (transformer_block_prefill_bundle_present(bundle_dir)) return true;

    char cmd[1280];
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet transformer-block-prefill --seq-len %lld --embed-dim %lld "
             "--num-heads %lld --num-kv-heads %lld --head-dim %lld --ffn-hidden %lld "
             "--rope-method-type %d --out %s%s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)seq_bucket, (long long)embed_dim,
             (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
             (long long)ffn_hidden_dim, rope_method_type, bundle_dir.c_str(), xdna_null_redirect());
    fprintf(stderr, "ggml-xdna: compiling transformer-block-prefill S=%lld E=%lld H=%lld KV=%lld d=%lld F=%lld "
                  "(first run, will be cached)...\n",
                  (long long)seq_bucket, (long long)embed_dim,
                  (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                  (long long)ffn_hidden_dim);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: transformer-block-prefill compile failed (exit %d)\n", ret);
        return false;
    }
    if (!transformer_block_prefill_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: transformer-block-prefill compile ran but bundle missing in %s\n",
                       bundle_dir.c_str());
        return false;
    }
    fprintf(stderr, "ggml-xdna: transformer-block-prefill compile complete, cached at %s\n",
                  bundle_dir.c_str());
    return true;
}

static xdna_transformer_block_prefill_entry * get_or_load_transformer_block_prefill_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t seq_bucket, int64_t embed_dim,
        int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t ffn_hidden_dim) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);
    auto it = ctx->transformer_block_prefill_cache.find(cache_key);
    if (it != ctx->transformer_block_prefill_cache.end()) return &it->second;

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    try {
        xdna_transformer_block_prefill_entry entry;
        entry.seq_len_padded = seq_bucket;
        entry.embed_dim      = embed_dim;
        entry.num_heads      = num_heads;
        entry.num_kv_heads   = num_kv_heads;
        entry.head_dim       = head_dim;
        entry.ffn_hidden_dim = ffn_hidden_dim;
        entry.cache_key      = cache_key;

        entry.xclbin = xrt::xclbin(bundle_dir + "/combined.xclbin");
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);

        for (int s = 0; s < XDNA_TBLOCK_NUM_SLOTS; s++) {
            entry.kernels[s] = xrt::kernel(entry.hw_ctx, XDNA_TBLOCK_PREFILL_KERNEL_NAMES[s]);
            const std::string insts_path =
                bundle_dir + "/tblock_prefill_" + XDNA_TBLOCK_PREFILL_INSTS_TAGS[s] + ".insts";
            entry.insts_data[s] = read_binary_file(insts_path);
            if (entry.insts_data[s].empty()) {
                GGML_LOG_ERROR("ggml-xdna: failed to read tblock-prefill insts: %s\n", insts_path.c_str());
                return nullptr;
            }
            entry.insts_bo[s] = xrt::bo(ctx->device, entry.insts_data[s].size(),
                                         xrt::bo::flags::cacheable,
                                         entry.kernels[s].group_id(1));
            entry.insts_bo[s].write(entry.insts_data[s].data());
            entry.insts_bo[s].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        auto [ins, _] = ctx->transformer_block_prefill_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded transformer-block-prefill bundle %s\n", cache_key.c_str());
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load transformer-block-prefill bundle: %s\n", e.what());
        return nullptr;
    }
}

// ============================================================================
// Layer 4A Phase 3.2 — Fused transformer-block ELF loader.
//
// Single ELF produced by FusedMLIROperator contains the whole transformer
// block (attn + SwiGLU FFN, 20 sub-ops) as a single @main runtime_sequence.
// Runtime: load ELF → hw_context from ELF → xrt::ext::kernel(ctx,
// "main:sequence"). The kernel takes exactly 3 BO args: (input_bo,
// output_bo, scratch_bo). Named tensors (x, w_q, w_k, ..., y) live at
// fixed offsets inside those 3 BOs described by a sidecar JSON layout.
// ============================================================================

// Named buffers in the fused bundle. Order matches layout.json "buffers" dict.
// X / Y are the per-layer activation and the block output; the rest are
// weights that stay in place once written.
static constexpr const char * XDNA_TBF_X           = "x";
static constexpr const char * XDNA_TBF_Y           = "y";
static constexpr const char * XDNA_TBF_W_NORM_ATTN = "w_norm_attn";
static constexpr const char * XDNA_TBF_W_Q         = "w_q";
static constexpr const char * XDNA_TBF_W_K         = "w_k";
static constexpr const char * XDNA_TBF_W_V         = "w_v";
static constexpr const char * XDNA_TBF_W_O         = "w_o";
static constexpr const char * XDNA_TBF_ANGLES      = "angles";
static constexpr const char * XDNA_TBF_W_NORM_FFN  = "w_norm_ffn";
static constexpr const char * XDNA_TBF_W_GATE      = "w_gate";
static constexpr const char * XDNA_TBF_W_UP        = "w_up";
static constexpr const char * XDNA_TBF_W_DOWN      = "w_down";

struct xdna_tblock_fused_layout {
    std::string kernel_name;   // "main:sequence"
    size_t      input_bytes         = 0;
    size_t      dynamic_input_bytes = 0;  // 0 when kernel is 3-BO (legacy)
    size_t      output_bytes        = 0;
    size_t      scratch_bytes       = 0;

    struct buf_info {
        std::string buf_type;  // "input" / "dynamic_input" / "output" / "scratch"
        size_t      offset = 0;
        size_t      length = 0;
        std::string dtype;     // "bfloat16" / "uint8" / ... (from layout JSON)
    };
    std::unordered_map<std::string, buf_info> buffers;
};

struct xdna_tblock_fused_entry {
    xrt::elf        elf;
    xrt::hw_context hw_ctx;
    xrt::kernel     kernel;   // xrt::ext::kernel for "main:sequence"

    xdna_tblock_fused_layout layout;

    // Up to 4 persistent BOs reused across every dispatch of this entry.
    // input_bo         = static weights (norm gains + 7 GEMMs), synced once
    //                    at prewarm and never re-synced per dispatch.
    // dynamic_input_bo = x + RoPE angles (the only inputs that change per
    //                    call). Small (~3 MB) so its per-dispatch sync is
    //                    cheap. When null (legacy 3-BO kernel), x and
    //                    angles live in input_bo instead.
    // output_bo        = y, synced device→host per dispatch.
    // scratch_bo       = on-device only.
    std::unique_ptr<xrt::bo> input_bo;
    std::unique_ptr<xrt::bo> dynamic_input_bo;
    std::unique_ptr<xrt::bo> output_bo;
    std::unique_ptr<xrt::bo> scratch_bo;

    int64_t seq_len_padded  = 0;
    int64_t embed_dim       = 0;
    int64_t num_heads       = 0;
    int64_t num_kv_heads    = 0;
    int64_t head_dim        = 0;
    int64_t ffn_hidden_dim  = 0;
    int     rope_method_type = 0;
    // Layer 4B multi-layer packing: number of consecutive transformer
    // blocks bundled into this entry's ELF. 1 = classic single-block
    // (backwards-compat with Phase 3.7 bundles). N>1 means weight slot
    // names are per-layer suffixed (w_q_0, w_q_1, ...).
    int     num_layers       = 1;
    // W8A16 attention path: the four attention projection weights
    // (w_q/w_k/w_v/w_o, and their per-layer suffixes for N>1) are
    // stored as packed INT8 + bf16 scales in uint8 BO slots. Detected
    // from layout.json via buf_info::dtype == "uint8".
    bool    w8a16            = false;
    // W8A16 FFN (Phase B): when true, FFN weights (w_gate/w_up/w_down)
    // are ALSO packed INT8 + bf16 scales. Implies w8a16. When false and
    // w8a16 is true, we're in A.7 mode: INT8 attn + bf16 FFN.
    bool    w8a16_ffn        = false;
    std::string cache_key;

    // Dedup: for each named buffer, the src pointer whose contents were last
    // written to it. Lets Phase 3.3 skip redundant weight copies when a
    // weight tensor is re-used across iterations.
    std::unordered_map<std::string, const void *> loaded_srcs;

    std::unique_ptr<std::mutex> mu = std::make_unique<std::mutex>();
};

// Layer 4B multi-layer packing group. Points to the entry holding the
// N-block ELF and carries the per-block matches so the dispatcher can
// read x from the FIRST block's inpL, write y to the LAST block's
// ffn_residual_add output, and extend Phase-C over the full 16*N range.
struct xdna_tblock_fused_group {
    int N                = 1;  // number of blocks packed in this group
    int range_begin      = -1; // first block's rms_norm_idx
    int range_end        = -1; // last block's ffn_residual_add_idx
    std::vector<xdna_transformer_block_match> tms;
    xdna_tblock_fused_entry * entry = nullptr;
};

// Returns the desired Layer 4B packing factor. Reads XDNA_ENABLE_TBLOCK_FUSED_N
// and clamps to {1, 2, 4}. Default 1 = classic single-block dispatch
// (backwards-compat; no grouping attempted).
static int xdna_tblock_fused_group_N() {
    const char * v = getenv("XDNA_ENABLE_TBLOCK_FUSED_N");
    if (!v || !*v) return 1;
    int n = atoi(v);
    if (n == 2 || n == 4) return n;
    return 1;
}

// Per-layer weight slot name: "w_q" for N=1 (backwards-compat), "w_q_{L}"
// for N>1. Output is a std::string; callers pass .c_str() when needed.
static std::string tblock_fused_slot_name(const char * base, int layer_idx, int num_layers) {
    if (num_layers <= 1) return std::string(base);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s_%d", base, layer_idx);
    return std::string(buf);
}

// Returns true when the process is configured to use W8A16 fused tblock
// bundles with INT8 attention projections only (A.7). Gated behind
// XDNA_ENABLE_TBLOCK_FUSED_W8A16=1; requires XDNA_ENABLE_TBLOCK_FUSED=1
// (the base tblock-fused path). Does NOT imply INT8 FFN — that is a
// separate gate below.
static bool xdna_tblock_fused_w8a16_enabled() {
    static const bool v = (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED_W8A16")) &&
                          (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED"));
    return v;
}

// Returns true when the process is additionally configured to use Phase B
// bundles (INT8 attention + INT8 FFN). Requires BOTH the base w8a16 gate
// and XDNA_ENABLE_TBLOCK_FUSED_W8A16_FFN=1.
static bool xdna_tblock_fused_w8a16_ffn_enabled() {
    static const bool v = xdna_tblock_fused_w8a16_enabled() &&
                          (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED_W8A16_FFN"));
    return v;
}

static std::string make_tblock_fused_cache_key(
        int64_t seq_bucket, int64_t embed_dim, int64_t num_heads,
        int64_t num_kv_heads, int64_t head_dim, int64_t ffn_hidden_dim,
        int rope_method_type, int num_layers = 1, bool w8a16 = false,
        bool w8a16_ffn = false) {
    GGML_ASSERT(!w8a16_ffn || w8a16);
    char buf[256];
    // v2 = 4-BO split_dynamic layout (x+angles carved into dynamic_input_bo);
    // cache-key bump forces regeneration of pre-split bundles. Bundles
    // predating this change won't parse against the new dispatch path.
    //
    // Layer 4B multi-layer packing appends _N{N} when num_layers > 1.
    // N=1 entries preserve the v2 hash format so existing cached bundles
    // are reused without recompile.
    //
    // W8A16 appends a tag so bf16 cache entries stay untouched. Three modes:
    //   - bf16                : no tag
    //   - A.7 (attn INT8)     : _w8a16    (FFN still bf16)
    //   - B   (attn+FFN INT8) : _w8a16ffn (FFN GEMMs also fused-dequant)
    // The _w8a16ffn tag has 3 additional uint8 buffers in layout.json vs
    // _w8a16, so the split forces a clean cache miss between A.7 and B.
    const char * w8_tag = w8a16_ffn ? "_w8a16ffn" : (w8a16 ? "_w8a16" : "");
    if (num_layers > 1) {
        // v5 = multi-layer packing + fdgemm_i8 4-row default (2026-04-20).
        // v3 added y_L as output BO slots for Phase-C interleave.
        // v5 bumps force a clean recompile so users pick up the 4-row
        // fdgemm_i8 kernel without manually clearing the tblock cache.
        snprintf(buf, sizeof(buf),
                 "tblock_fused_v5%s_N%d_S%lld_E%lld_H%lld_KV%lld_D%lld_F%lld_bf16_rope%d",
                 w8_tag, num_layers,
                 (long long)seq_bucket, (long long)embed_dim,
                 (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                 (long long)ffn_hidden_dim, rope_method_type);
    } else {
        // v4 = single-layer + fdgemm_i8 4-row default (2026-04-20).
        // v2 split x/angles into dynamic_input BO; v4 bumps so users
        // pick up the 4-row fdgemm_i8 kernel (Phase B +26% prefill) on
        // next run without manually clearing the tblock cache.
        snprintf(buf, sizeof(buf),
                 "tblock_fused_v4%s_S%lld_E%lld_H%lld_KV%lld_D%lld_F%lld_bf16_rope%d",
                 w8_tag,
                 (long long)seq_bucket, (long long)embed_dim,
                 (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                 (long long)ffn_hidden_dim, rope_method_type);
    }
    return std::string(buf);
}

static bool tblock_fused_bundle_present(const std::string & bundle_dir,
                                        bool w8a16 = false,
                                        bool w8a16_ffn = false) {
    GGML_ASSERT(!w8a16_ffn || w8a16);
    const std::string suf = w8a16_ffn ? std::string("_w8a16ffn")
                          : (w8a16 ? std::string("_w8a16") : std::string(""));
    std::ifstream elf_f(bundle_dir + "/tblock_fused" + suf + ".elf", std::ios::binary);
    std::ifstream lay_f(bundle_dir + "/tblock_fused" + suf + ".layout.json");
    return elf_f.good() && lay_f.good();
}

static bool ensure_tblock_fused_compiled(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t seq_bucket, int64_t embed_dim, int64_t num_heads,
        int64_t num_kv_heads, int64_t head_dim, int64_t ffn_hidden_dim,
        int rope_method_type, int num_layers = 1, bool w8a16 = false,
        bool w8a16_ffn = false) {
    GGML_ASSERT(!w8a16_ffn || w8a16);
    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    if (tblock_fused_bundle_present(bundle_dir, w8a16, w8a16_ffn)) return true;

    char cmd[1280];
    const char * w8_flag = w8a16_ffn ? " --w8a16 --w8a16-ffn"
                         : (w8a16 ? " --w8a16" : "");
    snprintf(cmd, sizeof(cmd),
             "%s %s --quiet transformer-block-prefill-fused --seq-len %lld --embed-dim %lld "
             "--num-heads %lld --num-kv-heads %lld --head-dim %lld --ffn-hidden %lld "
             "--rope-method-type %d --num-layers %d%s --out %s%s",
             xdna_python_cmd(), ctx->compile_script.c_str(),
             (long long)seq_bucket, (long long)embed_dim,
             (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
             (long long)ffn_hidden_dim, rope_method_type, num_layers, w8_flag,
             bundle_dir.c_str(), xdna_null_redirect());
    const char * mode_tag = w8a16_ffn ? " W8A16+FFN"
                          : (w8a16 ? " W8A16" : "");
    fprintf(stderr, "ggml-xdna: compiling fused tblock S=%lld E=%lld H=%lld KV=%lld d=%lld F=%lld rope=%d N=%d%s "
                  "(first run, will be cached)...\n",
                  (long long)seq_bucket, (long long)embed_dim,
                  (long long)num_heads, (long long)num_kv_heads, (long long)head_dim,
                  (long long)ffn_hidden_dim, rope_method_type, num_layers,
                  mode_tag);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock compile failed (exit %d)\n", ret);
        return false;
    }
    if (!tblock_fused_bundle_present(bundle_dir, w8a16, w8a16_ffn)) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock compile ran but bundle missing in %s\n",
                       bundle_dir.c_str());
        return false;
    }
    fprintf(stderr, "ggml-xdna: fused tblock compile complete, cached at %s\n",
                  bundle_dir.c_str());
    return true;
}

// Tiny hand-rolled JSON parser for tblock_fused.layout.json. The producer is
// `json.dump(layout, indent=2, sort_keys=True)` in compile.py, so the input
// is always well-formed ASCII JSON with no nested escapes beyond basic ones.
// We only need a narrow subset of values; generality isn't worth pulling in
// nlohmann/json as a dependency of ggml-xdna.
namespace tblock_fused_json {

struct cursor {
    const char * p;
    const char * end;
    void skip_ws() {
        while (p < end) {
            char c = *p;
            if (c==' '||c=='\t'||c=='\n'||c=='\r'||c==',') { p++; continue; }
            break;
        }
    }
    bool at(char c) { skip_ws(); return p < end && *p == c; }
    bool eat(char c) { if (at(c)) { p++; return true; } return false; }
};

static bool parse_string(cursor & c, std::string & out) {
    c.skip_ws();
    if (c.p >= c.end || *c.p != '"') return false;
    c.p++;
    out.clear();
    while (c.p < c.end && *c.p != '"') {
        if (*c.p == '\\' && c.p + 1 < c.end) {
            char esc = *(c.p + 1);
            c.p += 2;
            switch (esc) {
                case 'n':  out.push_back('\n'); break;
                case 't':  out.push_back('\t'); break;
                case 'r':  out.push_back('\r'); break;
                case '"':  out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/'); break;
                default:   out.push_back(esc); break;
            }
        } else {
            out.push_back(*c.p++);
        }
    }
    if (c.p >= c.end) return false;
    c.p++;  // closing quote
    return true;
}

static bool parse_int64(cursor & c, int64_t & out) {
    c.skip_ws();
    if (c.p >= c.end) return false;
    char * ep = nullptr;
    long long v = strtoll(c.p, &ep, 10);
    if (ep == c.p) return false;
    out = (int64_t)v;
    c.p = ep;
    return true;
}

// Skip a JSON value at the cursor — object, array, string, number, true/false/null.
static bool skip_value(cursor & c) {
    c.skip_ws();
    if (c.p >= c.end) return false;
    char ch = *c.p;
    if (ch == '"') { std::string tmp; return parse_string(c, tmp); }
    if (ch == '{' || ch == '[') {
        char open = ch, close = (ch == '{') ? '}' : ']';
        int depth = 1;
        c.p++;
        while (c.p < c.end && depth > 0) {
            char x = *c.p;
            if (x == '"') { std::string tmp; if (!parse_string(c, tmp)) return false; continue; }
            if (x == open)  depth++;
            else if (x == close) depth--;
            c.p++;
        }
        return depth == 0;
    }
    // number / bool / null — scan until JSON punctuation.
    while (c.p < c.end) {
        char x = *c.p;
        if (x == ',' || x == '}' || x == ']' || x == '\n' || x == ' ' || x == '\t' || x == '\r') break;
        c.p++;
    }
    return true;
}

// Apply `fn(key)` for each `"key":` entry in the object the cursor starts at.
// `fn` is responsible for consuming the value; if it returns false the scan
// falls back to skip_value for that key.
template <typename Fn>
static bool foreach_object(cursor & c, Fn fn) {
    c.skip_ws();
    if (!c.eat('{')) return false;
    while (true) {
        c.skip_ws();
        if (c.eat('}')) return true;
        std::string key;
        if (!parse_string(c, key)) return false;
        c.skip_ws();
        if (!c.eat(':')) return false;
        // Snapshot before fn in case it fails / refuses the value, so
        // skip_value can resume from the same point.
        const char * before = c.p;
        if (!fn(c, key)) {
            c.p = before;
            if (!skip_value(c)) return false;
        }
    }
}

}  // namespace tblock_fused_json

static bool parse_tblock_fused_layout(
        const std::string & path,
        xdna_tblock_fused_layout & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    std::string data((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    tblock_fused_json::cursor c{ data.data(), data.data() + data.size() };

    bool ok = tblock_fused_json::foreach_object(c, [&](tblock_fused_json::cursor & cc, const std::string & key) {
        using namespace tblock_fused_json;
        if (key == "kernel_name") {
            return parse_string(cc, out.kernel_name);
        }
        if (key == "buffer_sizes") {
            return foreach_object(cc, [&](cursor & cc2, const std::string & bk) {
                int64_t v = 0;
                if (!parse_int64(cc2, v)) return false;
                if      (bk == "input_bytes")         out.input_bytes         = (size_t)v;
                else if (bk == "dynamic_input_bytes") out.dynamic_input_bytes = (size_t)v;
                else if (bk == "output_bytes")        out.output_bytes        = (size_t)v;
                else if (bk == "scratch_bytes")       out.scratch_bytes       = (size_t)v;
                return true;
            });
        }
        if (key == "buffers") {
            return foreach_object(cc, [&](cursor & cc2, const std::string & bname) {
                xdna_tblock_fused_layout::buf_info info;
                bool ok2 = foreach_object(cc2, [&](cursor & cc3, const std::string & fk) {
                    if (fk == "buf_type")     return parse_string(cc3, info.buf_type);
                    if (fk == "dtype")        return parse_string(cc3, info.dtype);
                    int64_t v = 0;
                    if (fk == "offset_bytes") { if (!parse_int64(cc3, v)) return false; info.offset = (size_t)v; return true; }
                    if (fk == "length_bytes") { if (!parse_int64(cc3, v)) return false; info.length = (size_t)v; return true; }
                    return false;  // unknown field → skip_value
                });
                if (!ok2) return false;
                // Default to bfloat16 if dtype absent (legacy bundles predate the
                // dtype field in layout.json).
                if (info.dtype.empty()) info.dtype = "bfloat16";
                out.buffers.emplace(bname, std::move(info));
                return true;
            });
        }
        return false;  // other top-level keys (input_args, output_args, meta, slices, version) → skip_value
    });
    return ok && !out.kernel_name.empty() && out.input_bytes > 0 && out.output_bytes > 0;
}

// The fused bundle is keyed by shape only on disk, but every model layer
// has its own weights. We therefore want one *entry* per layer (so the
// in-BO weight write happens once per process, not once per batch). The
// in-memory map key is the layer-specific key (shape + a weight-tensor
// fingerprint); the on-disk bundle name is just the shape key. Pass both.
static xdna_tblock_fused_entry * get_or_load_tblock_fused_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,       // in-memory map key (layer-specific)
        const std::string & bundle_dir_key,  // on-disk bundle dir name (shape)
        int64_t seq_bucket, int64_t embed_dim,
        int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t ffn_hidden_dim, int rope_method_type, int num_layers = 1,
        bool w8a16 = false, bool w8a16_ffn = false) {
    GGML_ASSERT(!w8a16_ffn || w8a16);
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);
    auto it = ctx->tblock_fused_cache.find(cache_key);
    if (it != ctx->tblock_fused_cache.end()) return &it->second;

    const std::string bundle_dir = ctx->cache_dir + GGML_XDNA_PATH_SEP + bundle_dir_key;
    const std::string suf = w8a16_ffn ? std::string("_w8a16ffn")
                          : (w8a16 ? std::string("_w8a16") : std::string(""));
    const std::string elf_path    = bundle_dir + "/tblock_fused" + suf + ".elf";
    const std::string layout_path = bundle_dir + "/tblock_fused" + suf + ".layout.json";

    try {
        xdna_tblock_fused_entry entry;
        entry.seq_len_padded   = seq_bucket;
        entry.embed_dim        = embed_dim;
        entry.num_heads        = num_heads;
        entry.num_kv_heads     = num_kv_heads;
        entry.head_dim         = head_dim;
        entry.ffn_hidden_dim   = ffn_hidden_dim;
        entry.rope_method_type = rope_method_type;
        entry.num_layers       = num_layers;
        entry.w8a16            = w8a16;
        entry.w8a16_ffn        = w8a16_ffn;
        entry.cache_key        = cache_key;

        if (!parse_tblock_fused_layout(layout_path, entry.layout)) {
            GGML_LOG_ERROR("ggml-xdna: failed to parse fused tblock layout JSON: %s\n",
                           layout_path.c_str());
            return nullptr;
        }

        // Load the ELF and build the hw_context + kernel directly from it.
        // This is the xclbin-free fast path for full-ELF FusedMLIROperator
        // artifacts. "main:sequence" matches the @main runtime_sequence
        // emitted by FusedMLIROperator; compile.py writes it into
        // layout.json but it's fixed by the IRON side and we assert here.
        entry.elf    = xrt::elf(elf_path);
        entry.hw_ctx = xrt::hw_context(ctx->device, entry.elf);
        const std::string & kname = entry.layout.kernel_name.empty()
            ? std::string("main:sequence") : entry.layout.kernel_name;
        entry.kernel = xrt::ext::kernel(entry.hw_ctx, kname);

        // Allocate the persistent BOs via xrt::ext::bo bound to the
        // hw_context — matches FusedMLIROperator's XRTTensor pattern
        // (host_only; cheap sync / unified memory on NPU2). Sizes come
        // straight from the layout JSON. When dynamic_input_bytes > 0
        // the kernel is 4-BO (static weights vs dynamic x+angles split);
        // otherwise it's the legacy 3-BO layout with everything in
        // input_bo.
        entry.input_bo = std::make_unique<xrt::bo>(
            xrt::ext::bo(entry.hw_ctx, entry.layout.input_bytes));
        if (entry.layout.dynamic_input_bytes > 0) {
            entry.dynamic_input_bo = std::make_unique<xrt::bo>(
                xrt::ext::bo(entry.hw_ctx, entry.layout.dynamic_input_bytes));
        }
        entry.output_bo = std::make_unique<xrt::bo>(
            xrt::ext::bo(entry.hw_ctx, entry.layout.output_bytes));
        entry.scratch_bo = std::make_unique<xrt::bo>(
            xrt::ext::bo(entry.hw_ctx, entry.layout.scratch_bytes));

        auto [ins, _] = ctx->tblock_fused_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded fused tblock ELF %s (in=%zu dyn_in=%zu out=%zu scratch=%zu, %zu named bufs)\n",
                      cache_key.c_str(),
                      ins->second.layout.input_bytes,
                      ins->second.layout.dynamic_input_bytes,
                      ins->second.layout.output_bytes,
                      ins->second.layout.scratch_bytes,
                      ins->second.layout.buffers.size());
        return &ins->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load fused tblock bundle: %s\n", e.what());
        return nullptr;
    }
}

// Weight/gain warm helpers — duplicated from attn_prefill variants with the
// entry type swapped. Caller holds entry->weights_mutex.
static xrt::bo * tblock_prefill_warm_gemm_weight(
        ggml_backend_xdna_context * ctx,
        xdna_transformer_block_prefill_entry * entry,
        std::unordered_map<const void *, xrt::bo> & cache,
        const struct ggml_tensor * weight,
        int kernel_slot,
        int arg_group_id,
        const char * slot_name) {
    auto it = cache.find(weight->data);
    if (it != cache.end()) return &it->second;

    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];
    const size_t  n_elems = (size_t)K * (size_t)N;
    const size_t  n_bytes = n_elems * sizeof(uint16_t);

    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[kernel_slot].group_id(arg_group_id));
        uint16_t * dst_bf16 = (uint16_t *)new_bo.map<void*>();

        if (weight->type == GGML_TYPE_F32) {
            const float * src_f32 = (const float *)weight->data;
            for (int64_t k = 0; k < K; k++) {
                for (int64_t nn = 0; nn < N; nn++) {
                    float val = src_f32[nn * K + k];
                    uint32_t bits;
                    memcpy(&bits, &val, sizeof(bits));
                    bits += (0x7FFF + ((bits >> 16) & 1));
                    dst_bf16[k * N + nn] = (uint16_t)(bits >> 16);
                }
            }
        } else if (weight->type == GGML_TYPE_F16) {
            const uint16_t * src_f16 = (const uint16_t *)weight->data;
            for (int64_t k = 0; k < K; k++) {
                for (int64_t nn = 0; nn < N; nn++) {
                    dst_bf16[k * N + nn] = fp16_to_bf16(src_f16[nn * K + k]);
                }
            }
        } else {
            const uint16_t * src = (const uint16_t *)weight->data;
            for (int64_t k = 0; k < K; k++) {
                for (int64_t nn = 0; nn < N; nn++) {
                    dst_bf16[k * N + nn] = src[nn * K + k];
                }
            }
        }
        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(weight->data, std::move(new_bo));
        fprintf(stderr,
                "ggml-xdna: warm tblock-prefill %s K=%lld N=%lld weight=%s (%zu cached)\n",
                slot_name, (long long)K, (long long)N, weight->name, cache.size());
        fflush(stderr);
        return &ins->second;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm tblock-prefill %s weight: %s\n",
                       slot_name, e.what());
        return nullptr;
    }
}

static xrt::bo * tblock_prefill_warm_gain(
        ggml_backend_xdna_context * ctx,
        xdna_transformer_block_prefill_entry * entry,
        std::unordered_map<const void *, xrt::bo> & cache,
        const struct ggml_tensor * gain,
        int64_t n_elems,
        int kernel_slot,
        int arg_group_id) {
    auto it = cache.find(gain->data);
    if (it != cache.end()) return &it->second;

    const size_t n_bytes = (size_t)n_elems * sizeof(uint16_t);
    try {
        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                       entry->kernels[kernel_slot].group_id(arg_group_id));
        uint16_t * dst = (uint16_t *)new_bo.map<void*>();

        if (gain->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)gain->data, dst, (size_t)n_elems);
        } else if (gain->type == GGML_TYPE_F16) {
            const uint16_t * src = (const uint16_t *)gain->data;
            for (int64_t i = 0; i < n_elems; i++) dst[i] = fp16_to_bf16(src[i]);
        } else {
            memcpy(dst, gain->data, n_bytes);
        }
        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto [ins, _] = cache.emplace(gain->data, std::move(new_bo));
        fprintf(stderr, "ggml-xdna: warm tblock-prefill gain N=%lld weight=%s (%zu cached)\n",
                (long long)n_elems, gain->name, cache.size());
        fflush(stderr);
        return &ins->second;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to warm tblock-prefill gain: %s\n", e.what());
        return nullptr;
    }
}

// ----------------------------------------------------------------------------
// Bulk tblock-prefill weight pre-warm. Mirrors attn_prefill_bulk_prewarm but
// walks the graph through xdna_try_match_transformer_block_prefill so the FFN
// weights (w_gate/w_up/w_down + gain_ffn) are uploaded in parallel as well.
// ----------------------------------------------------------------------------
static void tblock_prefill_bulk_prewarm(ggml_backend_xdna_context * ctx,
                                        const struct ggml_cgraph * cgraph) {
    std::vector<xdna_transformer_block_match> matches;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_RMS_NORM) continue;
        xdna_transformer_block_match tm{};
        if (xdna_try_match_transformer_block_prefill(cgraph, i, &tm) && tm.attn.seq_len >= 256) {
            matches.push_back(tm);
        }
    }
    if (matches.empty()) return;

    struct task_t {
        std::function<void()> run;
    };
    std::vector<task_t> tasks;
    tasks.reserve(matches.size() * 9);

    std::unordered_set<uint64_t> seen;
    auto key_of = [](const xdna_transformer_block_prefill_entry * e, const void * p) {
        return ((uint64_t)(uintptr_t)e) ^ (uint64_t)(uintptr_t)p;
    };

    for (const auto & tm : matches) {
        const auto & m = tm.attn;
        const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
        if (seq_bucket < 0) continue;
        if (m.head_dim != 64) continue;
        if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) continue;
        if ((m.num_kv_heads * m.head_dim) % 64 != 0) continue;
        if (m.num_heads % m.num_kv_heads != 0) continue;
        if (tm.ffn_hidden_dim <= 0 || tm.ffn_hidden_dim % 64 != 0) continue;

        std::string cache_key = make_transformer_block_prefill_cache_key(
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
            tm.ffn_hidden_dim, m.rope_method_type);

        if (!ensure_transformer_block_prefill_compiled(
                ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
                m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim, m.rope_method_type)) {
            continue;
        }
        xdna_transformer_block_prefill_entry * entry =
            get_or_load_transformer_block_prefill_kernel(
                ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
                m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim);
        if (!entry) continue;

        struct slot_desc {
            std::unordered_map<const void *, xrt::bo> * cache;
            const struct ggml_tensor * weight;
            int kernel_slot;
            int arg_group_id;
            const char * slot_name;
            bool is_gain;
            int64_t gain_nelems;
        };
        const slot_desc slots[] = {
            { &entry->gain_attn_bo_cache, m.gain,       XDNA_TBLOCK_RMS_NORM_ATTN, 4, "gain_attn", true,  m.embed_dim       },
            { &entry->w_q_bo_cache,       m.wq,         XDNA_TBLOCK_GEMM_Q,        4, "w_q",       false, 0                 },
            { &entry->w_k_bo_cache,       m.wk,         XDNA_TBLOCK_GEMM_KV,       4, "w_k",       false, 0                 },
            { &entry->w_v_bo_cache,       m.wv,         XDNA_TBLOCK_GEMM_KV,       4, "w_v",       false, 0                 },
            { &entry->w_o_bo_cache,       m.wo,         XDNA_TBLOCK_GEMM_O,        4, "w_o",       false, 0                 },
            { &entry->gain_ffn_bo_cache,  tm.ffn_gain,  XDNA_TBLOCK_RMS_NORM_FFN,  4, "gain_ffn",  true,  m.embed_dim       },
            { &entry->w_gate_bo_cache,    tm.w_gate,    XDNA_TBLOCK_GEMM_GATE_UP,  4, "w_gate",    false, 0                 },
            { &entry->w_up_bo_cache,      tm.w_up,      XDNA_TBLOCK_GEMM_GATE_UP,  4, "w_up",      false, 0                 },
            { &entry->w_down_bo_cache,    tm.w_down,    XDNA_TBLOCK_GEMM_DOWN,     4, "w_down",    false, 0                 },
        };
        for (const auto & s : slots) {
            if (!s.weight || !s.weight->data) continue;
            uint64_t k = key_of(entry, s.weight->data);
            if (!seen.insert(k).second) continue;
            {
                std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                if (s.cache->count(s.weight->data)) continue;
            }
            task_t t;
            auto cache_ptr    = s.cache;
            auto weight_ptr   = s.weight;
            int  kslot        = s.kernel_slot;
            int  arg_group    = s.arg_group_id;
            const char * name = s.slot_name;
            bool gain         = s.is_gain;
            int64_t gn        = s.gain_nelems;
            t.run = [ctx, entry, cache_ptr, weight_ptr, kslot, arg_group, name, gain, gn]() {
                if (gain) {
                    const int64_t n = gn;
                    const size_t  n_bytes = (size_t)n * sizeof(uint16_t);
                    try {
                        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                                       entry->kernels[kslot].group_id(arg_group));
                        uint16_t * dst = (uint16_t *)new_bo.map<void*>();
                        if (weight_ptr->type == GGML_TYPE_F32) {
                            f32_to_bf16((const float *)weight_ptr->data, dst, (size_t)n);
                        } else if (weight_ptr->type == GGML_TYPE_F16) {
                            const uint16_t * src = (const uint16_t *)weight_ptr->data;
                            for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_bf16(src[i]);
                        } else {
                            memcpy(dst, weight_ptr->data, n_bytes);
                        }
                        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                        if (cache_ptr->count(weight_ptr->data) == 0) {
                            cache_ptr->emplace(weight_ptr->data, std::move(new_bo));
                        }
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("ggml-xdna: prewarm tblock %s failed: %s\n", name, e.what());
                    }
                } else {
                    const int64_t K = weight_ptr->ne[0];
                    const int64_t N = weight_ptr->ne[1];
                    const size_t  n_elems = (size_t)K * (size_t)N;
                    const size_t  n_bytes = n_elems * sizeof(uint16_t);
                    try {
                        xrt::bo new_bo(ctx->device, n_bytes, xrt::bo::flags::host_only,
                                       entry->kernels[kslot].group_id(arg_group));
                        uint16_t * dst_bf16 = (uint16_t *)new_bo.map<void*>();
                        if (weight_ptr->type == GGML_TYPE_F32) {
                            const float * src_f32 = (const float *)weight_ptr->data;
                            for (int64_t k2 = 0; k2 < K; k2++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    float val = src_f32[nn * K + k2];
                                    uint32_t bits;
                                    memcpy(&bits, &val, sizeof(bits));
                                    bits += (0x7FFF + ((bits >> 16) & 1));
                                    dst_bf16[k2 * N + nn] = (uint16_t)(bits >> 16);
                                }
                            }
                        } else if (weight_ptr->type == GGML_TYPE_F16) {
                            const uint16_t * src_f16 = (const uint16_t *)weight_ptr->data;
                            for (int64_t k2 = 0; k2 < K; k2++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    dst_bf16[k2 * N + nn] = fp16_to_bf16(src_f16[nn * K + k2]);
                                }
                            }
                        } else {
                            const uint16_t * src = (const uint16_t *)weight_ptr->data;
                            for (int64_t k2 = 0; k2 < K; k2++) {
                                for (int64_t nn = 0; nn < N; nn++) {
                                    dst_bf16[k2 * N + nn] = src[nn * K + k2];
                                }
                            }
                        }
                        new_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        std::lock_guard<std::mutex> lock(*entry->weights_mutex);
                        if (cache_ptr->count(weight_ptr->data) == 0) {
                            cache_ptr->emplace(weight_ptr->data, std::move(new_bo));
                        }
                    } catch (const std::exception & e) {
                        GGML_LOG_ERROR("ggml-xdna: prewarm tblock %s failed: %s\n", name, e.what());
                    }
                }
            };
            tasks.push_back(std::move(t));
        }
    }

    if (tasks.empty()) return;

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned nworkers = (unsigned)std::min<size_t>(hw, tasks.size());

    std::atomic<size_t> next{0};
    std::vector<std::future<void>> futs;
    futs.reserve(nworkers);
    for (unsigned w = 0; w < nworkers; w++) {
        futs.push_back(std::async(std::launch::async, [&tasks, &next]() {
            for (;;) {
                size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= tasks.size()) break;
                try {
                    tasks[idx].run();
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("ggml-xdna: tblock-prefill bulk prewarm task failed: %s\n",
                                   e.what());
                }
            }
        }));
    }
    for (auto & f : futs) f.wait();

    static const bool tp_dbg = getenv("XDNA_DEBUG") != NULL;
    if (tp_dbg) {
        fprintf(stderr,
                "ggml-xdna: tblock-prefill bulk pre-warm done: %zu matches, "
                "%zu uploads, %u workers\n",
                matches.size(), tasks.size(), nworkers);
        fflush(stderr);
    }
}

// ----------------------------------------------------------------------------
// TransformerBlockPrefill dispatch — full 20-run chained submit covering the
// attention block + SwiGLU FFN + two residuals.
// ----------------------------------------------------------------------------
static bool ggml_backend_xdna_transformer_block_prefill(
        ggml_backend_xdna_context * ctx,
        const xdna_transformer_block_match & tm,
        struct ggml_cgraph * cgraph) {
    if (!ctx->device_valid) return false;
    const auto & m = tm.attn;
    if (m.head_dim != 64) return false;
    if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) return false;
    if ((m.num_kv_heads * m.head_dim) % 64 != 0) return false;
    if (m.num_heads % m.num_kv_heads != 0) return false;
    if (tm.ffn_hidden_dim <= 0 || tm.ffn_hidden_dim % 64 != 0) return false;

    const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
    if (seq_bucket < 0) return false;

    std::string cache_key = make_transformer_block_prefill_cache_key(
        seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
        tm.ffn_hidden_dim, m.rope_method_type);

    if (!ensure_transformer_block_prefill_compiled(
            ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
            m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim, m.rope_method_type)) {
        return false;
    }

    xdna_transformer_block_prefill_entry * entry =
        get_or_load_transformer_block_prefill_kernel(
            ctx, cache_key, seq_bucket, m.embed_dim, m.num_heads,
            m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim);
    if (!entry) return false;

    const int64_t S      = m.seq_len;
    const int64_t S_pad  = seq_bucket;
    const int64_t E      = m.embed_dim;
    const int64_t H      = m.num_heads;
    const int64_t KV     = m.num_kv_heads;
    const int64_t d      = m.head_dim;
    const int64_t F      = tm.ffn_hidden_dim;
    const int64_t H_d    = H * d;
    const int64_t KV_d   = KV * d;
    const int64_t xE_pad   = S_pad * E;
    const int64_t qHD_pad  = S_pad * H_d;
    const int64_t kvHD_pad = S_pad * KV_d;
    const int64_t mha_buf  = H * d * S_pad;
    const int64_t xF_pad   = S_pad * F;

    const size_t bytes_xE     = (size_t)xE_pad    * sizeof(uint16_t);
    const size_t bytes_qHD    = (size_t)qHD_pad   * sizeof(uint16_t);
    const size_t bytes_kvHD   = (size_t)kvHD_pad  * sizeof(uint16_t);
    const size_t bytes_mha    = (size_t)mha_buf   * sizeof(uint16_t);
    const size_t bytes_angles = (size_t)S_pad * (size_t)d * sizeof(uint16_t);
    const size_t bytes_xF     = (size_t)xF_pad    * sizeof(uint16_t);
    const size_t bytes_actual_xE = (size_t)S * (size_t)E * sizeof(uint16_t);

    const bool _ap_prof = xdna_attn_prof_enabled();
    using _ap_clock = std::chrono::steady_clock;
    auto _ap_now = []() { return _ap_clock::now(); };
    auto _ap_us = [](_ap_clock::time_point a, _ap_clock::time_point b) {
        return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const auto _ap_t_dispatch_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};

    try {
        // Attention-side BOs.
        if (!entry->bo_x_in) {
            entry->bo_x_in = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_RMS_NORM_ATTN].group_id(3));
        }
        if (!entry->bo_x_norm) {
            entry->bo_x_norm = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_RMS_NORM_ATTN].group_id(5));
        }
        if (!entry->bo_q_proj) {
            entry->bo_q_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_Q].group_id(5));
        }
        if (!entry->bo_k_proj) {
            entry->bo_k_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_KV].group_id(5));
        }
        if (!entry->bo_v_proj) {
            entry->bo_v_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_KV].group_id(5));
        }
        if (!entry->bo_q_rope) {
            entry->bo_q_rope = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ROPE_Q].group_id(5));
        }
        if (!entry->bo_k_rope) {
            entry->bo_k_rope = std::make_unique<xrt::bo>(
                ctx->device, bytes_kvHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ROPE_K].group_id(5));
        }
        if (!entry->bo_q_perm) {
            entry->bo_q_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_MHA].group_id(3));
        }
        if (!entry->bo_k_perm) {
            entry->bo_k_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_MHA].group_id(4));
        }
        if (!entry->bo_v_perm) {
            entry->bo_v_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_MHA].group_id(5));
        }
        if (!entry->bo_attn_out) {
            entry->bo_attn_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_mha, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_MHA].group_id(6));
        }
        if (!entry->bo_o_perm) {
            entry->bo_o_perm = std::make_unique<xrt::bo>(
                ctx->device, bytes_qHD, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_PERM_O].group_id(4));
        }
        if (!entry->bo_o_proj) {
            entry->bo_o_proj = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_O].group_id(5));
        }
        if (!entry->bo_attn_residual) {
            entry->bo_attn_residual = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ADD_ATTN].group_id(5));
        }
        if (!entry->bo_angles) {
            entry->bo_angles = std::make_unique<xrt::bo>(
                ctx->device, bytes_angles, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ROPE_Q].group_id(4));
        }
        // FFN-side BOs.
        if (!entry->bo_ffn_norm) {
            entry->bo_ffn_norm = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_RMS_NORM_FFN].group_id(5));
        }
        if (!entry->bo_gate_out) {
            entry->bo_gate_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_xF, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_GATE_UP].group_id(5));
        }
        if (!entry->bo_up_out) {
            entry->bo_up_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_xF, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_GATE_UP].group_id(5));
        }
        if (!entry->bo_silu_out) {
            entry->bo_silu_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_xF, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_SILU].group_id(4));
        }
        if (!entry->bo_mul_out) {
            entry->bo_mul_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_xF, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ELTWISE_MUL].group_id(5));
        }
        if (!entry->bo_down_out) {
            entry->bo_down_out = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_GEMM_DOWN].group_id(5));
        }
        if (!entry->bo_output) {
            entry->bo_output = std::make_unique<xrt::bo>(
                ctx->device, bytes_xE, xrt::bo::flags::host_only,
                entry->kernels[XDNA_TBLOCK_ADD_FFN].group_id(5));
        }

        xrt::bo * gain_attn_bo = nullptr;
        xrt::bo * wq_bo = nullptr;
        xrt::bo * wk_bo = nullptr;
        xrt::bo * wv_bo = nullptr;
        xrt::bo * wo_bo = nullptr;
        xrt::bo * gain_ffn_bo = nullptr;
        xrt::bo * wgate_bo = nullptr;
        xrt::bo * wup_bo = nullptr;
        xrt::bo * wdown_bo = nullptr;
        const auto _ap_t_ww_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            std::lock_guard<std::mutex> lock(*entry->weights_mutex);
            gain_attn_bo = tblock_prefill_warm_gain(
                ctx, entry, entry->gain_attn_bo_cache, m.gain, E,
                XDNA_TBLOCK_RMS_NORM_ATTN, 4);
            wq_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_q_bo_cache, m.wq,
                XDNA_TBLOCK_GEMM_Q, 4, "w_q");
            wk_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_k_bo_cache, m.wk,
                XDNA_TBLOCK_GEMM_KV, 4, "w_k");
            wv_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_v_bo_cache, m.wv,
                XDNA_TBLOCK_GEMM_KV, 4, "w_v");
            wo_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_o_bo_cache, m.wo,
                XDNA_TBLOCK_GEMM_O, 4, "w_o");
            gain_ffn_bo = tblock_prefill_warm_gain(
                ctx, entry, entry->gain_ffn_bo_cache, tm.ffn_gain, E,
                XDNA_TBLOCK_RMS_NORM_FFN, 4);
            wgate_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_gate_bo_cache, tm.w_gate,
                XDNA_TBLOCK_GEMM_GATE_UP, 4, "w_gate");
            wup_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_up_bo_cache, tm.w_up,
                XDNA_TBLOCK_GEMM_GATE_UP, 4, "w_up");
            wdown_bo = tblock_prefill_warm_gemm_weight(
                ctx, entry, entry->w_down_bo_cache, tm.w_down,
                XDNA_TBLOCK_GEMM_DOWN, 4, "w_down");
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_WEIGHT_WARM, _ap_us(_ap_t_ww_begin, _ap_now()));
        }
        if (!gain_attn_bo || !wq_bo || !wk_bo || !wv_bo || !wo_bo
            || !gain_ffn_bo || !wgate_bo || !wup_bo || !wdown_bo) return false;

        const auto _ap_t_rope_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            std::vector<uint16_t> angles(S_pad * d, 0);
            if (!xdna_compute_rope_angles_bf16(m.q_rope_node,
                                               S, S_pad, d, angles.data())) {
                return false;
            }
            memcpy(entry->bo_angles->map<void*>(), angles.data(),
                   angles.size() * sizeof(uint16_t));
            entry->bo_angles->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_ROPE_PRECOMPUTE, _ap_us(_ap_t_rope_begin, _ap_now()));
        }

        const auto _ap_t_in_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        if (S_pad > S) memset(entry->bo_x_in->map<void*>(), 0, bytes_xE);
        {
            void * dst_in = entry->bo_x_in->map<void*>();
            if (m.inpL->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)m.inpL->data,
                            (uint16_t *)dst_in, (size_t)S * (size_t)E);
            } else if (m.inpL->type == GGML_TYPE_F16) {
                const uint16_t * src = (const uint16_t *)m.inpL->data;
                uint16_t * dst = (uint16_t *)dst_in;
                for (int64_t i = 0; i < S * E; i++) dst[i] = fp16_to_bf16(src[i]);
            } else {
                memcpy(dst_in, m.inpL->data, bytes_actual_xE);
            }
            entry->bo_x_in->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_TO_DEV, _ap_us(_ap_t_in_begin, _ap_now()));
        }

        const uint32_t i_rms_a  = (uint32_t)entry->insts_data[XDNA_TBLOCK_RMS_NORM_ATTN].size();
        const uint32_t i_gq     = (uint32_t)entry->insts_data[XDNA_TBLOCK_GEMM_Q].size();
        const uint32_t i_gkv    = (uint32_t)entry->insts_data[XDNA_TBLOCK_GEMM_KV].size();
        const uint32_t i_rq     = (uint32_t)entry->insts_data[XDNA_TBLOCK_ROPE_Q].size();
        const uint32_t i_rk     = (uint32_t)entry->insts_data[XDNA_TBLOCK_ROPE_K].size();
        const uint32_t i_pq     = (uint32_t)entry->insts_data[XDNA_TBLOCK_PERM_Q].size();
        const uint32_t i_pkv    = (uint32_t)entry->insts_data[XDNA_TBLOCK_PERM_KV].size();
        const uint32_t i_mha    = (uint32_t)entry->insts_data[XDNA_TBLOCK_MHA].size();
        const uint32_t i_po     = (uint32_t)entry->insts_data[XDNA_TBLOCK_PERM_O].size();
        const uint32_t i_go     = (uint32_t)entry->insts_data[XDNA_TBLOCK_GEMM_O].size();
        const uint32_t i_add_a  = (uint32_t)entry->insts_data[XDNA_TBLOCK_ADD_ATTN].size();
        const uint32_t i_rms_f  = (uint32_t)entry->insts_data[XDNA_TBLOCK_RMS_NORM_FFN].size();
        const uint32_t i_ggu    = (uint32_t)entry->insts_data[XDNA_TBLOCK_GEMM_GATE_UP].size();
        const uint32_t i_silu   = (uint32_t)entry->insts_data[XDNA_TBLOCK_SILU].size();
        const uint32_t i_emul   = (uint32_t)entry->insts_data[XDNA_TBLOCK_ELTWISE_MUL].size();
        const uint32_t i_gdn    = (uint32_t)entry->insts_data[XDNA_TBLOCK_GEMM_DOWN].size();
        const uint32_t i_add_f  = (uint32_t)entry->insts_data[XDNA_TBLOCK_ADD_FFN].size();

        const auto _ap_t_rlbuild_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        xrt::runlist rl(entry->hw_ctx);

        // 1. rms_norm_attn(x_in, gain_attn, x_norm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_RMS_NORM_ATTN]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_RMS_NORM_ATTN]); r.set_arg(2, i_rms_a);
            r.set_arg(3, *entry->bo_x_in);
            r.set_arg(4, *gain_attn_bo);
            r.set_arg(5, *entry->bo_x_norm);
            rl.add(r);
        }
        // 2. gemm_q(x_norm, w_q, q_proj)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_Q]); r.set_arg(2, i_gq);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wq_bo);
            r.set_arg(5, *entry->bo_q_proj);
            rl.add(r);
        }
        // 3. gemm_kv(x_norm, w_k, k_proj)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_KV]); r.set_arg(2, i_gkv);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wk_bo);
            r.set_arg(5, *entry->bo_k_proj);
            rl.add(r);
        }
        // 4. gemm_kv(x_norm, w_v, v_proj)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_KV]); r.set_arg(2, i_gkv);
            r.set_arg(3, *entry->bo_x_norm);
            r.set_arg(4, *wv_bo);
            r.set_arg(5, *entry->bo_v_proj);
            rl.add(r);
        }
        // 5. rope_q(q_proj, angles, q_rope)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_ROPE_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_ROPE_Q]); r.set_arg(2, i_rq);
            r.set_arg(3, *entry->bo_q_proj);
            r.set_arg(4, *entry->bo_angles);
            r.set_arg(5, *entry->bo_q_rope);
            rl.add(r);
        }
        // 6. rope_k(k_proj, angles, k_rope)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_ROPE_K]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_ROPE_K]); r.set_arg(2, i_rk);
            r.set_arg(3, *entry->bo_k_proj);
            r.set_arg(4, *entry->bo_angles);
            r.set_arg(5, *entry->bo_k_rope);
            rl.add(r);
        }
        // 7. perm_q(q_rope, q_perm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_PERM_Q]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_PERM_Q]); r.set_arg(2, i_pq);
            r.set_arg(3, *entry->bo_q_rope);
            r.set_arg(4, *entry->bo_q_perm);
            rl.add(r);
        }
        // 8. perm_kv(k_rope, k_perm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_PERM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_PERM_KV]); r.set_arg(2, i_pkv);
            r.set_arg(3, *entry->bo_k_rope);
            r.set_arg(4, *entry->bo_k_perm);
            rl.add(r);
        }
        // 9. perm_kv(v_proj, v_perm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_PERM_KV]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_PERM_KV]); r.set_arg(2, i_pkv);
            r.set_arg(3, *entry->bo_v_proj);
            r.set_arg(4, *entry->bo_v_perm);
            rl.add(r);
        }
        // 10. mha(q_perm, k_perm, v_perm, attn_out)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_MHA]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_MHA]); r.set_arg(2, i_mha);
            r.set_arg(3, *entry->bo_q_perm);
            r.set_arg(4, *entry->bo_k_perm);
            r.set_arg(5, *entry->bo_v_perm);
            r.set_arg(6, *entry->bo_attn_out);
            rl.add(r);
        }
        // 11. perm_o(attn_out, o_perm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_PERM_O]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_PERM_O]); r.set_arg(2, i_po);
            r.set_arg(3, *entry->bo_attn_out);
            r.set_arg(4, *entry->bo_o_perm);
            rl.add(r);
        }
        // 12. gemm_o(o_perm, w_o, o_proj)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_O]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_O]); r.set_arg(2, i_go);
            r.set_arg(3, *entry->bo_o_perm);
            r.set_arg(4, *wo_bo);
            r.set_arg(5, *entry->bo_o_proj);
            rl.add(r);
        }
        // 13. add_attn(o_proj, x_in, attn_residual)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_ADD_ATTN]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_ADD_ATTN]); r.set_arg(2, i_add_a);
            r.set_arg(3, *entry->bo_o_proj);
            r.set_arg(4, *entry->bo_x_in);
            r.set_arg(5, *entry->bo_attn_residual);
            rl.add(r);
        }
        // 14. rms_norm_ffn(attn_residual, gain_ffn, ffn_norm)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_RMS_NORM_FFN]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_RMS_NORM_FFN]); r.set_arg(2, i_rms_f);
            r.set_arg(3, *entry->bo_attn_residual);
            r.set_arg(4, *gain_ffn_bo);
            r.set_arg(5, *entry->bo_ffn_norm);
            rl.add(r);
        }
        // 15. gemm_gate_up(ffn_norm, w_gate, gate_out)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_GATE_UP]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_GATE_UP]); r.set_arg(2, i_ggu);
            r.set_arg(3, *entry->bo_ffn_norm);
            r.set_arg(4, *wgate_bo);
            r.set_arg(5, *entry->bo_gate_out);
            rl.add(r);
        }
        // 16. gemm_gate_up(ffn_norm, w_up, up_out)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_GATE_UP]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_GATE_UP]); r.set_arg(2, i_ggu);
            r.set_arg(3, *entry->bo_ffn_norm);
            r.set_arg(4, *wup_bo);
            r.set_arg(5, *entry->bo_up_out);
            rl.add(r);
        }
        // 17. silu(gate_out, silu_out) — unary: in=3, out=4
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_SILU]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_SILU]); r.set_arg(2, i_silu);
            r.set_arg(3, *entry->bo_gate_out);
            r.set_arg(4, *entry->bo_silu_out);
            rl.add(r);
        }
        // 18. eltwise_mul(silu_out, up_out, mul_out)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_ELTWISE_MUL]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_ELTWISE_MUL]); r.set_arg(2, i_emul);
            r.set_arg(3, *entry->bo_silu_out);
            r.set_arg(4, *entry->bo_up_out);
            r.set_arg(5, *entry->bo_mul_out);
            rl.add(r);
        }
        // 19. gemm_down(mul_out, w_down, down_out)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_GEMM_DOWN]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_GEMM_DOWN]); r.set_arg(2, i_gdn);
            r.set_arg(3, *entry->bo_mul_out);
            r.set_arg(4, *wdown_bo);
            r.set_arg(5, *entry->bo_down_out);
            rl.add(r);
        }
        // 20. add_ffn(down_out, attn_residual, output)
        {
            xrt::run r(entry->kernels[XDNA_TBLOCK_ADD_FFN]);
            r.set_arg(0, 3u); r.set_arg(1, entry->insts_bo[XDNA_TBLOCK_ADD_FFN]); r.set_arg(2, i_add_f);
            r.set_arg(3, *entry->bo_down_out);
            r.set_arg(4, *entry->bo_attn_residual);
            r.set_arg(5, *entry->bo_output);
            rl.add(r);
        }

        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_BUILD, _ap_us(_ap_t_rlbuild_begin, _ap_now()));
        }
        const auto _ap_t_exec_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        rl.execute();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_EXEC, _ap_us(_ap_t_exec_begin, _ap_now()));
        }
        const auto _ap_t_wait_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        rl.wait();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_WAIT, _ap_us(_ap_t_wait_begin, _ap_now()));
        }

        const auto _ap_t_out_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        entry->bo_output->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_FROM_DEV, _ap_us(_ap_t_out_begin, _ap_now()));
        }
        struct ggml_tensor * dst = cgraph->nodes[tm.ffn_residual_add_idx];
        const size_t actual_elems = (size_t)S * (size_t)E;
        if (dst->type == GGML_TYPE_F32) {
            bf16_to_f32((const uint16_t *)entry->bo_output->map<void*>(),
                        (float *)dst->data, actual_elems);
        } else if (dst->type == GGML_TYPE_BF16) {
            memcpy(dst->data, entry->bo_output->map<void*>(),
                   actual_elems * sizeof(uint16_t));
        } else {
            GGML_LOG_ERROR("ggml-xdna: tblock-prefill unsupported dst dtype %d\n",
                           (int)dst->type);
            return false;
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_TOTAL_PER_LAYER,
                                  _ap_us(_ap_t_dispatch_begin, _ap_now()));
        }
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: transformer-block-prefill dispatch failed (%s)\n", e.what());
        return false;
    }
}

// ============================================================================
// Layer 4A Phase 3.3 — Fused transformer-block dispatch.
//
// Same matcher as Layer 3A (attn + SwiGLU FFN window), but the whole block
// runs as a SINGLE xrt::run against one ELF. Host work per layer collapses
// to: write x at its input_bo offset, compute + write RoPE angles, sync
// input_bo, run, sync output_bo, copy y out.
//
// Weights (w_q, w_k, w_v, w_o, w_gate, w_up, w_down, w_norm_attn,
// w_norm_ffn) are uploaded once per layer-lifetime at the entry's
// input_bo offsets. To get that, we key the in-memory cache by layer
// (shape + w_q->data ptr) while keeping a single on-disk bundle per
// shape — the ELF + layout are identical across layers.
// ============================================================================

static std::string make_tblock_fused_layer_cache_key(
        const std::string & shape_key, const void * wq_ptr) {
    char buf[48];
    snprintf(buf, sizeof(buf), "_wq%lx", (unsigned long)(uintptr_t)wq_ptr);
    return shape_key + buf;
}

// ============================================================================
// W8A16 host-side weight packer.
//
// Mirrors the IRON reference packer at
//   iron/operators/fused_dequant_gemm_i8/reference.py::pack_weights
// byte-for-byte. The packed buffer is a uint8 byte stream laid out as:
//
//   for col in [0, cols):
//     for n_tile in [0, n_per_col / tile_n):
//       for k_tile in [0, K / tile_k):
//         [tile_k * tile_n bytes int8, row-major (k_in_tile, n_in_tile)]
//         [(tile_k/G) * tile_n bytes bf16, row-major (g_in_tile, n_in_tile)]
//
// Per group g, per n: scale = max(|B[g*G:(g+1)*G, n]|) / 127.
// Symmetric, zero-point=0. All-zero groups pick scale=1.0 to avoid
// divide-by-zero. int8 = round(B / scale) clamped to [-128, 127].
//
// Input B is supplied as the `(K, N)` bf16 "logical" orientation (K along
// axis 0). ggml stores weights as (N, K), so callers transpose on the
// fly when converting F16/F32 -> bf16 (K, N) below.
//
// NOTE: tile_k / tile_n / group_size are pinned to the defaults the IRON
// FusedDequantGEMMi8 op uses (64, 64, 32). If IRON ever changes those
// defaults this packer will silently desync — keep them in agreement.
// ============================================================================

static constexpr int XDNA_TBF_W8A16_TILE_K     = 64;
static constexpr int XDNA_TBF_W8A16_TILE_N     = 64;
static constexpr int XDNA_TBF_W8A16_GROUP_SIZE = 32;

// Convert fp32 -> bf16 raw bits with round-to-nearest-even.
static inline uint16_t xdna_f32_to_bf16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    // Round-to-nearest-even: bump by 0x7FFF + lsb of the truncated high half.
    bits += (0x7FFFu + ((bits >> 16) & 1u));
    return (uint16_t)(bits >> 16);
}

// Compute the total packed byte length for a W8A16 weight of shape (K, N)
// with given cols. Matches the byte count the IRON op's uint8 arg_spec
// reports. Used to size-check layout.json entries.
static size_t xdna_tblock_fused_w8a16_packed_bytes(
        int64_t K, int64_t N, int cols) {
    const int tile_k = XDNA_TBF_W8A16_TILE_K;
    const int tile_n = XDNA_TBF_W8A16_TILE_N;
    const int G      = XDNA_TBF_W8A16_GROUP_SIZE;
    const int64_t num_groups_per_tile = tile_k / G;
    const int64_t num_k_tiles         = K / tile_k;
    const int64_t n_per_col           = N / cols;
    const int64_t num_n_tiles_per_col = n_per_col / tile_n;
    const int64_t packed_tile_bytes =
        (int64_t)tile_k * (int64_t)tile_n +
        num_groups_per_tile * (int64_t)tile_n * 2;
    const int64_t bytes_per_col = num_n_tiles_per_col * num_k_tiles * packed_tile_bytes;
    return (size_t)((int64_t)cols * bytes_per_col);
}

// Infer the `cols` parameter from K, N, and the packed byte length stored
// in layout.json. Since tile_k/tile_n/G are fixed, cols is the only free
// knob on the layout.length side of the equation. Returns 0 on failure.
static int xdna_tblock_fused_w8a16_infer_cols(
        int64_t K, int64_t N, size_t layout_length) {
    const int candidates[] = { 8, 4, 2, 1 };
    for (int c : candidates) {
        if (N % c != 0) continue;
        if ((N / c) % XDNA_TBF_W8A16_TILE_N != 0) continue;
        if (K % XDNA_TBF_W8A16_TILE_K != 0) continue;
        const size_t need = xdna_tblock_fused_w8a16_packed_bytes(K, N, c);
        if (need == layout_length) return c;
    }
    return 0;
}

// Core packer. Input K×N fp32 weight (row-major, axis-0 = K, axis-1 = N).
// Writes the packed uint8 stream into `out` (exactly `packed_bytes` bytes).
// OpenMP-parallel across `col`.
static void xdna_tblock_fused_w8a16_pack_from_f32(
        const float * __restrict__ B_kn,
        int64_t K, int64_t N, int cols,
        uint8_t * __restrict__ out) {
    const int tile_k = XDNA_TBF_W8A16_TILE_K;
    const int tile_n = XDNA_TBF_W8A16_TILE_N;
    const int G      = XDNA_TBF_W8A16_GROUP_SIZE;
    const int64_t num_groups_per_col  = K / G;
    const int64_t num_groups_per_tile = tile_k / G;
    const int64_t num_k_tiles         = K / tile_k;
    const int64_t n_per_col           = N / cols;
    const int64_t num_n_tiles_per_col = n_per_col / tile_n;

    const int64_t packed_tile_bytes =
        (int64_t)tile_k * (int64_t)tile_n +
        num_groups_per_tile * (int64_t)tile_n * 2;
    const int64_t bytes_per_col = num_n_tiles_per_col * num_k_tiles * packed_tile_bytes;

    // Per-column scales buffer reused across n_tiles (G_count × N_col).
    // Serial loop: bulk_prewarm already parallelises per-weight across its
    // std::async worker pool, so each packer invocation running single-
    // threaded is fine. Adding OpenMP here would double-dip on CPU cores
    // and worsen cache thrash during the init prewarm.
    for (int col = 0; col < cols; col++) {
        const int64_t n_col_start = (int64_t)col * n_per_col;
        uint8_t * col_out = out + (int64_t)col * bytes_per_col;

        // Per-group scales for this column. Two parallel arrays:
        //   scales_f32 — used directly for quantization (matches the
        //                reference which uses torch.float32 scales).
        //   scales_u16 — bf16 raw bits, written into the packed tile bytes
        //                exactly as the device MAC will read them.
        // The reference does INT8 quant with fp32 scales AND dequant with
        // fp32 scales (the bf16 conversion happens only on the packed-byte
        // copy path + on-device dequant). We mirror that.
        std::vector<float>    scales_f32((size_t)num_groups_per_col * (size_t)n_per_col);
        std::vector<uint16_t> scales_u16((size_t)num_groups_per_col * (size_t)n_per_col);

        for (int64_t g = 0; g < num_groups_per_col; g++) {
            const int64_t k_base = g * G;
            for (int64_t nn_local = 0; nn_local < n_per_col; nn_local++) {
                const int64_t nn = n_col_start + nn_local;
                float amax = 0.0f;
                for (int gi = 0; gi < G; gi++) {
                    float v = B_kn[(k_base + gi) * N + nn];
                    float a = v < 0.0f ? -v : v;
                    if (a > amax) amax = a;
                }
                float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
                const size_t idx =
                    (size_t)g * (size_t)n_per_col + (size_t)nn_local;
                scales_f32[idx] = scale;
                scales_u16[idx] = xdna_f32_to_bf16_bits(scale);
            }
        }

        int64_t byte_offset = 0;
        for (int64_t n_tile_idx = 0; n_tile_idx < num_n_tiles_per_col; n_tile_idx++) {
            const int64_t n_start_local = n_tile_idx * tile_n;
            const int64_t n_start = n_col_start + n_start_local;

            for (int64_t k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
                const int64_t k_start = k_tile_idx * tile_k;

                // 1) INT8 weights block, micro-tile (i_blk, j_blk, s, t)
                // layout to match the new fdgemm_i8 kernel (Phase A.7).
                // For (r,s,t)=(4,8,8): colA=tile_k/8, colB=tile_n/8.  Per
                // micro-tile: 64 contiguous int8 bytes in (s, t) row-major.
                int8_t * tile_int8 = (int8_t *)(col_out + byte_offset);
                const int s_dim = 8;
                const int t_dim = 8;
                const int64_t colB_local = tile_n / t_dim;
                for (int ki = 0; ki < tile_k; ki++) {
                    const int64_t k = k_start + ki;
                    const int64_t g = k / G;
                    const int64_t i_blk = ki / s_dim;
                    const int64_t s_in  = ki % s_dim;
                    for (int ni = 0; ni < tile_n; ni++) {
                        const int64_t nn = n_start + ni;
                        const int64_t j_blk = ni / t_dim;
                        const int64_t t_in  = ni % t_dim;
                        float v = B_kn[k * N + nn];
                        float scale = scales_f32[(size_t)g * (size_t)n_per_col +
                                                 (size_t)(nn - n_col_start)];
                        int32_t qi = (int32_t)rintf(v / scale);
                        if (qi < -128) qi = -128;
                        else if (qi > 127) qi = 127;
                        const int64_t mt_off =
                            (i_blk * colB_local + j_blk) * (s_dim * t_dim) +
                            s_in * t_dim + t_in;
                        tile_int8[mt_off] = (int8_t)qi;
                    }
                }
                byte_offset += (int64_t)tile_k * (int64_t)tile_n;

                // 2) bf16 scales block, shape (num_groups_per_tile, tile_n).
                uint16_t * tile_scales = (uint16_t *)(col_out + byte_offset);
                const int64_t g_start = k_start / G;
                for (int64_t gi = 0; gi < num_groups_per_tile; gi++) {
                    for (int ni = 0; ni < tile_n; ni++) {
                        const int64_t nn = n_start + ni;
                        tile_scales[gi * tile_n + ni] =
                            scales_u16[(size_t)(g_start + gi) * (size_t)n_per_col +
                                       (size_t)(nn - n_col_start)];
                    }
                }
                byte_offset += num_groups_per_tile * (int64_t)tile_n * 2;
            }
        }
        (void)byte_offset;  // matches bytes_per_col by construction
    }
}

// Write a W8A16 attention projection weight into entry->input_bo. Accepts
// F16, F32, BF16, Q8_0 ggml tensors. All dtypes go through a (K, N) fp32
// staging buffer + the generic packer. A direct Q8_0 → tile-layout
// repack was attempted but had a correctness bug in the sub-tile byte
// layout — reverted. Future optimization: rewrite with byte-level unit
// test against the fp32-packer output.
//
// ggml stores B as (N, K) row-major (B_ggml[n, k] at offset n*K + k).
// The packer expects (K, N) row-major: B_kn[k, n] = B_ggml[n, k].
static bool tblock_fused_upload_w8a16_weight(
        xdna_tblock_fused_entry * entry,
        const char * name,
        const struct ggml_tensor * weight) {
    auto it = entry->layout.buffers.find(name);
    if (it == entry->layout.buffers.end()) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 weight '%s' missing from layout\n", name);
        return false;
    }
    if (it->second.buf_type != "input") {
        GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 weight '%s' in unexpected buf_type=%s\n",
                       name, it->second.buf_type.c_str());
        return false;
    }
    if (it->second.dtype != "uint8") {
        GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 weight '%s' dtype=%s (expected uint8)\n",
                       name, it->second.dtype.c_str());
        return false;
    }

    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];

    const int cols = xdna_tblock_fused_w8a16_infer_cols(K, N, it->second.length);
    if (cols == 0) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 weight '%s' K=%lld N=%lld: can't infer cols "
                       "from layout length %zu\n", name, (long long)K, (long long)N, it->second.length);
        return false;
    }

    // Staging fp32 (K, N).
    std::vector<float> B_kn((size_t)K * (size_t)N);
    if (weight->type == GGML_TYPE_F32) {
        const float * src = (const float *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                B_kn[(size_t)k * (size_t)N + (size_t)nn] = src[nn * K + k];
            }
        }
    } else if (weight->type == GGML_TYPE_F16) {
        const uint16_t * src = (const uint16_t *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                ggml_fp16_t h;
                memcpy(&h, &src[nn * K + k], sizeof(h));
                B_kn[(size_t)k * (size_t)N + (size_t)nn] = ggml_fp16_to_fp32(h);
            }
        }
    } else if (weight->type == GGML_TYPE_BF16) {
        const uint16_t * src = (const uint16_t *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                uint32_t bits = ((uint32_t)src[nn * K + k]) << 16;
                float v;
                memcpy(&v, &bits, sizeof(v));
                B_kn[(size_t)k * (size_t)N + (size_t)nn] = v;
            }
        }
    } else if (weight->type == GGML_TYPE_Q8_0) {
        // Q8_0 block-32: per (row, k-block) = [fp16 scale, 32 int8].
        // Dequant to fp32 then repack. Our packer re-derives scales from
        // the dequanted fp32 values; because Q8_0 uses the same symmetric
        // `max(|vals|)/127` scheme, this round-trip is numerically
        // identical to Q8_0's original int8 bytes (up to fp32 rounding).
        // A direct Q8_0 → tile-layout path was attempted but had a
        // correctness bug in the sub-tile byte layout — reverted until
        // someone rewrites with a byte-level unit test against the
        // fp32-packer output.
        constexpr int QK8_0 = 32;
        const size_t blk_bytes = sizeof(ggml_fp16_t) + (size_t)QK8_0;
        if (K % QK8_0 != 0) {
            GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 Q8_0 weight '%s' K=%lld "
                           "not multiple of QK8_0=32\n", name, (long long)K);
            return false;
        }
        const int64_t num_blks_per_row = K / QK8_0;
        const size_t row_stride = (size_t)num_blks_per_row * blk_bytes;
        const uint8_t * src = (const uint8_t *)weight->data;
        for (int64_t nn = 0; nn < N; nn++) {
            const uint8_t * row = src + (size_t)nn * row_stride;
            for (int64_t g = 0; g < num_blks_per_row; g++) {
                const uint8_t * blk = row + (size_t)g * blk_bytes;
                ggml_fp16_t dh;
                memcpy(&dh, blk, sizeof(dh));
                const float d = ggml_fp16_to_fp32(dh);
                const int8_t * qs = (const int8_t *)(blk + sizeof(ggml_fp16_t));
                for (int j = 0; j < QK8_0; j++) {
                    const int64_t k = g * QK8_0 + j;
                    B_kn[(size_t)k * (size_t)N + (size_t)nn] = d * (float)qs[j];
                }
            }
        }
    } else {
        GGML_LOG_ERROR("ggml-xdna: fused tblock W8A16 weight '%s' unsupported dtype %d\n",
                       name, (int)weight->type);
        return false;
    }

    uint8_t * dst = (uint8_t *)((char *)entry->input_bo->map<void*>() + it->second.offset);
    xdna_tblock_fused_w8a16_pack_from_f32(B_kn.data(), K, N, cols, dst);
    return true;
}

// Write the weight tensor (ggml-native [N, K] layout in src->data) into
// entry->input_bo at layout.buffers[name].offset as bf16 [K, N] row-major.
// That is what FusedMLIROperator's internal GEMM sub-ops consume directly
// (IRON GEMM default b_col_maj=False).
//
// W8A16 dispatch: if layout.json says the slot dtype is uint8, route
// through the INT8+scales packer instead of the bf16 transpose below.
// Only attention projection slots (w_q/w_k/w_v/w_o) carry uint8 in
// Phase A; FFN weights stay bf16.
static bool tblock_fused_upload_gemm_weight(
        xdna_tblock_fused_entry * entry,
        const char * name,
        const struct ggml_tensor * weight) {
    auto it = entry->layout.buffers.find(name);
    if (it == entry->layout.buffers.end()) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock weight '%s' missing from layout\n", name);
        return false;
    }
    if (it->second.buf_type != "input") {
        GGML_LOG_ERROR("ggml-xdna: fused tblock weight '%s' in unexpected buf_type=%s\n",
                       name, it->second.buf_type.c_str());
        return false;
    }
    if (it->second.dtype == "uint8") {
        return tblock_fused_upload_w8a16_weight(entry, name, weight);
    }
    const int64_t K = weight->ne[0];
    const int64_t N = weight->ne[1];
    const size_t  n_elems = (size_t)K * (size_t)N;
    const size_t  need_bytes = n_elems * sizeof(uint16_t);
    if (it->second.length < need_bytes) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock weight '%s' layout length %zu < needed %zu\n",
                       name, it->second.length, need_bytes);
        return false;
    }
    uint16_t * dst = (uint16_t *)((char *)entry->input_bo->map<void*>() + it->second.offset);
    if (weight->type == GGML_TYPE_F32) {
        const float * src = (const float *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                float v = src[nn * K + k];
                uint32_t bits;
                memcpy(&bits, &v, sizeof(bits));
                bits += (0x7FFF + ((bits >> 16) & 1));
                dst[k * N + nn] = (uint16_t)(bits >> 16);
            }
        }
    } else if (weight->type == GGML_TYPE_F16) {
        const uint16_t * src = (const uint16_t *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                dst[k * N + nn] = fp16_to_bf16(src[nn * K + k]);
            }
        }
    } else if (weight->type == GGML_TYPE_BF16) {
        const uint16_t * src = (const uint16_t *)weight->data;
        for (int64_t k = 0; k < K; k++) {
            for (int64_t nn = 0; nn < N; nn++) {
                dst[k * N + nn] = src[nn * K + k];
            }
        }
    } else if (weight->type == GGML_TYPE_Q8_0) {
        // Q8_0 → bf16 dequant + transpose (ggml (N, K) → packer (K, N)).
        constexpr int QK8_0 = 32;
        const size_t blk_bytes = sizeof(ggml_fp16_t) + (size_t)QK8_0;
        if (K % QK8_0 != 0) {
            GGML_LOG_ERROR("ggml-xdna: fused tblock bf16 weight '%s' K=%lld not "
                           "multiple of QK8_0=32\n", name, (long long)K);
            return false;
        }
        const int64_t num_blks_per_row = K / QK8_0;
        const size_t row_stride = (size_t)num_blks_per_row * blk_bytes;
        const uint8_t * src = (const uint8_t *)weight->data;
        for (int64_t nn = 0; nn < N; nn++) {
            const uint8_t * row = src + (size_t)nn * row_stride;
            for (int64_t g = 0; g < num_blks_per_row; g++) {
                const uint8_t * blk = row + (size_t)g * blk_bytes;
                ggml_fp16_t dh;
                memcpy(&dh, blk, sizeof(dh));
                const float d = ggml_fp16_to_fp32(dh);
                const int8_t * qs = (const int8_t *)(blk + sizeof(ggml_fp16_t));
                for (int j = 0; j < QK8_0; j++) {
                    const int64_t k = g * QK8_0 + j;
                    const float v = d * (float)qs[j];
                    dst[k * N + nn] = xdna_f32_to_bf16_bits(v);
                }
            }
        }
    } else {
        GGML_LOG_ERROR("ggml-xdna: fused tblock weight '%s' unsupported dtype %d\n",
                       name, (int)weight->type);
        return false;
    }
    return true;
}

// Write an RMSNorm gain vector (1-D, no transpose) to input_bo at layout offset.
static bool tblock_fused_upload_gain(
        xdna_tblock_fused_entry * entry,
        const char * name,
        const struct ggml_tensor * gain) {
    auto it = entry->layout.buffers.find(name);
    if (it == entry->layout.buffers.end()) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock gain '%s' missing from layout\n", name);
        return false;
    }
    const int64_t E = gain->ne[0];
    const size_t need_bytes = (size_t)E * sizeof(uint16_t);
    if (it->second.length < need_bytes) return false;
    uint16_t * dst = (uint16_t *)((char *)entry->input_bo->map<void*>() + it->second.offset);
    if (gain->type == GGML_TYPE_F32) {
        f32_to_bf16((const float *)gain->data, dst, (size_t)E);
    } else if (gain->type == GGML_TYPE_F16) {
        const uint16_t * src = (const uint16_t *)gain->data;
        for (int64_t i = 0; i < E; i++) dst[i] = fp16_to_bf16(src[i]);
    } else if (gain->type == GGML_TYPE_BF16) {
        memcpy(dst, gain->data, need_bytes);
    } else {
        return false;
    }
    return true;
}

// Bulk fused-tblock weight pre-warm. Mirrors tblock_prefill_bulk_prewarm but
// writes into each layer-entry's single input_bo at the layout offsets
// documented in entry->layout.buffers (as the per-dispatch upload path does).
// The expensive step is the per-weight bf16 transpose — parallelizing it
// across CPU cores converts first-batch ~27 t/s into steady-state throughput.
//
// Guarded per-cgraph pointer. The dispatch path's loaded_srcs check still
// covers correctness if this ever runs twice on overlapping matches.
// ----------------------------------------------------------------------------
static void tblock_fused_bulk_prewarm(ggml_backend_xdna_context * ctx,
                                      const struct ggml_cgraph * cgraph) {
    {
        std::lock_guard<std::mutex> lock(ctx->cache_mutex);
        if (ctx->tblock_fused_prewarmed_cgraphs.count(cgraph)) return;
    }

    std::vector<xdna_transformer_block_match> matches;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_RMS_NORM) continue;
        xdna_transformer_block_match tm{};
        if (xdna_try_match_transformer_block_prefill(cgraph, i, &tm) &&
            tm.attn.seq_len >= 256) {
            matches.push_back(tm);
        }
    }
    if (matches.empty()) return;

    // Layer 4B multi-layer packing (opt-in). When XDNA_ENABLE_TBLOCK_FUSED_N
    // is set to 2 or 4, walk `matches` and greedily group N-consecutive
    // blocks that share shape/rope/ffn_hidden. Each group gets ONE entry
    // loaded from the N-block ELF bundle with per-layer suffixed weight
    // slots (w_q_0..w_q_{N-1}, ...). Tail blocks that don't fit a full
    // N-group fall back to classic N=1 entries.
    const int desired_N = xdna_tblock_fused_group_N();
    auto tms_compat = [](const xdna_transformer_block_match & a,
                         const xdna_transformer_block_match & b) -> bool {
        const auto & x = a.attn;
        const auto & y = b.attn;
        return x.seq_len        == y.seq_len
            && x.embed_dim      == y.embed_dim
            && x.num_heads      == y.num_heads
            && x.num_kv_heads   == y.num_kv_heads
            && x.head_dim       == y.head_dim
            && x.rope_method_type == y.rope_method_type
            && a.ffn_hidden_dim == b.ffn_hidden_dim;
    };
    std::vector<std::vector<xdna_transformer_block_match>> groups;
    std::vector<xdna_transformer_block_match> singles;
    if (desired_N <= 1) {
        for (const auto & tm : matches) singles.push_back(tm);
    } else {
        size_t idx = 0;
        while (idx < matches.size()) {
            if (idx + (size_t)desired_N <= matches.size()) {
                bool compat = true;
                for (int k = 1; k < desired_N; k++) {
                    if (!tms_compat(matches[idx], matches[idx + k])) {
                        compat = false;
                        break;
                    }
                }
                if (compat) {
                    std::vector<xdna_transformer_block_match> g;
                    for (int k = 0; k < desired_N; k++) {
                        g.push_back(matches[idx + k]);
                    }
                    groups.push_back(std::move(g));
                    idx += desired_N;
                    continue;
                }
            }
            // Not enough remaining or shape-incompatible -> tail single.
            singles.push_back(matches[idx]);
            idx++;
        }
    }

    struct weight_task {
        xdna_tblock_fused_entry * entry;
        std::string slot_name;  // owned (may be suffixed for N>1)
        const struct ggml_tensor * weight;
        bool is_gain;
    };
    std::vector<weight_task> tasks;
    std::unordered_set<xdna_tblock_fused_entry *> entries_seen;

    // Common weight slot ordering (shared between single-block and N-block
    // paths). For N>1 each entry gets N copies of this list with per-layer
    // suffixes.
    struct slot_desc {
        const char * base_name;
        bool is_gain;
    };
    const slot_desc slot_bases[] = {
        { XDNA_TBF_W_NORM_ATTN, true  },
        { XDNA_TBF_W_Q,         false },
        { XDNA_TBF_W_K,         false },
        { XDNA_TBF_W_V,         false },
        { XDNA_TBF_W_O,         false },
        { XDNA_TBF_W_NORM_FFN,  true  },
        { XDNA_TBF_W_GATE,      false },
        { XDNA_TBF_W_UP,        false },
        { XDNA_TBF_W_DOWN,      false },
    };
    auto match_weight_for = [](const xdna_transformer_block_match & tm,
                               const char * base) -> const struct ggml_tensor * {
        const auto & m = tm.attn;
        if (strcmp(base, XDNA_TBF_W_NORM_ATTN) == 0) return m.gain;
        if (strcmp(base, XDNA_TBF_W_Q)         == 0) return m.wq;
        if (strcmp(base, XDNA_TBF_W_K)         == 0) return m.wk;
        if (strcmp(base, XDNA_TBF_W_V)         == 0) return m.wv;
        if (strcmp(base, XDNA_TBF_W_O)         == 0) return m.wo;
        if (strcmp(base, XDNA_TBF_W_NORM_FFN)  == 0) return tm.ffn_gain;
        if (strcmp(base, XDNA_TBF_W_GATE)      == 0) return tm.w_gate;
        if (strcmp(base, XDNA_TBF_W_UP)        == 0) return tm.w_up;
        if (strcmp(base, XDNA_TBF_W_DOWN)      == 0) return tm.w_down;
        return nullptr;
    };

    // Groups first: one N-block entry per group, weights suffixed per-layer.
    std::vector<xdna_tblock_fused_group> built_groups;
    for (const auto & g : groups) {
        const auto & head = g.front();
        const auto & m = head.attn;
        const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
        if (seq_bucket < 0) continue;
        if (m.head_dim != 64) continue;
        if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) continue;
        if ((m.num_kv_heads * m.head_dim) % 64 != 0) continue;
        if (m.num_heads % m.num_kv_heads != 0) continue;
        if (head.ffn_hidden_dim <= 0 || head.ffn_hidden_dim % 64 != 0) continue;

        const int group_N = (int)g.size();
        const bool w8a16_gated     = xdna_tblock_fused_w8a16_enabled();
        const bool w8a16_ffn_gated = xdna_tblock_fused_w8a16_ffn_enabled();
        const std::string shape_key = make_tblock_fused_cache_key(
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
            head.ffn_hidden_dim, m.rope_method_type, group_N, w8a16_gated,
            w8a16_ffn_gated);
        // Layer key uses the FIRST block's wq ptr so distinct groups get
        // distinct in-memory entries (separate BOs) even when sharing the
        // same on-disk N-block bundle.
        const std::string layer_key = make_tblock_fused_layer_cache_key(
            shape_key, m.wq->data);

        if (!ensure_tblock_fused_compiled(
                ctx, shape_key, seq_bucket, m.embed_dim, m.num_heads,
                m.num_kv_heads, m.head_dim, head.ffn_hidden_dim,
                m.rope_method_type, group_N, w8a16_gated, w8a16_ffn_gated)) {
            continue;
        }
        xdna_tblock_fused_entry * entry = get_or_load_tblock_fused_kernel(
            ctx, layer_key, shape_key,
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads,
            m.head_dim, head.ffn_hidden_dim, m.rope_method_type, group_N,
            w8a16_gated, w8a16_ffn_gated);
        if (!entry) continue;

        xdna_tblock_fused_group gd;
        gd.N           = group_N;
        gd.range_begin = g.front().attn.rms_norm_idx;
        gd.range_end   = g.back().ffn_residual_add_idx;
        gd.tms         = g;
        gd.entry       = entry;
        built_groups.push_back(std::move(gd));

        bool any_needed = false;
        for (int L = 0; L < group_N; L++) {
            for (const auto & sd : slot_bases) {
                const struct ggml_tensor * w = match_weight_for(g[L], sd.base_name);
                if (!w || !w->data) continue;
                std::string sname = tblock_fused_slot_name(
                    sd.base_name, L, group_N);
                {
                    std::lock_guard<std::mutex> lock(*entry->mu);
                    auto it = entry->loaded_srcs.find(sname);
                    if (it != entry->loaded_srcs.end() && it->second == w->data) continue;
                }
                weight_task wt;
                wt.entry     = entry;
                wt.slot_name = std::move(sname);
                wt.weight    = w;
                wt.is_gain   = sd.is_gain;
                tasks.push_back(std::move(wt));
                any_needed = true;
            }
        }
        if (any_needed) entries_seen.insert(entry);
    }

    // Singles: one entry per tail block, legacy unsuffixed weight slots.
    for (const auto & tm : singles) {
        const auto & m = tm.attn;
        const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
        if (seq_bucket < 0) continue;
        if (m.head_dim != 64) continue;
        if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) continue;
        if ((m.num_kv_heads * m.head_dim) % 64 != 0) continue;
        if (m.num_heads % m.num_kv_heads != 0) continue;
        if (tm.ffn_hidden_dim <= 0 || tm.ffn_hidden_dim % 64 != 0) continue;

        const bool w8a16_gated     = xdna_tblock_fused_w8a16_enabled();
        const bool w8a16_ffn_gated = xdna_tblock_fused_w8a16_ffn_enabled();
        const std::string shape_key = make_tblock_fused_cache_key(
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
            tm.ffn_hidden_dim, m.rope_method_type, /*num_layers=*/1,
            w8a16_gated, w8a16_ffn_gated);
        const std::string layer_key = make_tblock_fused_layer_cache_key(
            shape_key, m.wq->data);

        if (!ensure_tblock_fused_compiled(
                ctx, shape_key, seq_bucket, m.embed_dim, m.num_heads,
                m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim, m.rope_method_type,
                /*num_layers=*/1, w8a16_gated, w8a16_ffn_gated)) {
            continue;
        }
        xdna_tblock_fused_entry * entry = get_or_load_tblock_fused_kernel(
            ctx, layer_key, shape_key,
            seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads,
            m.head_dim, tm.ffn_hidden_dim, m.rope_method_type,
            /*num_layers=*/1, w8a16_gated, w8a16_ffn_gated);
        if (!entry) continue;

        bool any_needed = false;
        for (const auto & sd : slot_bases) {
            const struct ggml_tensor * w = match_weight_for(tm, sd.base_name);
            if (!w || !w->data) continue;
            std::string sname(sd.base_name);
            {
                std::lock_guard<std::mutex> lock(*entry->mu);
                auto it = entry->loaded_srcs.find(sname);
                if (it != entry->loaded_srcs.end() && it->second == w->data) continue;
            }
            weight_task wt;
            wt.entry     = entry;
            wt.slot_name = std::move(sname);
            wt.weight    = w;
            wt.is_gain   = sd.is_gain;
            tasks.push_back(std::move(wt));
            any_needed = true;
        }
        if (any_needed) entries_seen.insert(entry);
    }

    // Register any N-block groups on the context map — dispatch reads this
    // to redirect grouped block-start indices to the N-block xrt::run path.
    if (!built_groups.empty()) {
        std::lock_guard<std::mutex> lock(ctx->cache_mutex);
        auto & gmap = ctx->tblock_fused_groups_per_cgraph[cgraph];
        for (const auto & gd : built_groups) {
            gmap[gd.range_begin] = gd;
        }
    }

    if (tasks.empty()) {
        std::lock_guard<std::mutex> lock(ctx->cache_mutex);
        ctx->tblock_fused_prewarmed_cgraphs.insert(cgraph);
        return;
    }

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned nworkers = (unsigned)std::min<size_t>(hw, tasks.size());

    std::atomic<size_t> next{0};
    std::vector<std::future<void>> futs;
    futs.reserve(nworkers);
    for (unsigned w = 0; w < nworkers; w++) {
        futs.push_back(std::async(std::launch::async, [&tasks, &next]() {
            for (;;) {
                size_t idx = next.fetch_add(1, std::memory_order_relaxed);
                if (idx >= tasks.size()) break;
                const auto & t = tasks[idx];
                bool ok = t.is_gain
                    ? tblock_fused_upload_gain(t.entry, t.slot_name.c_str(), t.weight)
                    : tblock_fused_upload_gemm_weight(t.entry, t.slot_name.c_str(), t.weight);
                if (ok) {
                    std::lock_guard<std::mutex> lock(*t.entry->mu);
                    t.entry->loaded_srcs[t.slot_name] = t.weight->data;
                } else {
                    GGML_LOG_ERROR("ggml-xdna: fused tblock prewarm upload '%s' failed\n",
                                   t.slot_name.c_str());
                }
            }
        }));
    }
    for (auto & f : futs) f.wait();

    // One sync per unique entry pushes all static weights on-device once.
    // With the split-dynamic layout, dispatch re-syncs ONLY the small
    // dynamic_input_bo (x + angles, ~3 MB) per call instead of the full
    // 122 MB static input_bo. Legacy 3-BO bundles (dynamic_input_bo=null)
    // still pay the whole-BO re-sync cost per dispatch.
    for (auto * entry : entries_seen) {
        try {
            entry->input_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("ggml-xdna: fused tblock prewarm sync failed: %s\n", e.what());
        }
    }

    {
        std::lock_guard<std::mutex> lock(ctx->cache_mutex);
        ctx->tblock_fused_prewarmed_cgraphs.insert(cgraph);
    }

    static const bool tpf_dbg = getenv("XDNA_DEBUG") != NULL;
    if (tpf_dbg) {
        fprintf(stderr,
                "ggml-xdna: fused tblock bulk pre-warm done: %zu matches, "
                "%zu uploads, %zu entries, %u workers\n",
                matches.size(), tasks.size(), entries_seen.size(), nworkers);
        fflush(stderr);
    }
}

// Single dispatch of the fused transformer block for one layer. Analog of
// ggml_backend_xdna_transformer_block_prefill but replaces the 20-kernel
// runlist with a single xrt::run on a 3-BO (input/output/scratch) kernel
// loaded from a full ELF.
static bool ggml_backend_xdna_transformer_block_prefill_fused(
        ggml_backend_xdna_context * ctx,
        const xdna_transformer_block_match & tm,
        struct ggml_cgraph * cgraph) {
    if (!ctx->device_valid) return false;
    const auto & m = tm.attn;
    if (m.head_dim != 64) return false;
    if (m.embed_dim <= 0 || m.embed_dim % 8 != 0) return false;
    if ((m.num_kv_heads * m.head_dim) % 64 != 0) return false;
    if (m.num_heads % m.num_kv_heads != 0) return false;
    if (tm.ffn_hidden_dim <= 0 || tm.ffn_hidden_dim % 64 != 0) return false;

    const int64_t seq_bucket = xdna_select_attention_prefill_bucket(m.seq_len);
    if (seq_bucket < 0) return false;

    const bool w8a16_gated     = xdna_tblock_fused_w8a16_enabled();
    const bool w8a16_ffn_gated = xdna_tblock_fused_w8a16_ffn_enabled();
    const std::string shape_key = make_tblock_fused_cache_key(
        seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads, m.head_dim,
        tm.ffn_hidden_dim, m.rope_method_type, /*num_layers=*/1,
        w8a16_gated, w8a16_ffn_gated);
    const std::string layer_key = make_tblock_fused_layer_cache_key(
        shape_key, m.wq->data);

    if (!ensure_tblock_fused_compiled(
            ctx, shape_key, seq_bucket, m.embed_dim, m.num_heads,
            m.num_kv_heads, m.head_dim, tm.ffn_hidden_dim, m.rope_method_type,
            /*num_layers=*/1, w8a16_gated, w8a16_ffn_gated)) {
        return false;
    }

    xdna_tblock_fused_entry * entry = get_or_load_tblock_fused_kernel(
        ctx, layer_key, shape_key,
        seq_bucket, m.embed_dim, m.num_heads, m.num_kv_heads,
        m.head_dim, tm.ffn_hidden_dim, m.rope_method_type,
        /*num_layers=*/1, w8a16_gated, w8a16_ffn_gated);
    if (!entry) return false;

    const int64_t S     = m.seq_len;
    const int64_t S_pad = seq_bucket;
    const int64_t E     = m.embed_dim;
    const int64_t d     = m.head_dim;
    const size_t  bytes_x_actual = (size_t)S * (size_t)E * sizeof(uint16_t);
    const size_t  bytes_x_pad    = (size_t)S_pad * (size_t)E * sizeof(uint16_t);

    const bool _ap_prof = xdna_attn_prof_enabled();
    using _ap_clock = std::chrono::steady_clock;
    auto _ap_now = []() { return _ap_clock::now(); };
    auto _ap_us = [](_ap_clock::time_point a, _ap_clock::time_point b) {
        return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const auto _ap_t_disp_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};

    try {
        // Upload weights + gains once per layer-entry.
        {
            std::lock_guard<std::mutex> lock(*entry->mu);
            const auto _ap_t_ww_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};

            auto needs_upload = [&](const char * name, const void * src_ptr) {
                auto it = entry->loaded_srcs.find(name);
                return it == entry->loaded_srcs.end() || it->second != src_ptr;
            };
            auto mark_uploaded = [&](const char * name, const void * src_ptr) {
                entry->loaded_srcs[name] = src_ptr;
            };

            struct gemm_w { const char * name; const struct ggml_tensor * t; };
            const gemm_w gemm_weights[] = {
                { XDNA_TBF_W_Q,    m.wq       },
                { XDNA_TBF_W_K,    m.wk       },
                { XDNA_TBF_W_V,    m.wv       },
                { XDNA_TBF_W_O,    m.wo       },
                { XDNA_TBF_W_GATE, tm.w_gate  },
                { XDNA_TBF_W_UP,   tm.w_up    },
                { XDNA_TBF_W_DOWN, tm.w_down  },
            };
            for (const auto & gw : gemm_weights) {
                if (!needs_upload(gw.name, gw.t->data)) continue;
                if (!tblock_fused_upload_gemm_weight(entry, gw.name, gw.t)) return false;
                mark_uploaded(gw.name, gw.t->data);
            }

            struct gain_w { const char * name; const struct ggml_tensor * t; };
            const gain_w gain_weights[] = {
                { XDNA_TBF_W_NORM_ATTN, m.gain     },
                { XDNA_TBF_W_NORM_FFN,  tm.ffn_gain },
            };
            for (const auto & g : gain_weights) {
                if (!needs_upload(g.name, g.t->data)) continue;
                if (!tblock_fused_upload_gain(entry, g.name, g.t)) return false;
                mark_uploaded(g.name, g.t->data);
            }

            if (_ap_prof) {
                xdna_attn_prof_record(XDNA_AP_WEIGHT_WARM,
                                      _ap_us(_ap_t_ww_begin, _ap_now()));
            }
        }

        // Where x and angles live depends on the kernel layout:
        //   - Legacy 3-BO kernel: both live in input_bo (buf_type="input"),
        //     and we must re-sync the full 122 MB input_bo each call.
        //   - Split 4-BO kernel (bundles built 2026-04-20+): x/angles live
        //     in a separate dynamic_input_bo (buf_type="dynamic_input"),
        //     and only that small (~3 MB) BO needs to be synced per call.
        const bool split_dyn = (entry->dynamic_input_bo != nullptr);
        xrt::bo * xa_bo = split_dyn ? entry->dynamic_input_bo.get()
                                    : entry->input_bo.get();
        const char * xa_type = split_dyn ? "dynamic_input" : "input";

        // RoPE angles — precomputed per-call (cheap, ~us-scale).
        const auto _ap_t_rope_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            auto it = entry->layout.buffers.find(XDNA_TBF_ANGLES);
            if (it == entry->layout.buffers.end() || it->second.buf_type != xa_type) {
                GGML_LOG_ERROR("ggml-xdna: fused tblock 'angles' missing from layout or wrong buf_type\n");
                return false;
            }
            const size_t need = (size_t)S_pad * (size_t)d * sizeof(uint16_t);
            if (it->second.length < need) return false;
            uint16_t * dst = (uint16_t *)((char *)xa_bo->map<void*>()
                                          + it->second.offset);
            // Zero the full padded region; RoPE helper writes only the first S rows.
            memset(dst, 0, it->second.length);
            if (!xdna_compute_rope_angles_bf16(m.q_rope_node, S, S_pad, d, dst)) {
                return false;
            }
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_ROPE_PRECOMPUTE,
                                  _ap_us(_ap_t_rope_begin, _ap_now()));
        }

        // Activation x — write into the dynamic BO (or input_bo in legacy
        // mode) at its offset.
        const auto _ap_t_in_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            auto it = entry->layout.buffers.find(XDNA_TBF_X);
            if (it == entry->layout.buffers.end() || it->second.buf_type != xa_type) {
                GGML_LOG_ERROR("ggml-xdna: fused tblock 'x' missing from layout or wrong buf_type\n");
                return false;
            }
            if (it->second.length < bytes_x_pad) return false;
            uint16_t * dst = (uint16_t *)((char *)xa_bo->map<void*>()
                                          + it->second.offset);
            if (S_pad > S) memset(dst, 0, bytes_x_pad);
            if (m.inpL->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)m.inpL->data, dst, (size_t)S * (size_t)E);
            } else if (m.inpL->type == GGML_TYPE_F16) {
                const uint16_t * src = (const uint16_t *)m.inpL->data;
                for (int64_t i = 0; i < S * E; i++) dst[i] = fp16_to_bf16(src[i]);
            } else {
                memcpy(dst, m.inpL->data, bytes_x_actual);
            }
            // Sync only the (possibly-small) x/angles BO. The large static
            // input_bo was synced once during bulk prewarm and never again.
            xa_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_TO_DEV,
                                  _ap_us(_ap_t_in_begin, _ap_now()));
        }

        // Single xrt::run on the fused kernel — one NPU submission for the
        // whole transformer block. This is the entire point of Layer 4A.
        // Kernel signature: (input, [dynamic_input,] output, scratch).
        const auto _ap_t_rlbuild_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        xrt::run run(entry->kernel);
        int argi = 0;
        run.set_arg(argi++, *entry->input_bo);
        if (split_dyn) run.set_arg(argi++, *entry->dynamic_input_bo);
        run.set_arg(argi++, *entry->output_bo);
        run.set_arg(argi++, *entry->scratch_bo);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_BUILD,
                                  _ap_us(_ap_t_rlbuild_begin, _ap_now()));
        }
        const auto _ap_t_exec_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        run.start();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_EXEC,
                                  _ap_us(_ap_t_exec_begin, _ap_now()));
        }
        const auto _ap_t_wait_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        ert_cmd_state st = run.wait();
        if (st != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("ggml-xdna: fused tblock xrt::run.wait() returned %d\n", (int)st);
            return false;
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_WAIT,
                                  _ap_us(_ap_t_wait_begin, _ap_now()));
        }

        // Pull output back and copy into the dst tensor.
        const auto _ap_t_out_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_FROM_DEV,
                                  _ap_us(_ap_t_out_begin, _ap_now()));
        }
        {
            auto it = entry->layout.buffers.find(XDNA_TBF_Y);
            if (it == entry->layout.buffers.end() || it->second.buf_type != "output") {
                GGML_LOG_ERROR("ggml-xdna: fused tblock 'y' missing from layout\n");
                return false;
            }
            const uint16_t * src = (const uint16_t *)((const char *)entry->output_bo->map<void*>()
                                                      + it->second.offset);
            struct ggml_tensor * dst = cgraph->nodes[tm.ffn_residual_add_idx];
            const size_t actual_elems = (size_t)S * (size_t)E;
            if (it->second.length < actual_elems * sizeof(uint16_t)) return false;
            if (dst->type == GGML_TYPE_F32) {
                bf16_to_f32(src, (float *)dst->data, actual_elems);
            } else if (dst->type == GGML_TYPE_BF16) {
                memcpy(dst->data, src, actual_elems * sizeof(uint16_t));
            } else {
                GGML_LOG_ERROR("ggml-xdna: fused tblock unsupported dst dtype %d\n",
                               (int)dst->type);
                return false;
            }
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_TOTAL_PER_LAYER,
                                  _ap_us(_ap_t_disp_begin, _ap_now()));
        }
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock dispatch failed: %s\n", e.what());
        return false;
    }
}

// ============================================================================
// Layer 4B — multi-layer-packed fused transformer-block dispatcher.
// Uses a pre-built xdna_tblock_fused_group holding N matches and the
// N-block ELF entry (weights already uploaded during bulk_prewarm at
// suffixed slots w_q_0 ... w_q_{N-1}). Writes layer 0's x into the
// dynamic_input BO, computes angles once, runs one xrt::run covering
// all N blocks, and writes the last block's output to its destination
// tensor.
// ============================================================================
static bool ggml_backend_xdna_tblock_fused_multi(
        ggml_backend_xdna_context * ctx,
        const xdna_tblock_fused_group & group,
        struct ggml_cgraph * cgraph) {
    if (!ctx->device_valid) return false;
    if (group.tms.empty() || group.entry == nullptr) return false;

    xdna_tblock_fused_entry * entry = group.entry;
    const auto & head = group.tms.front();
    const auto & m = head.attn;
    const int N = group.N;

    const int64_t S     = m.seq_len;
    const int64_t S_pad = entry->seq_len_padded;
    const int64_t E     = m.embed_dim;
    const int64_t d     = m.head_dim;
    const size_t  bytes_x_actual = (size_t)S * (size_t)E * sizeof(uint16_t);
    const size_t  bytes_x_pad    = (size_t)S_pad * (size_t)E * sizeof(uint16_t);

    const bool _ap_prof = xdna_attn_prof_enabled();
    using _ap_clock = std::chrono::steady_clock;
    auto _ap_now = []() { return _ap_clock::now(); };
    auto _ap_us = [](_ap_clock::time_point a, _ap_clock::time_point b) {
        return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    const auto _ap_t_disp_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};

    try {
        // Per-layer weight re-upload check (cheap needs_upload dedup). The
        // prewarm path already uploaded weights to suffixed slots, but the
        // per-dispatch check covers the case where a weight tensor pointer
        // changed between graph_compute calls (e.g. buffer reload).
        {
            std::lock_guard<std::mutex> lock(*entry->mu);
            const auto _ap_t_ww_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
            auto needs_upload = [&](const std::string & name, const void * src_ptr) {
                auto it = entry->loaded_srcs.find(name);
                return it == entry->loaded_srcs.end() || it->second != src_ptr;
            };
            auto mark_uploaded = [&](const std::string & name, const void * src_ptr) {
                entry->loaded_srcs[name] = src_ptr;
            };
            struct base_slot { const char * base; bool is_gain; };
            const base_slot bases[] = {
                { XDNA_TBF_W_NORM_ATTN, true  },
                { XDNA_TBF_W_Q,         false },
                { XDNA_TBF_W_K,         false },
                { XDNA_TBF_W_V,         false },
                { XDNA_TBF_W_O,         false },
                { XDNA_TBF_W_NORM_FFN,  true  },
                { XDNA_TBF_W_GATE,      false },
                { XDNA_TBF_W_UP,        false },
                { XDNA_TBF_W_DOWN,      false },
            };
            for (int L = 0; L < N; L++) {
                const auto & tmL = group.tms[L];
                for (const auto & b : bases) {
                    const struct ggml_tensor * w = nullptr;
                    if      (strcmp(b.base, XDNA_TBF_W_NORM_ATTN) == 0) w = tmL.attn.gain;
                    else if (strcmp(b.base, XDNA_TBF_W_Q)         == 0) w = tmL.attn.wq;
                    else if (strcmp(b.base, XDNA_TBF_W_K)         == 0) w = tmL.attn.wk;
                    else if (strcmp(b.base, XDNA_TBF_W_V)         == 0) w = tmL.attn.wv;
                    else if (strcmp(b.base, XDNA_TBF_W_O)         == 0) w = tmL.attn.wo;
                    else if (strcmp(b.base, XDNA_TBF_W_NORM_FFN)  == 0) w = tmL.ffn_gain;
                    else if (strcmp(b.base, XDNA_TBF_W_GATE)      == 0) w = tmL.w_gate;
                    else if (strcmp(b.base, XDNA_TBF_W_UP)        == 0) w = tmL.w_up;
                    else if (strcmp(b.base, XDNA_TBF_W_DOWN)      == 0) w = tmL.w_down;
                    if (!w || !w->data) continue;
                    std::string sname = tblock_fused_slot_name(b.base, L, N);
                    if (!needs_upload(sname, w->data)) continue;
                    bool ok = b.is_gain
                        ? tblock_fused_upload_gain(entry, sname.c_str(), w)
                        : tblock_fused_upload_gemm_weight(entry, sname.c_str(), w);
                    if (!ok) return false;
                    mark_uploaded(sname, w->data);
                }
            }
            if (_ap_prof) {
                xdna_attn_prof_record(XDNA_AP_WEIGHT_WARM,
                                      _ap_us(_ap_t_ww_begin, _ap_now()));
            }
        }

        const bool split_dyn = (entry->dynamic_input_bo != nullptr);
        xrt::bo * xa_bo = split_dyn ? entry->dynamic_input_bo.get()
                                    : entry->input_bo.get();
        const char * xa_type = split_dyn ? "dynamic_input" : "input";

        // RoPE angles: computed once from the head block's rope node. RoPE
        // is position-only; the same cos/sin table is shared by every
        // transformer layer in the model, so computing from layer 0 matches
        // what layer 1..N-1 would produce.
        const auto _ap_t_rope_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            auto it = entry->layout.buffers.find(XDNA_TBF_ANGLES);
            if (it == entry->layout.buffers.end() || it->second.buf_type != xa_type) {
                GGML_LOG_ERROR("ggml-xdna: fused tblock multi 'angles' missing from layout or wrong buf_type\n");
                return false;
            }
            const size_t need = (size_t)S_pad * (size_t)d * sizeof(uint16_t);
            if (it->second.length < need) return false;
            uint16_t * dst = (uint16_t *)((char *)xa_bo->map<void*>() + it->second.offset);
            memset(dst, 0, it->second.length);
            if (!xdna_compute_rope_angles_bf16(m.q_rope_node, S, S_pad, d, dst)) {
                return false;
            }
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_ROPE_PRECOMPUTE,
                                  _ap_us(_ap_t_rope_begin, _ap_now()));
        }

        // Activation x — first block's input (inpL). Subsequent blocks read
        // the residual chain on-device via scratch buffer `y_{L}`.
        const auto _ap_t_in_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        {
            auto it = entry->layout.buffers.find(XDNA_TBF_X);
            if (it == entry->layout.buffers.end() || it->second.buf_type != xa_type) {
                GGML_LOG_ERROR("ggml-xdna: fused tblock multi 'x' missing from layout or wrong buf_type\n");
                return false;
            }
            if (it->second.length < bytes_x_pad) return false;
            uint16_t * dst = (uint16_t *)((char *)xa_bo->map<void*>() + it->second.offset);
            if (S_pad > S) memset(dst, 0, bytes_x_pad);
            if (m.inpL->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)m.inpL->data, dst, (size_t)S * (size_t)E);
            } else if (m.inpL->type == GGML_TYPE_F16) {
                const uint16_t * src = (const uint16_t *)m.inpL->data;
                for (int64_t i = 0; i < S * E; i++) dst[i] = fp16_to_bf16(src[i]);
            } else {
                memcpy(dst, m.inpL->data, bytes_x_actual);
            }
            xa_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_TO_DEV,
                                  _ap_us(_ap_t_in_begin, _ap_now()));
        }

        const auto _ap_t_rlbuild_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        xrt::run run(entry->kernel);
        int argi = 0;
        run.set_arg(argi++, *entry->input_bo);
        if (split_dyn) run.set_arg(argi++, *entry->dynamic_input_bo);
        run.set_arg(argi++, *entry->output_bo);
        run.set_arg(argi++, *entry->scratch_bo);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_BUILD,
                                  _ap_us(_ap_t_rlbuild_begin, _ap_now()));
        }
        const auto _ap_t_exec_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        run.start();
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_EXEC,
                                  _ap_us(_ap_t_exec_begin, _ap_now()));
        }
        const auto _ap_t_wait_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        ert_cmd_state st = run.wait();
        if (st != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("ggml-xdna: fused tblock multi xrt::run.wait() returned %d\n", (int)st);
            return false;
        }
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_RL_WAIT,
                                  _ap_us(_ap_t_wait_begin, _ap_now()));
        }

        // Sync output BO from device. Actual copy-back of y_L values to
        // ggml tensors is interleaved with PhaseC_post by the caller to
        // avoid ggml-alloc aliasing hazards: if the allocator decided
        // tms[L].ffn_residual_add and tms[M].ffn_residual_add share
        // storage (possible when their lifetimes don't overlap in the
        // original graph), writing both back-to-back in the dispatcher
        // would clobber. Caller does copy(L) → phase_c(L+1) pairs.
        const auto _ap_t_out_begin = _ap_prof ? _ap_now() : _ap_clock::time_point{};
        entry->output_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_BO_SYNC_FROM_DEV,
                                  _ap_us(_ap_t_out_begin, _ap_now()));
        }

        if (_ap_prof) {
            xdna_attn_prof_record(XDNA_AP_TOTAL_PER_LAYER,
                                  _ap_us(_ap_t_disp_begin, _ap_now()));
        }
        return true;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock multi dispatch failed: %s\n", e.what());
        return false;
    }
}

// Copy one block's residual output from the multi entry's output_bo to
// the corresponding ggml tensor. Called interleaved with PhaseC_post by
// the main dispatch branch so each y_L is consumed before the next
// (potentially aliasing) y_{L+1} is written.
static bool tblock_fused_multi_copy_y(
        const xdna_tblock_fused_group & group,
        int L, struct ggml_cgraph * cgraph) {
    if (L < 0 || L >= group.N) return false;
    xdna_tblock_fused_entry * entry = group.entry;
    const int64_t S = group.tms[L].attn.seq_len;
    const int64_t E = group.tms[L].attn.embed_dim;
    const size_t actual_elems = (size_t)S * (size_t)E;

    std::string y_name = (L == group.N - 1)
        ? std::string(XDNA_TBF_Y)
        : (std::string("y_") + std::to_string(L));
    auto it = entry->layout.buffers.find(y_name);
    if (it == entry->layout.buffers.end() || it->second.buf_type != "output") {
        GGML_LOG_ERROR("ggml-xdna: fused tblock multi '%s' missing from layout or wrong buf_type\n",
                       y_name.c_str());
        return false;
    }
    if (it->second.length < actual_elems * sizeof(uint16_t)) return false;
    const uint16_t * src = (const uint16_t *)((const char *)entry->output_bo->map<void*>()
                                              + it->second.offset);
    const int add_idx = group.tms[L].ffn_residual_add_idx;
    struct ggml_tensor * dst = cgraph->nodes[add_idx];

    const size_t need_bytes = (dst->type == GGML_TYPE_F32)
        ? actual_elems * sizeof(float)
        : actual_elems * sizeof(uint16_t);
    const size_t dst_nbytes = ggml_nbytes(dst);
    if (!dst->data || dst_nbytes < need_bytes) {
        GGML_LOG_ERROR("ggml-xdna: fused tblock multi '%s' dst at node %d "
                       "has data=%p nbytes=%zu need=%zu (type=%d, ne=[%lld,%lld,%lld,%lld])\n",
                       y_name.c_str(), add_idx, (void*)dst->data,
                       dst_nbytes, need_bytes, (int)dst->type,
                       (long long)dst->ne[0], (long long)dst->ne[1],
                       (long long)dst->ne[2], (long long)dst->ne[3]);
        return false;
    }
    if (dst->type == GGML_TYPE_F32) {
        bf16_to_f32(src, (float *)dst->data, actual_elems);
    } else if (dst->type == GGML_TYPE_BF16) {
        memcpy(dst->data, src, actual_elems * sizeof(uint16_t));
    } else {
        GGML_LOG_ERROR("ggml-xdna: fused tblock multi unsupported dst dtype %d\n",
                       (int)dst->type);
        return false;
    }
    return true;
}

// ============================================================================
// RMSNorm dispatch — standalone single-kernel op. Opt-in via XDNA_ENABLE_RMS_NORM.
// ============================================================================

// Human-readable cache key for standalone RMSNorm. The Python bridge's
// rms_norm_cache_key() hashes a JSON blob for its own cache layer, but the
// C++ path always drives compilation through ensure_rms_norm_compiled() with
// an explicit --out <bundle_dir>, so both sides agree on layout regardless of
// the scheme here. Readable names make the on-disk cache easier to audit.
static std::string make_rms_norm_cache_key(int64_t size, const char * dtype,
                                           int num_cols, int num_channels,
                                           int tile_size, bool weighted) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "rms_norm_S%lld_%s_%dc%dch_t%d%s",
             (long long)size, dtype, num_cols, num_channels, tile_size,
             weighted ? "_w" : "");
    return std::string(buf);
}

// Select (num_cols, num_channels) such that size is divisible by the
// per-core tile footprint. Prefer more cores (wider AIE utilization); fall
// back on the divisibility constraint. Returns true on success.
static bool xdna_select_rms_norm_params(int64_t size, int max_cols,
                                        int * out_cols, int * out_channels,
                                        int * out_tile_size) {
    const int tile_size = 32;  // matches the bridge default and IRON test grid

    // Support forcing ch=1 via environment variable (useful for matching specific kernels)
    const bool force_ch1 = xdna_env_enabled("GGML_XDNA_FORCE_CH1");
    std::vector<int> ch_candidates = force_ch1 ? std::vector<int>{1} : std::vector<int>{2, 1};

    // Try widest column counts first, then fall back.
    for (int cols : {max_cols, 4, 2, 1}) {
        if (cols < 1 || cols > max_cols) continue;
        for (int ch : ch_candidates) {
            if ((int64_t)cols * ch * tile_size == 0) continue;
            if (size % ((int64_t)cols * ch * tile_size) != 0) continue;
            *out_cols       = cols;
            *out_channels   = ch;
            *out_tile_size  = tile_size;
            return true;
        }
    }
    return false;
}

static bool rms_norm_bundle_present(const std::string & bundle_dir) {
    std::string xclbin_path = bundle_dir + "\\combined.xclbin";
    std::string insts_path = bundle_dir + "\\rms_norm_main.insts";

    std::ifstream xf(xclbin_path);
    if (!xf.good()) {
        // fprintf(stderr, "ggml-xdna: rms_norm_bundle_present: missing %s\n", xclbin_path.c_str());
        return false;
    }
    std::ifstream f(insts_path);
    if (!f.good()) {
        // fprintf(stderr, "ggml-xdna: rms_norm_bundle_present: missing %s\n", insts_path.c_str());
        return false;
    }
    return true;
}

static bool xdna_env_enabled(const char * name) {
    const char * val = getenv(name);
    return val != NULL && strcmp(val, "0") != 0 && strcmp(val, "OFF") != 0 && strcmp(val, "off") != 0;
}

static bool ensure_rms_norm_compiled(ggml_backend_xdna_context * ctx,
                                     const std::string & cache_key,
                                     int64_t size, int num_cols,
                                     int num_channels, int tile_size,
                                     bool weighted) {
    if (!xdna_env_enabled("XDNA_ENABLE_RMS_NORM")) return false;

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;

    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    if (dbg) {
        fprintf(stderr, "ggml-xdna: searching for rms_norm in %s\n", bundle_dir.c_str());
    }

    if (rms_norm_bundle_present(bundle_dir)) {
        if (dbg) {
            fprintf(stderr, "ggml-xdna: found precompiled rms_norm in cache\n");
        }
        return true;
    }

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "%s \"%s\" --quiet rms_norm --size %lld --dtype bf16 "
             "--num-aie-columns %d --num-channels %d --tile-size %d%s "
             "--out \"%s\"%s",
             xdna_python_cmd(),
             ctx->compile_script.c_str(),
             (long long)size, num_cols, num_channels, tile_size,
             weighted ? " --weighted" : "",
             bundle_dir.c_str(), xdna_null_redirect());
    fprintf(stderr, "ggml-xdna: compiling RMSNorm size=%lld cols=%d ch=%d tile=%d "
                  "(first run, will be cached)...\n",
                  (long long)size, num_cols, num_channels, tile_size);

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: RMSNorm compilation failed (exit code %d)\n", ret);
        return false;
    }

    if (!rms_norm_bundle_present(bundle_dir)) {
        GGML_LOG_ERROR("ggml-xdna: RMSNorm compilation succeeded but bundle "
                       "files missing in %s\n", bundle_dir.c_str());
        return false;
    }

    fprintf(stderr, "ggml-xdna: RMSNorm compilation complete, cached at %s\n",
                  bundle_dir.c_str());
    return true;
}

static xdna_rms_norm_entry * get_or_load_rms_norm_kernel(
        ggml_backend_xdna_context * ctx,
        const std::string & cache_key,
        int64_t size, int num_cols, int num_channels, int tile_size) {
    std::lock_guard<std::mutex> lock(ctx->cache_mutex);

    auto it = ctx->rms_norm_cache.find(cache_key);
    if (it != ctx->rms_norm_cache.end()) {
        return &it->second;
    }

    const std::string bundle_dir = ctx->cache_dir + "\\" + cache_key;
    const std::string xclbin_path = bundle_dir + "/combined.xclbin";
    const std::string insts_path  = bundle_dir + "/rms_norm_main.insts";

    try {
        xdna_rms_norm_entry entry;
        entry.cache_key    = cache_key;
        entry.size         = size;
        entry.num_cols     = num_cols;
        entry.num_channels = num_channels;
        entry.tile_size    = tile_size;

        entry.xclbin = xrt::xclbin(xclbin_path);
        ctx->device.register_xclbin(entry.xclbin);
        auto uuid = entry.xclbin.get_uuid();
        entry.hw_ctx = xrt::hw_context(ctx->device, uuid);

        entry.kernel = xrt::kernel(entry.hw_ctx, "MLIR_AIE");

        entry.insts = read_binary_file(insts_path);
        if (entry.insts.empty()) {
            GGML_LOG_ERROR("ggml-xdna: failed to read RMSNorm insts file: %s\n",
                           insts_path.c_str());
            return nullptr;
        }
        entry.insts_bo = xrt::bo(ctx->device, entry.insts.size(),
                                 xrt::bo::flags::cacheable,
                                 entry.kernel.group_id(1));
        entry.insts_bo.write(entry.insts.data());
        entry.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto [inserted_it, _] = ctx->rms_norm_cache.emplace(cache_key, std::move(entry));
        fprintf(stderr, "ggml-xdna: loaded RMSNorm kernel for %s\n", cache_key.c_str());
        return &inserted_it->second;

    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: failed to load RMSNorm kernel %s: %s\n",
                       cache_key.c_str(), e.what());
        return nullptr;
    }
}

// Dispatch a single GGML_OP_RMS_NORM node to the NPU. Returns true on success,
// false on any fallback condition (gate off, unsupported shape/dtype, eps
// mismatch, XRT failure). Caller must fall the node into the CPU accumulator
// when this returns false.
static bool ggml_backend_xdna_rms_norm(ggml_backend_xdna_context * ctx,
                                       struct ggml_tensor * node) {
    static const bool rms_enabled = xdna_env_enabled("XDNA_ENABLE_RMS_NORM");
    if (!rms_enabled) return false;
    if (!ctx->device_valid) return false;
    if (node->op != GGML_OP_RMS_NORM) return false;

    const struct ggml_tensor * src = node->src[0];
    if (src == nullptr) return false;
    if (!ggml_is_contiguous(src)) return false;
    if (!ggml_is_contiguous(node)) return false;

    // Only f32 or bf16 supported on the activation side. Weight src[1] (which
    // ggml uses for RMSNormWeighted) isn't wired in yet.
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_BF16) return false;
    if (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) return false;

    // Epsilon is baked into the AIE kernel at 1e-5. For typical activations
    // with mean(x^2) ~ O(1), the difference between 1e-6 and 1e-4 changes
    // inv_rms by < 1e-4, well below bf16's quantization noise. Widened
    // tolerance so Qwen/Llama-family models (eps in {1e-6, 1e-5}) both
    // dispatch. Reject only pathologically large eps that would materially
    // change the normalization.
    float eps = 0.0f;
    std::memcpy(&eps, node->op_params, sizeof(float));
    if (!(eps >= 1e-7f && eps <= 1e-4f)) {
        static const bool dbg = getenv("XDNA_DEBUG") != NULL;
        if (dbg) {
            fprintf(stderr, "ggml-xdna: rms_norm reject: eps=%.3e outside "
                            "[1e-7, 1e-4] (kernel hardcodes 1e-5)\n", (double)eps);
        }
        return false;
    }

    const int64_t size  = src->ne[0];
    const int64_t nrows = src->ne[1] * src->ne[2] * src->ne[3];
    if (size <= 0 || nrows <= 0) return false;

    int num_cols = 0, num_channels = 0, tile_size = 0;
    // NPU1 = 4 cols, NPU2 = 8 cols. Use the same "8 first, fall back to 4"
    // strategy other paths follow. The selector rejects configs that don't
    // divide; xrt::device.cols isn't exposed cleanly through the C++ API so
    // we try both and let the divisibility check decide.
    const int device_max_cols = ctx->num_cols;
    static const bool dbg = getenv("XDNA_DEBUG") != NULL;
    if (!xdna_select_rms_norm_params(size, device_max_cols,
                                     &num_cols, &num_channels, &tile_size)) {
        if (dbg) {
            fprintf(stderr, "ggml-xdna: rms_norm reject: no (cols,channels,"
                            "tile) divides size=%lld\n", (long long)size);
        }
        return false;
    }

    const std::string cache_key = make_rms_norm_cache_key(
        size, "bf16", num_cols, num_channels, tile_size, /*weighted=*/false);

    if (dbg) {
        fprintf(stderr, "ggml-xdna: using rms_norm cache_key: %s\n", cache_key.c_str());
    }

    if (!ensure_rms_norm_compiled(ctx, cache_key, size, num_cols, num_channels,
                                  tile_size, /*weighted=*/false)) {
        return false;
    }

    xdna_rms_norm_entry * entry = get_or_load_rms_norm_kernel(
        ctx, cache_key, size, num_cols, num_channels, tile_size);
    if (!entry) return false;

    try {
        const size_t elem_bytes = sizeof(uint16_t);  // bf16 on the NPU
        const size_t row_elems  = (size_t)size;
        const size_t row_bytes  = row_elems * elem_bytes;

        // Lazily allocate persistent input/output BOs sized for one row. Rows
        // are processed one at a time — RMSNorm is per-row and the AIE
        // dataflow is shaped for a single (size,) vector per invocation.
        if (!entry->in_bo) {
            entry->in_bo = std::make_unique<xrt::bo>(
                ctx->device, row_bytes, xrt::bo::flags::host_only,
                entry->kernel.group_id(3));
        }
        if (!entry->out_bo) {
            entry->out_bo = std::make_unique<xrt::bo>(
                ctx->device, row_bytes, xrt::bo::flags::host_only,
                entry->kernel.group_id(4));
        }

        uint16_t * in_map  = entry->in_bo->map<uint16_t *>();
        uint16_t * out_map = entry->out_bo->map<uint16_t *>();

        for (int64_t r = 0; r < nrows; r++) {
            const char * src_row = (const char *)src->data + (size_t)r * src->nb[1];
            char * dst_row       = (char *)node->data + (size_t)r * node->nb[1];

            if (src->type == GGML_TYPE_F32) {
                f32_to_bf16((const float *)src_row, in_map, row_elems);
            } else {
                std::memcpy(in_map, src_row, row_bytes);
            }
            entry->in_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto run = entry->kernel(3, entry->insts_bo,
                                     (uint32_t)entry->insts.size(),
                                     *entry->in_bo, *entry->out_bo);
            run.wait();

            entry->out_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            if (node->type == GGML_TYPE_F32) {
                bf16_to_f32(out_map, (float *)dst_row, row_elems);
            } else {
                std::memcpy(dst_row, out_map, row_bytes);
            }
        }
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-xdna: RMSNorm XRT dispatch failed (%s)\n", e.what());
        return false;
    }

    if (dbg) {
        fprintf(stderr, "ggml-xdna: rms_norm dispatch ok size=%lld nrows=%lld "
                        "cols=%d channels=%d tile=%d\n",
                (long long)size, (long long)nrows, num_cols, num_channels, tile_size);
    }
    return true;
}

// ============================================================================
// Backend interface
// ============================================================================

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return "XDNA";

    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;
    static const bool debug = getenv("XDNA_DEBUG") != NULL;
    if (debug) {
        auto mib = [](size_t b) { return (double)b / (1024.0 * 1024.0); };
        fprintf(stderr,
                "ggml-xdna: buffer traffic summary — "
                "set_tensor: %zu calls / %.1f MiB, "
                "get_tensor: %zu calls / %.1f MiB, "
                "cpy_tensor: %zu calls / %.1f MiB\n",
                g_set_tensor_calls.load(), mib(g_set_tensor_bytes.load()),
                g_get_tensor_calls.load(), mib(g_get_tensor_bytes.load()),
                g_cpy_tensor_calls.load(), mib(g_cpy_tensor_bytes.load()));
    }
    delete ctx;
    delete backend;
}

// Can we run this node on the NPU right now? (MUL_MAT + tileable shape.)
static bool xdna_node_npu_dispatchable(const struct ggml_tensor * node) {
    // TEMP DEBUG: force CPU fallback for all MUL_MAT to isolate NPU bug from scheduler/buffer bugs.
    static const bool force_cpu = getenv("XDNA_FORCE_CPU") != NULL;
    if (force_cpu) return false;
    if (node->op != GGML_OP_MUL_MAT) return false;
    const struct ggml_tensor * s0 = node->src[0];
    const struct ggml_tensor * s1 = node->src[1];
    const int64_t K = s0->ne[0];
    const int64_t N = s0->ne[1];
    const int64_t M = s1->ne[1];
    if (M == 1) return xdna_shape_dispatchable_gemv(K, N);
    return xdna_shape_dispatchable(M, K, N);
}

// Local equivalent of ggml_graph_view (not exported by ggml-base)
static struct ggml_cgraph xdna_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph;
    memset(&cgraph, 0, sizeof(cgraph));
    cgraph.n_nodes = i1 - i0;
    cgraph.nodes = cgraph0->nodes + i0;
    cgraph.use_counts = cgraph0->use_counts;
    cgraph.visited_hash_set = cgraph0->visited_hash_set;
    cgraph.order = cgraph0->order;
    return cgraph;
}

// Delegate a contiguous run of nodes [i0, i1) to the CPU backend as a single
// batched call — cheaper than per-node calls and lets the CPU backend keep
// its own state (threads, arenas) across related nodes.
static ggml_status xdna_delegate_range(ggml_backend_xdna_context * ctx,
                                       struct ggml_cgraph * cgraph,
                                       int i0, int i1) {
    if (i1 <= i0) return GGML_STATUS_SUCCESS;
    if (!ctx->cpu_backend) {
        GGML_LOG_ERROR("ggml-xdna: CPU backend not initialized, cannot delegate\n");
        return GGML_STATUS_FAILED;
    }
    struct ggml_cgraph sub = xdna_graph_view(cgraph, i0, i1);
    return ggml_backend_graph_compute(ctx->cpu_backend, &sub);
}

// ============================================================================
// Expanded decode-attention matcher for FlowKV.
//
// Matches the pattern: MUL_MAT(Q@K^T) [SCALE] [ADD mask] SOFT_MAX MUL_MAT(scores@V)

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;
    int n = cgraph->n_nodes;

    // Reset attention-prefill per-phase profiling samples (no-op when gate off).
    xdna_attn_prof_reset();

    static const bool debug = getenv("XDNA_DEBUG") != NULL;
    static const bool attention_prefill_dbg_enabled = xdna_env_enabled("XDNA_ENABLE_ATTENTION_PREFILL");
    static const bool transformer_block_dbg_enabled =
        (xdna_env_enabled("XDNA_ENABLE_TRANSFORMER_BLOCK")) ||
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED"));
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
    static const bool swiglu_enabled = xdna_env_enabled("XDNA_ENABLE_SWIGLU");
    static const bool qkv_enabled    = xdna_env_enabled("XDNA_ENABLE_QKV");
    static const bool rms_norm_enabled = xdna_env_enabled("XDNA_ENABLE_RMS_NORM");
    static const bool attention_prefill_enabled = xdna_env_enabled("XDNA_ENABLE_ATTENTION_PREFILL");
    static const bool transformer_block_enabled =
        (xdna_env_enabled("XDNA_ENABLE_TRANSFORMER_BLOCK")) ||
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED"));
    static const bool flowkv_decode_enabled = xdna_env_enabled("XDNA_ENABLE_FLOWKV_DECODE");
    // ── FlowKV POC: saved permuted Q/K/V tensor pointers ──────────
    // Persisted across graph_compute calls so CONT(kqv_out) segment
    // can use Q/K/V from the preceding QKV segment.
    static const struct ggml_tensor * flowkv_poc_q_perm = nullptr;
    static const struct ggml_tensor * flowkv_poc_k_perm = nullptr;
    static const struct ggml_tensor * flowkv_poc_v_perm = nullptr;
    static int64_t flowkv_poc_head_dim = 0;
    static int64_t flowkv_poc_seq_len = 0;
    static int64_t flowkv_poc_num_kv_heads = 0;
    static int64_t flowkv_poc_num_q_heads = 0;
    static bool flowkv_poc_valid = false;
    // NOTE: Do NOT reset flowkv_poc_valid per cgraph — QKV dispatch and
    // CONT(kqv_out) are in different cgraphs (scheduler splits them).
    // Pointers are overwritten per-layer; stale state is harmless because
    // each layer's QKV dispatch runs before its CONT(kqv_out).

    // FlowKV early-dispatch: set after QKV segment when permuted tensors
    // are ready. Used to skip SOFT_MAX and CONT(kqv_out) in subsequent
    // segments of the same layer — FlowKV already computed full attention.
    static bool flowkv_dispatched_this_layer = false;

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
    // Also excludes attention Q@K^T MUL_MATs (followed by SOFT_MAX).
    xdna_decode_batcher decode_batcher;
    std::unordered_set<int> decode_batch_indices;
    if (decode_batcher.is_enabled()) {
        decode_batch_indices = xdna_plan_decode_batch(cgraph, qkv_plan);
    }

    // Pre-scan: identify MUL_MAT nodes that are part of attention patterns.
    // Two cases:
    //   1. Q@K^T: MUL_MAT followed by [SCALE] [ADD] SOFT_MAX
    //   2. scores@V: MUL_MAT preceded by SOFT_MAX (within a short window)
    // Both have permuted/interleaved data that the NPU GEMV kernel cannot
    // handle correctly. They must stay on CPU. This set covers the individual
    // dispatch path (decode_batch has its own equivalent check).
    std::unordered_set<int> attn_mulmat_indices;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) continue;
        if (node->src[1]->ne[1] != 1) continue;
        // Case 1: forward scan — Q@K^T before SOFT_MAX
        for (int j = i + 1; j < cgraph->n_nodes && j < i + 8; j++) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_SOFT_MAX) {
                attn_mulmat_indices.insert(i);
                break;
            }
            if (op == GGML_OP_SCALE || op == GGML_OP_ADD ||
                op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                op == GGML_OP_PERMUTE || op == GGML_OP_CONT) continue;
            break;
        }
        if (attn_mulmat_indices.count(i)) continue;
        // Case 2: backward scan — scores@V after SOFT_MAX
        for (int j = i - 1; j >= 0 && j > i - 8; j--) {
            enum ggml_op op = cgraph->nodes[j]->op;
            if (op == GGML_OP_SOFT_MAX) {
                attn_mulmat_indices.insert(i);
                break;
            }
            if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
                op == GGML_OP_PERMUTE || op == GGML_OP_CONT) continue;
            break;
        }
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
    if (transformer_block_enabled && !xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED")) {
        // L3A prewarm uploads per-kernel weight BOs that the fused path
        // doesn't use. Skip it in fused mode — fused weights are written
        // at first dispatch per layer, into the single input_bo.
        tblock_prefill_bulk_prewarm(ctx, cgraph);
    }
    if (transformer_block_enabled && xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED")) {
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

        // FlowKV: skip SOFT_MAX when FlowKV already dispatched full
        // attention in the preceding QKV segment. The CPU softmax would
        // compute Q@K^T scores that FlowKV replaces entirely.
        if (flowkv_decode_enabled && flowkv_dispatched_this_layer &&
            node->op == GGML_OP_SOFT_MAX) {
            if (cpu_run_start >= 0) {
                ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, i);
                if (s != GGML_STATUS_SUCCESS) return s;
                cpu_run_start = -1;
            }
            continue;
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
                // ── FlowKV diagnostic: log QKV tensors and surrounding nodes ──
                if (flowkv_decode_enabled) {
                    fprintf(stderr, "ggml-xdna: [FlowKV-DIAG] QKV @%d: "
                            "q_mm dst=%s ne=[%lld,%lld] src1=%s ne=[%lld,%lld]\n",
                            i, q_mm->name,
                            (long long)q_mm->ne[0], (long long)q_mm->ne[1],
                            q_mm->src[1]->name,
                            (long long)q_mm->src[1]->ne[0], (long long)q_mm->src[1]->ne[1]);
                    // Log nodes i..i+20 (the QKV + cache section)
                    int log_end = i + 20 < n ? i + 20 : n;
                    for (int j = i; j < log_end; j++) {
                        struct ggml_tensor * nd = cgraph->nodes[j];
                        fprintf(stderr, "  [%d] op=%d name=%s ne=[%lld,%lld,%lld] "
                                "src0=%s src1=%s\n",
                                j, (int)nd->op, nd->name,
                                (long long)nd->ne[0], (long long)nd->ne[1], (long long)nd->ne[2],
                                nd->src[0] ? nd->src[0]->name : "(null)",
                                nd->src[1] ? nd->src[1]->name : "(null)");
                    }
                    fflush(stderr);
                }
                // ── FlowKV POC: save permuted Q/K/V tensor pointers ──
                // The segment structure after Q MUL_MAT is fixed:
                //   +15 = PERMUTE cache_v (V cache permuted)
                //   +17 = PERMUTE cache_k (K cache permuted)
                //   +19 = PERMUTE Qcur   (Q rotated permuted)
                // Save these for the CONT(kqv_out) FlowKV dispatch.
                if (flowkv_decode_enabled && q_mm->src[1]->ne[1] == 1) {
                    int v_perm_idx = i + 15;
                    int k_perm_idx = i + 17;
                    int q_perm_idx = i + 19;
                    if (q_perm_idx < n) {
                        struct ggml_tensor * v_perm = cgraph->nodes[v_perm_idx];
                        struct ggml_tensor * k_perm = cgraph->nodes[k_perm_idx];
                        struct ggml_tensor * q_perm = cgraph->nodes[q_perm_idx];
                        if (q_perm->op == GGML_OP_PERMUTE &&
                            k_perm->op == GGML_OP_PERMUTE &&
                            v_perm->op == GGML_OP_PERMUTE) {
                            flowkv_poc_q_perm = q_perm;
                            flowkv_poc_k_perm = k_perm;
                            flowkv_poc_v_perm = v_perm;
                            flowkv_poc_head_dim = q_perm->ne[0];       // 64
                            flowkv_poc_seq_len = k_perm->ne[1];        // seq_len
                            flowkv_poc_num_kv_heads = k_perm->ne[2];   // 8
                            flowkv_poc_num_q_heads = q_perm->ne[2];    // 32
                            flowkv_poc_valid = true;
                            fprintf(stderr, "ggml-xdna: [FlowKV-POC] Saved Q=%p[%lld,%lld,%lld] "
                                    "K=%p[%lld,%lld,%lld] V=%p[%lld,%lld,%lld]\n",
                                    q_perm->data, (long long)q_perm->ne[0], (long long)q_perm->ne[1], (long long)q_perm->ne[2],
                                    k_perm->data, (long long)k_perm->ne[0], (long long)k_perm->ne[1], (long long)k_perm->ne[2],
                                    v_perm->data, (long long)v_perm->ne[0], (long long)v_perm->ne[1], (long long)v_perm->ne[2]);
                            fflush(stderr);
                        }
                    }
                }
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
        //
        // The plan is rebuilt on every segment to enable cross-segment matching:
        // when the graph scheduler splits the attention pattern (Q@K^T + SOFT_MAX
        // in one segment, scores@V in the next), pending heads carry over via
        // static state inside xdna_plan_flowkv().
        if (flowkv_decode_enabled) {
            static std::vector<xdna_flowkv_group> flowkv_groups;
            static std::unordered_set<int> flowkv_dispatched_qk;
            static std::unordered_set<int> flowkv_dispatched_pv;
            static const struct ggml_cgraph * flowkv_last_cgraph = nullptr;

            // Clear dispatched sets when cgraph changes (new inference).
            if (flowkv_last_cgraph != cgraph) {
                flowkv_dispatched_qk.clear();
                flowkv_dispatched_pv.clear();
                flowkv_last_cgraph = cgraph;
            }

            // Rebuild plan every segment (enables cross-segment matching).
            flowkv_groups = xdna_plan_flowkv(cgraph);

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
                    xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED");
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

        // Attention guard: never dispatch Q@K^T MUL_MATs to NPU as GEMVs.
        // These are attention score computations (followed by SOFT_MAX) with
        // permuted Q and interleaved K cache — the NPU GEMV kernel produces
        // garbage on them. The decode_batch plan already excludes them; this
        // check covers the individual dispatch path.
        if (attn_mulmat_indices.count(i)) {
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

        // ── FlowKV POC: dispatch at CONT(kqv_out) boundary ─────────
        // CPU attention has already run and written to kqv_out.
        // We dispatch FlowKV on NPU and OVERWRITE kqv_out with the
        // NPU result. If output matches → FlowKV computes correctly.
        if (flowkv_decode_enabled && flowkv_poc_valid &&
            node->op == GGML_OP_CONT &&
            node->name && strstr(node->name, "kqv_out")) {

            struct ggml_tensor * kqv_out = node->src[0];
            static const bool poc_dbg = getenv("XDNA_DEBUG") != NULL;
            int64_t head_dim = flowkv_poc_head_dim;
            int64_t seq_len = flowkv_poc_seq_len;
            int64_t num_kv_heads = flowkv_poc_num_kv_heads;
            int64_t num_q_heads = flowkv_poc_num_q_heads;
            int64_t q_heads_per_kv = num_q_heads / num_kv_heads;
            int chunk_size = 32;
            int num_cols = 1;

            if (poc_dbg) {
                fprintf(stderr, "ggml-xdna: [FlowKV-POC] CONT(kqv_out) @%d: "
                        "head_dim=%lld seq_len=%lld kv_heads=%lld q_heads=%lld "
                        "kqv_out data=%p type=%d | Q type=%d K type=%d V type=%d\n",
                        i, (long long)head_dim, (long long)seq_len,
                        (long long)num_kv_heads, (long long)num_q_heads,
                        kqv_out->data, (int)kqv_out->type,
                        (int)flowkv_poc_q_perm->type,
                        (int)flowkv_poc_k_perm->type,
                        (int)flowkv_poc_v_perm->type);
                fflush(stderr);
            }

            // Ensure FlowKV kernel is compiled and loaded.
            xdna_flowkv_entry * fk_entry = get_or_load_flowkv_kernel(
                ctx, q_heads_per_kv, /*num_kv_heads=*/1, head_dim, seq_len, chunk_size, num_cols);

            if (fk_entry) {
                try {
                    size_t row_bytes = head_dim * sizeof(uint16_t);

                    // Allocate BOs if needed.
                    // IRON xclbin arg layout: opcode(0), insts(1), insts_size(2),
                    // DDR_buf_0(3)=kv, DDR_buf_1(4)=q, DDR_buf_2(5)=out
                    size_t kv_size = 1 * seq_len * 2 * row_bytes;
                    if (!fk_entry->bo_kv) {
                        fk_entry->bo_kv = std::make_unique<xrt::bo>(
                            xrt::bo(ctx->device, kv_size, xrt::bo::flags::host_only,
                                    fk_entry->kernel.group_id(3)));
                    }
                    size_t q_size = 1 * (q_heads_per_kv * head_dim + head_dim + 2) * sizeof(uint16_t);  // +2 for actual_seq_len + DMA alignment
                    if (!fk_entry->bo_q) {
                        fk_entry->bo_q = std::make_unique<xrt::bo>(
                            xrt::bo(ctx->device, q_size, xrt::bo::flags::host_only,
                                    fk_entry->kernel.group_id(4)));
                    }
                    size_t out_size = q_heads_per_kv * row_bytes;
                    if (!fk_entry->bo_out) {
                        fk_entry->bo_out = std::make_unique<xrt::bo>(
                            xrt::bo(ctx->device, out_size, xrt::bo::flags::host_only,
                                    fk_entry->kernel.group_id(5)));
                    }

                    // Diagnostic: save CPU result before overwriting
                    static std::vector<float> cpu_save;
                    static bool cpu_saved = false;
                    if (poc_dbg && !cpu_saved) {
                        cpu_save.assign((float *)kqv_out->data,
                                        (float *)kqv_out->data + num_q_heads * head_dim);
                        cpu_saved = true;
                    }

                    const char * k_data = (const char *)flowkv_poc_k_perm->data;
                    const char * v_data = (const char *)flowkv_poc_v_perm->data;
                    const char * q_data = (const char *)flowkv_poc_q_perm->data;
                    size_t k_nb0 = flowkv_poc_k_perm->nb[0];
                    size_t k_nb1 = flowkv_poc_k_perm->nb[1];
                    size_t k_nb2 = flowkv_poc_k_perm->nb[2];
                    size_t v_nb0 = flowkv_poc_v_perm->nb[0];
                    size_t v_nb1 = flowkv_poc_v_perm->nb[1];
                    size_t v_nb2 = flowkv_poc_v_perm->nb[2];
                    size_t q_nb0 = flowkv_poc_q_perm->nb[0];
                    size_t q_nb1 = flowkv_poc_q_perm->nb[1];
                    size_t q_nb2 = flowkv_poc_q_perm->nb[2];

                    // Detect tensor types for f32→bf16 conversion.
                    // QKV dispatch outputs f32 (bf16_to_f32), KV cache may be f32 or bf16.
                    // FlowKV kernel expects bf16 input, produces bf16 output.
                    const bool k_is_f32 = (flowkv_poc_k_perm->type == GGML_TYPE_F32);
                    const bool v_is_f32 = (flowkv_poc_v_perm->type == GGML_TYPE_F32);
                    const bool q_is_f32 = (flowkv_poc_q_perm->type == GGML_TYPE_F32);
                    const bool out_is_f32 = (kqv_out->type == GGML_TYPE_F32);

                    // === DIAG: dump tensor metadata and raw source data ===
                    if (poc_dbg) {
                        fprintf(stderr, "\n  [DIAG-FLOWKV] Tensor metadata:\n");
                        fprintf(stderr, "    K: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                                k_data, (int)flowkv_poc_k_perm->type,
                                (long long)flowkv_poc_k_perm->ne[0], (long long)flowkv_poc_k_perm->ne[1], (long long)flowkv_poc_k_perm->ne[2],
                                k_nb0, k_nb1, k_nb2);
                        fprintf(stderr, "    V: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                                v_data, (int)flowkv_poc_v_perm->type,
                                (long long)flowkv_poc_v_perm->ne[0], (long long)flowkv_poc_v_perm->ne[1], (long long)flowkv_poc_v_perm->ne[2],
                                v_nb0, v_nb1, v_nb2);
                        fprintf(stderr, "    Q: data=%p type=%d ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu]\n",
                                q_data, (int)flowkv_poc_q_perm->type,
                                (long long)flowkv_poc_q_perm->ne[0], (long long)flowkv_poc_q_perm->ne[1], (long long)flowkv_poc_q_perm->ne[2],
                                q_nb0, q_nb1, q_nb2);
                        fprintf(stderr, "    kqv_out: data=%p type=%d ne=[%lld,%lld,%lld]\n",
                                kqv_out->data, (int)kqv_out->type,
                                (long long)kqv_out->ne[0], (long long)kqv_out->ne[1], (long long)kqv_out->ne[2]);

                        // Dump raw K data at pos=0, kv_h=0 (first head_dim elements)
                        fprintf(stderr, "    K src raw [pos=0,kv_h=0] first 8 bf16:");
                        for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                            size_t off = 0 * k_nb1 + 0 * k_nb2 + d * k_nb0;
                            uint16_t val;
                            memcpy(&val, k_data + off, 2);
                            fprintf(stderr, " 0x%04X", val);
                        }
                        fprintf(stderr, "\n");

                        // Dump raw K data at pos=0, kv_h=0 as f32 (if f32 type)
                        if (k_is_f32) {
                            fprintf(stderr, "    K src f32 [pos=0,kv_h=0] first 8:");
                            for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                                size_t off = 0 * k_nb1 + 0 * k_nb2 + d * k_nb0;
                                float val;
                                memcpy(&val, k_data + off, 4);
                                fprintf(stderr, " %.6f", val);
                            }
                            fprintf(stderr, "\n");
                        }

                        // Dump raw V data at pos=0, kv_h=0
                        fprintf(stderr, "    V src raw [pos=0,kv_h=0] first 8 bf16:");
                        for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                            size_t off = 0 * v_nb1 + 0 * v_nb2 + d * v_nb0;
                            uint16_t val;
                            memcpy(&val, v_data + off, 2);
                            fprintf(stderr, " 0x%04X", val);
                        }
                        fprintf(stderr, "\n");

                        // Dump raw Q data at q_head=0 (convert to bf16 for comparison)
                        fprintf(stderr, "    Q src raw [head=0] first 8 (as bf16):");
                        for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                            size_t off = 0 * q_nb2 + d * q_nb0;
                            if (q_is_f32) {
                                float f32_val;
                                memcpy(&f32_val, q_data + off, 4);
                                uint16_t bf16_val;
                                f32_to_bf16(&f32_val, &bf16_val, 1);
                                fprintf(stderr, " 0x%04X", bf16_val);
                            } else {
                                uint16_t val;
                                memcpy(&val, q_data + off, 2);
                                fprintf(stderr, " 0x%04X", val);
                            }
                        }
                        fprintf(stderr, "\n");

                        // Also check K/V data at a middle position (pos=128)
                        fprintf(stderr, "    K src raw [pos=128,kv_h=0] first 8 bf16:");
                        for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                            size_t off = 128 * k_nb1 + 0 * k_nb2 + d * k_nb0;
                            uint16_t val;
                            memcpy(&val, k_data + off, 2);
                            fprintf(stderr, " 0x%04X", val);
                        }
                        fprintf(stderr, "\n");

                        // Check if K src[0] (the source tensor of the PERMUTE) has different data
                        struct ggml_tensor * k_src0 = flowkv_poc_k_perm->src[0];
                        if (k_src0 && k_src0->data && k_src0->data != k_data) {
                            const char * k_src0_data = (const char *)k_src0->data;
                            fprintf(stderr, "    K src0 (pre-permute): data=%p type=%d ne=[%lld,%lld,%lld]\n",
                                    k_src0_data, (int)k_src0->type,
                                    (long long)k_src0->ne[0], (long long)k_src0->ne[1], (long long)k_src0->ne[2]);
                            fprintf(stderr, "    K src0 raw [0] first 8 bf16:");
                            for (int d = 0; d < 8; d++) {
                                uint16_t val;
                                memcpy(&val, k_src0_data + d * 2, 2);
                                fprintf(stderr, " 0x%04X", val);
                            }
                            fprintf(stderr, "\n");
                        } else if (k_src0 && k_src0->data == k_data) {
                            fprintf(stderr, "    K src0 data == K perm data (same pointer)\n");
                        }

                        // Check V src[0]
                        struct ggml_tensor * v_src0 = flowkv_poc_v_perm->src[0];
                        if (v_src0 && v_src0->data && v_src0->data != v_data) {
                            const char * v_src0_data = (const char *)v_src0->data;
                            fprintf(stderr, "    V src0 (pre-permute): data=%p type=%d ne=[%lld,%lld,%lld]\n",
                                    v_src0_data, (int)v_src0->type,
                                    (long long)v_src0->ne[0], (long long)v_src0->ne[1], (long long)v_src0->ne[2]);
                            fprintf(stderr, "    V src0 raw [0] first 8 bf16:");
                            for (int d = 0; d < 8; d++) {
                                uint16_t val;
                                memcpy(&val, v_src0_data + d * 2, 2);
                                fprintf(stderr, " 0x%04X", val);
                            }
                            fprintf(stderr, "\n");
                        } else if (v_src0 && v_src0->data == v_data) {
                            fprintf(stderr, "    V src0 data == V perm data (same pointer)\n");
                        }

                        fflush(stderr);
                    }

                    // Identity RoPE angles (cos=0x3F80, sin=0x0000).
                    char angle_cos_bf16[2] = {(char)0x80, (char)0x3F};  // 0x3F80 = 1.0 bf16
                    char angle_sin_bf16[2] = {(char)0x00, (char)0x00};  // 0x0000 = 0.0 bf16

                    // Determine actual number of filled KV positions.
                    // k_perm->ne[1] is the padded n_kv (≥256), not the real token count.
                    // We scan K data to find the last non-zero position.
                    int64_t actual_seq_len = seq_len;
                    {
                        // Quick scan: check if K at position seq_len-1 is all zeros.
                        // If so, binary search for the boundary.
                        auto is_k_zero = [&](int64_t pos) -> bool {
                            for (int64_t d = 0; d < head_dim; d++) {
                                size_t off = pos * k_nb1 + 0 * k_nb2 + d * k_nb0;
                                uint16_t val;
                                memcpy(&val, k_data + off, 2);
                                if (val != 0) return false;
                            }
                            return true;
                        };
                        if (is_k_zero(seq_len - 1)) {
                            int64_t lo = 0, hi = seq_len - 1;
                            while (lo < hi) {
                                int64_t mid = (lo + hi + 1) / 2;
                                if (is_k_zero(mid)) hi = mid - 1;
                                else lo = mid;
                            }
                            actual_seq_len = lo + 1;
                        }
                    }
                    if (poc_dbg) {
                        fprintf(stderr, "    actual_seq_len=%lld (padded=%lld)\n",
                                (long long)actual_seq_len, (long long)seq_len);
                    }

                    // Process each KV head.
                    for (int64_t kv_h = 0; kv_h < num_kv_heads; kv_h++) {
                        // --- Prepare KV buffer (contiguous K then V, bf16) ---
                        // Layout: [K_pos0 K_pos1 ... K_posN | V_pos0 V_pos1 ... V_posN]
                        // Each region = seq_len * head_dim * 2 bytes.
                        {
                            auto kv_ptr = fk_entry->bo_kv->map<char *>();
                            // Zero entire BO first — unused positions must be zero
                            // so the kernel's online softmax doesn't dilute attention.
                            memset(kv_ptr, 0, (size_t)(1 * seq_len * 2 * row_bytes));
                            size_t kv_region = (size_t)seq_len * row_bytes;  // size of K or V region
                            for (int64_t pos = 0; pos < actual_seq_len; pos++) {
                                // K: copy head_dim elements → bf16 in K region.
                                size_t dst_k = pos * row_bytes;
                                const bool k_is_f16 = (flowkv_poc_k_perm->type == GGML_TYPE_F16);
                                if (k_is_f32) {
                                    // f32 K: convert each element to bf16.
                                    const float * src = (const float *)(k_data + pos * k_nb1 + kv_h * k_nb2);
                                    uint16_t * dst = (uint16_t *)(kv_ptr + dst_k);
                                    f32_to_bf16(src, dst, (size_t)head_dim);
                                } else if (k_is_f16) {
                                    // f16 K: convert f16→f32→bf16.
                                    size_t src_k = pos * k_nb1 + kv_h * k_nb2;
                                    for (int64_t d = 0; d < head_dim; d++) {
                                        ggml_fp16_t f16_val;
                                        memcpy(&f16_val, k_data + src_k + d * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                                        float f32_val = ggml_fp16_to_fp32(f16_val);
                                        uint16_t bf16_val;
                                        f32_to_bf16(&f32_val, &bf16_val, 1);
                                        memcpy(kv_ptr + dst_k + d * 2, &bf16_val, 2);
                                    }
                                } else {
                                    // bf16 K: direct copy.
                                    size_t src_k = pos * k_nb1 + kv_h * k_nb2;
                                    memcpy(kv_ptr + dst_k, k_data + src_k, row_bytes);
                                }

                                // V: copy head_dim elements → bf16 in V region.
                                size_t dst_v = kv_region + pos * row_bytes;
                                const bool v_is_f16 = (flowkv_poc_v_perm->type == GGML_TYPE_F16);
                                if (v_is_f32) {
                                    // f32 V: gather with stride conversion.
                                    uint16_t * dst = (uint16_t *)(kv_ptr + dst_v);
                                    if (v_nb0 == (size_t)head_dim * 4) {
                                        // V layout: [head_dim, seq_len, kv_heads] → contiguous f32
                                        const float * src = (const float *)(v_data + pos * v_nb1 + kv_h * v_nb2);
                                        f32_to_bf16(src, dst, (size_t)head_dim);
                                    } else {
                                        // V layout: [seq_len, head_dim, kv_heads] → strided f32
                                        for (int64_t d = 0; d < head_dim; d++) {
                                            const float * src = (const float *)(v_data + pos * v_nb0 + d * v_nb1 + kv_h * v_nb2);
                                            float val = *src;
                                            // Manual f32→bf16 round-to-nearest-even.
                                            uint32_t bits;
                                            memcpy(&bits, &val, 4);
                                            uint16_t bf = (uint16_t)(bits >> 16);
                                            // Round: add 0x7FFF + round-bit, then shift.
                                            if ((bits & 0x7FFFFFFF) < 0x7F800000) {
                                                uint32_t rounding = 0x7FFF + ((bf & 1) ^ 1);
                                                bf = (uint16_t)((bits + rounding) >> 16);
                                            }
                                            memcpy(kv_ptr + dst_v + d * 2, &bf, 2);
                                        }
                                    }
                                } else if (v_is_f16) {
                                    // f16 V: convert f16→f32→bf16 with stride gather.
                                    for (int64_t d = 0; d < head_dim; d++) {
                                        size_t src_v = pos * v_nb0 + d * v_nb1 + kv_h * v_nb2;
                                        ggml_fp16_t f16_val;
                                        memcpy(&f16_val, v_data + src_v, sizeof(ggml_fp16_t));
                                        float f32_val = ggml_fp16_to_fp32(f16_val);
                                        uint16_t bf16_val;
                                        f32_to_bf16(&f32_val, &bf16_val, 1);
                                        memcpy(kv_ptr + dst_v + d * 2, &bf16_val, 2);
                                    }
                                } else {
                                    // bf16 V: gather with correct strides.
                                    if (v_nb0 == (size_t)head_dim * 2) {
                                        // V layout: [head_dim, seq_len, kv_heads] → contiguous bf16
                                        size_t src_v = pos * v_nb1 + kv_h * v_nb2;
                                        memcpy(kv_ptr + dst_v, v_data + src_v, row_bytes);
                                    } else {
                                        // V layout: [seq_len, head_dim, kv_heads] → strided bf16
                                        for (int64_t d = 0; d < head_dim; d++) {
                                            size_t src_v = pos * v_nb0 + d * v_nb1 + kv_h * v_nb2;
                                            memcpy(kv_ptr + dst_v + d * 2, v_data + src_v, 2);
                                        }
                                    }
                                }
                            }
                            fk_entry->bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                            // === DIAG: host→DDR coherency check (POC path) ===
                            if (poc_dbg && kv_h == 0) {
                                // Save what we wrote
                                auto kv_map = fk_entry->bo_kv->map<char *>();
                                uint16_t pre_k0[8];
                                memcpy(pre_k0, kv_map, 16);

                                // Sync back from device
                                fk_entry->bo_kv->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                                auto verify_ptr = reinterpret_cast<const uint16_t*>(kv_map);

                                fprintf(stderr, "  [DIAG-SYNC] bo_kv verify after FROM_DEVICE:\n");
                                fprintf(stderr, "    K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                                    verify_ptr[0], verify_ptr[1], verify_ptr[2], verify_ptr[3],
                                    verify_ptr[4], verify_ptr[5], verify_ptr[6], verify_ptr[7]);

                                bool k0_match = (memcmp(pre_k0, kv_map, 16) == 0);
                                fprintf(stderr, "    K[0] match=%s\n", k0_match ? "YES" : "NO!!");
                                fflush(stderr);

                                // Re-sync TO_DEVICE
                                fk_entry->bo_kv->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                            }
                            // === END DIAG ===

                            // === DIAG: verify what was copied into bo_kv ===
                            if (poc_dbg && kv_h == 0) {
                                auto verify_ptr = fk_entry->bo_kv->map<char *>();
                                const uint16_t * verify_bf16 = (const uint16_t *)verify_ptr;
                                fprintf(stderr, "    [DIAG] bo_kv after copy, kv_h=0 (contiguous layout):\n");
                                fprintf(stderr, "      K[0][0:8] (offset 0):");
                                for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", verify_bf16[d]);
                                fprintf(stderr, "\n");
                                fprintf(stderr, "      K[1][0:8] (offset %d):", (int)head_dim);
                                for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", verify_bf16[head_dim + d]);
                                fprintf(stderr, "\n");
                                fprintf(stderr, "      V[0][0:8] (offset %d):", (int)(seq_len * head_dim));
                                for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", verify_bf16[seq_len * head_dim + d]);
                                fprintf(stderr, "\n");

                                // Also dump what SHOULD be at K[0] based on source strides
                                fprintf(stderr, "      K src expected [pos=0,kv_h=0,d=0:8]:");
                                for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                                    size_t off = 0 * k_nb1 + 0 * k_nb2 + d * k_nb0;
                                    uint16_t val;
                                    memcpy(&val, k_data + off, 2);
                                    fprintf(stderr, " 0x%04X", val);
                                }
                                fprintf(stderr, "\n");
                                fflush(stderr);
                            }
                        }

                        // --- Prepare Q buffer (bf16) ---
                        {
                            auto q_ptr = fk_entry->bo_q->map<char *>();
                            size_t q_head_bytes_bf16 = head_dim * sizeof(uint16_t);
                            const bool q_is_f16 = (flowkv_poc_q_perm->type == GGML_TYPE_F16);
                            for (int64_t g = 0; g < q_heads_per_kv; g++) {
                                int64_t q_head = kv_h * q_heads_per_kv + g;
                                const char * src = q_data + q_head * q_nb2;
                                char * dst = q_ptr + g * q_head_bytes_bf16;
                                if (q_is_f32) {
                                    f32_to_bf16((const float *)src, (uint16_t *)dst, (size_t)head_dim);
                                } else if (q_is_f16) {
                                    // f16 Q: convert f16→f32→bf16.
                                    for (int64_t d = 0; d < head_dim; d++) {
                                        ggml_fp16_t f16_val;
                                        memcpy(&f16_val, src + d * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                                        float f32_val = ggml_fp16_to_fp32(f16_val);
                                        f32_to_bf16(&f32_val, (uint16_t *)(dst + d * 2), 1);
                                    }
                                } else {
                                    memcpy(dst, src, q_head_bytes_bf16);
                                }
                            }
                            // Append identity RoPE angles.
                            size_t angles_off = q_heads_per_kv * q_head_bytes_bf16;
                            for (int64_t d = 0; d < head_dim; d += 2) {
                                memcpy(q_ptr + angles_off + d * 2, angle_cos_bf16, 2);
                                memcpy(q_ptr + angles_off + d * 2 + 2, angle_sin_bf16, 2);
                            }
                            // Encode actual_seq_len as bf16 at angles[64] (byte offset
                            // angles_off + 128). The kernel reads this to skip empty
                            // KV positions that would dilute the softmax.
                            {
                                float f = (float)actual_seq_len;
                                uint32_t bits;
                                memcpy(&bits, &f, 4);
                                uint16_t bf = (uint16_t)(bits >> 16);
                                if ((bits & 0x7FFFFFFF) < 0x7F800000) {
                                    uint32_t rounding = 0x7FFF + ((bf & 1) ^ 1);
                                    bf = (uint16_t)((bits + rounding) >> 16);
                                }
                                memcpy(q_ptr + angles_off + head_dim * 2, &bf, 2);
                            }
                            fk_entry->bo_q->sync(XCL_BO_SYNC_BO_TO_DEVICE);
                        }

                        // --- Dispatch FlowKV kernel ---
                        // IRON xclbin arg layout: opcode(0), insts(1), insts_size(2),
                        // DDR_buf_0(3)=kv, DDR_buf_1(4)=q, DDR_buf_2(5)=out
                        {
                            // Diagnostic: verify BO data before dispatch
                            if (poc_dbg && kv_h == 0) {
                                auto diag_kv = fk_entry->bo_kv->map<char *>();
                                auto diag_q = fk_entry->bo_q->map<char *>();
                                const uint16_t * kv_bf16 = (const uint16_t *)diag_kv;
                                const uint16_t * q_bf16 = (const uint16_t *)diag_q;
                                fprintf(stderr, "  BO check kv_h=0: K[0] first 8 bf16:");
                                for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", kv_bf16[d]);
                                fprintf(stderr, "\n  BO check kv_h=0: Q first 8 bf16:");
                                for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", q_bf16[d]);
                                fprintf(stderr, "\n  BO check: insts size=%u bo_kv size=%zu bo_q size=%zu bo_out size=%zu\n",
                                        (uint32_t)fk_entry->insts_data.size(),
                                        (size_t)(1 * seq_len * 2 * row_bytes),
                                        (size_t)(1 * (q_heads_per_kv * head_dim + head_dim + 2) * sizeof(uint16_t)),
                                        (size_t)(q_heads_per_kv * row_bytes));
                                fflush(stderr);
                            }

                            // Use set_arg with all 8 kernel args
                            auto run = xrt::run(fk_entry->kernel);
                            run.set_arg(0, 3u);  // opcode
                            run.set_arg(1, fk_entry->insts_bo);  // insts
                            run.set_arg(2, (uint32_t)fk_entry->insts_data.size());  // insts_size
                            run.set_arg(3, *fk_entry->bo_kv);  // DDR buf 0
                            run.set_arg(4, *fk_entry->bo_q);   // DDR buf 1
                            run.set_arg(5, *fk_entry->bo_out); // DDR buf 2
                            run.set_arg(6, 0u);  // unknown arg
                            run.set_arg(7, 0u);  // unknown arg

                            std::lock_guard<std::mutex> lock(*fk_entry->mu);
                            auto t0 = std::chrono::steady_clock::now();
                            run.start();
                            auto state = run.wait(30000);
                            auto t1 = std::chrono::steady_clock::now();

                            if (state != ERT_CMD_STATE_COMPLETED) {
                                GGML_LOG_ERROR("ggml-xdna: [FlowKV-POC] dispatch failed state=%d kv_h=%lld\n",
                                               (int)state, (long long)kv_h);
                                continue;
                            }

                            if (poc_dbg) {
                                float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
                                fprintf(stderr, "ggml-xdna: [FlowKV-POC] kv_h=%lld dispatched %.2f ms\n",
                                        (long long)kv_h, ms);

                                // === DIAG: post-dispatch bo_kv integrity + K_DIAG compare ===
                                if (kv_h == 0) {
                                    fk_entry->bo_kv->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                                    auto post_kv = reinterpret_cast<const uint16_t*>(
                                        fk_entry->bo_kv->map<char *>());
                                    fprintf(stderr, "  [DIAG-POST] bo_kv after dispatch:\n");
                                    fprintf(stderr, "    K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                                        post_kv[0], post_kv[1], post_kv[2], post_kv[3],
                                        post_kv[4], post_kv[5], post_kv[6], post_kv[7]);

                                    // K_DIAG is in last head slot of bo_out
                                    fk_entry->bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                                    auto out_diag = fk_entry->bo_out->map<char *>();
                                    size_t out_hb = head_dim * sizeof(uint16_t);
                                    const uint16_t * kdiag = reinterpret_cast<const uint16_t*>(
                                        out_diag + (q_heads_per_kv - 1) * out_hb);

                                    fprintf(stderr, "  [DIAG-COMPARE] kv_h=0:\n");
                                    fprintf(stderr, "    Host  K[0][0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                                        post_kv[0], post_kv[1], post_kv[2], post_kv[3],
                                        post_kv[4], post_kv[5], post_kv[6], post_kv[7]);
                                    fprintf(stderr, "    K_DIAG    [0:8]: 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X 0x%04X\n",
                                        kdiag[0], kdiag[1], kdiag[2], kdiag[3],
                                        kdiag[4], kdiag[5], kdiag[6], kdiag[7]);

                                    int match_count = 0;
                                    for (int i = 0; i < 8; i++) {
                                        if (kdiag[i] == post_kv[i]) match_count++;
                                    }
                                    fprintf(stderr, "    Match count (first 8): %d/8 → %s\n",
                                        match_count, match_count > 0 ? "PARTIAL MATCH" : "ZERO MATCH — DMA reads wrong data!");
                                    fflush(stderr);
                                }
                                // === END DIAG ===
                            }
                        }

                        // --- Read back and write to kqv_out (bf16→f32 if needed) ---
                        fk_entry->bo_out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                        {
                            auto out_ptr = fk_entry->bo_out->map<char *>();
                            char * kqv_data = (char *)kqv_out->data;
                            size_t out_head_bytes = head_dim * sizeof(uint16_t);
                            for (int64_t g = 0; g < q_heads_per_kv; g++) {
                                int64_t q_head = kv_h * q_heads_per_kv + g;
                                if (out_is_f32) {
                                    bf16_to_f32((const uint16_t *)(out_ptr + g * out_head_bytes),
                                                (float *)(kqv_data + q_head * head_dim * sizeof(float)),
                                                (size_t)head_dim);
                                } else {
                                    memcpy(kqv_data + q_head * out_head_bytes,
                                           out_ptr + g * out_head_bytes, out_head_bytes);
                                }
                            }
                        }
                    } // for kv_h

                    if (poc_dbg) {
                        fprintf(stderr, "ggml-xdna: [FlowKV-POC] completed, overwrote kqv_out @%d\n", i);
                        // Diagnostic: check raw bo_out data after sync
                        {
                            auto raw = fk_entry->bo_out->map<char *>();
                            const uint16_t * raw_bf16 = (const uint16_t *)raw;
                            bool all_zero = true;
                            for (int d = 0; d < q_heads_per_kv * head_dim; d++) {
                                if (raw_bf16[d] != 0) { all_zero = false; break; }
                            }
                            fprintf(stderr, "  bo_out raw: first 8 bf16:");
                            for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", raw_bf16[d]);
                            fprintf(stderr, " all_zero=%d\n", all_zero);
                            // DIAG: print last head's data (K[0] from kernel)
                            int last_head_off = (q_heads_per_kv - 1) * head_dim;
                            fprintf(stderr, "  bo_out K_DIAG [last head, off=%d]:", last_head_off);
                            for (int d = 0; d < 8; d++) fprintf(stderr, " 0x%04X", raw_bf16[last_head_off + d]);
                            fprintf(stderr, "\n");
                        }
                        // Compare NPU vs CPU for first head
                        if (!cpu_save.empty()) {
                            const float * npu_f32 = (const float *)kqv_out->data;
                            fprintf(stderr, "  NPU vs CPU head0 first 8 values:\n");
                            for (int d = 0; d < 8 && d < (int)head_dim; d++) {
                                fprintf(stderr, "  [%d] NPU=%.6f  CPU=%.6f  diff=%.6f\n",
                                        d, npu_f32[d], cpu_save[d], npu_f32[d] - cpu_save[d]);
                            }
                            cpu_save.clear();
                        }
                        fflush(stderr);
                    }
                    // FlowKV dispatched successfully — don't add CONT to
                    // CPU range. The NPU result in kqv_out must not be
                    // overwritten by CPU attention.
                    flowkv_dispatched_this_layer = false;  // consumed
                    continue;
                } catch (const std::exception & e) {
                    GGML_LOG_ERROR("ggml-xdna: [FlowKV-POC] exception: %s\n", e.what());
                }
            } else {
                GGML_LOG_ERROR("ggml-xdna: [FlowKV-POC] kernel load failed\n");
            }
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

    // FlowKV early dispatch: after QKV segment's CPU delegate range,
    // permuted Q/K/V tensors are ready. Dispatch FlowKV now so the
    // next segment (SOFT_MAX) can be skipped entirely.
    // This replaces the POC dispatch at CONT(kqv_out) for layers where
    // the QKV segment precedes the attention segment.
    if (flowkv_decode_enabled && flowkv_poc_valid &&
        flowkv_poc_q_perm && flowkv_poc_q_perm->data) {
        int64_t head_dim = flowkv_poc_head_dim;
        int64_t seq_len = flowkv_poc_seq_len;
        int64_t num_kv_heads = flowkv_poc_num_kv_heads;
        int64_t num_q_heads = flowkv_poc_num_q_heads;
        int64_t q_heads_per_kv = num_q_heads / num_kv_heads;
        int chunk_size = 32;
        int num_cols = 1;

        xdna_flowkv_entry * fk_entry = get_or_load_flowkv_kernel(
            ctx, q_heads_per_kv, /*num_kv_heads=*/1, head_dim, seq_len, chunk_size, num_cols);

        if (fk_entry) {
            static const bool early_dbg = getenv("XDNA_DEBUG") != NULL;
            if (early_dbg) {
                fprintf(stderr, "ggml-xdna: [FlowKV-early] dispatching after QKV segment "
                        "H=%lld KV=%lld d=%lld S=%lld\n",
                        (long long)q_heads_per_kv, (long long)num_kv_heads,
                        (long long)head_dim, (long long)seq_len);
                fflush(stderr);
            }
            // Note: the actual BO setup + dispatch + readback is identical
            // to the POC path. For now, just set the flag — the POC at
            // CONT(kqv_out) will handle the actual dispatch. The flag
            // causes SOFT_MAX to be skipped.
            flowkv_dispatched_this_layer = true;
        }
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
    static const bool gemv_enabled = xdna_env_enabled("XDNA_ENABLE_GEMV");
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
    static const bool rms_norm_enabled = xdna_env_enabled("XDNA_ENABLE_RMS_NORM");
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
            // FlowKV decode attention: Q@K^T and scores@V are batched over
            // heads (ne[2] = n_heads or n_kv_heads) and non-contiguous after
            // ggml_permute(0,2,1,3). We must claim them here so the scheduler
            // assigns them to XDNA; graph_compute's FlowKV matcher handles
            // the actual dispatch decision.
            //
            // DISABLED: claiming K=64 MUL_MATs changes graph segmentation,
            // splitting the attention pattern across XNA/CPU segments. The
            // CPU backend then can't compute the full attention (Q@K^T →
            // softmax → scores@V) as a unit, producing garbage output.
            // Re-enable once FlowKV cross-segment dispatch is proven correct.
            //
            // static const bool flowkv_decode_enabled_for_supports = xdna_env_enabled("XDNA_ENABLE_FLOWKV_DECODE");
            // if (flowkv_decode_enabled_for_supports && src1->ne[1] == 1 && src0->ne[0] == 64 && src0->ne[1] > 1) {
            //     if (src1->type != GGML_TYPE_F32) return false;
            //     if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) return true;
            //     return false;
            // }
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
            static const bool int8_ok = xdna_env_enabled("XDNA_ENABLE_SWIGLU_INT8");
            static const bool tblock_w8a16_ok =
                (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED")) &&
                (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED_W8A16"));
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
    static const bool qkv_enabled                 = xdna_env_enabled("XDNA_ENABLE_QKV");
    static const bool attention_prefill_enabled   = xdna_env_enabled("XDNA_ENABLE_ATTENTION_PREFILL");
    static const bool transformer_block_enabled   =
        (xdna_env_enabled("XDNA_ENABLE_TRANSFORMER_BLOCK")) ||
        (xdna_env_enabled("XDNA_ENABLE_TBLOCK_FUSED"));
    static const bool flowkv_decode_enabled       = xdna_env_enabled("XDNA_ENABLE_FLOWKV_DECODE");
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
                    // and N=seq_len (Q@K^T).
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
                // During decode, keep these on CPU natively.  The FlowKV
                // matcher now handles cross-segment matching, so there is
                // no need to force scheduler grouping via aggressive claims.
                // (Claiming these during decode fragments scheduling with
                // delegate_range round-trips and was causing incorrect
                // output when FlowKV couldn't dispatch.)
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
