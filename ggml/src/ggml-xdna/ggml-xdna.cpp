#include "ggml-impl.h"
#include "ggml-xdna.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

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

// ============================================================================
// Cached kernel entry — one per unique (op, shape, dtype) tuple
// ============================================================================

enum xdna_op_kind : int {
    XDNA_OP_GEMM = 0,  // M>=32 prefill MUL_MAT via IRON GEMM
    XDNA_OP_GEMV = 1,  // M==1  decode  MUL_MAT via IRON GEMV
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
// Backend context — holds XRT device and kernel cache
// ============================================================================

struct ggml_backend_xdna_context {
    xrt::device device;
    std::string cache_dir;
    std::string compile_script;
    std::unordered_map<std::string, xdna_kernel_entry> kernel_cache;
    std::mutex cache_mutex;
    bool device_valid;
    // CPU backend for delegating ops we don't run on NPU.
    // Our buffers are plain host RAM so CPU can compute on them directly.
    ggml_backend_t cpu_backend;

    ggml_backend_xdna_context() : device_valid(false), cpu_backend(nullptr) {
        // Cache directory
        const char * cache_env = getenv("GGML_XDNA_CACHE_DIR");
        const char * home = getenv("HOME");
        if (cache_env) {
            cache_dir = cache_env;
        } else if (home) {
            cache_dir = std::string(home) + "/.cache/ggml-xdna/xclbin";
        } else {
            cache_dir = "/tmp/ggml-xdna/xclbin";
        }

        // Find compile.py relative to this shared library's location
        // Fallback: look for GGML_XDNA_COMPILE_SCRIPT env var
        const char * script_env = getenv("GGML_XDNA_COMPILE_SCRIPT");
        if (script_env) {
            compile_script = script_env;
        } else {
            compile_script = "compile.py";  // must be in PATH or CWD
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
        snprintf(buf, sizeof(buf), "gemv_K%ld_N%ld_%s_%dcol",
                 (long)K, (long)N, dtype_in, num_cols);
    } else {
        snprintf(buf, sizeof(buf), "gemm_%ldx%ldx%ld_%s_%dcol",
                 (long)M, (long)K, (long)N, dtype_in, num_cols);
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

// ============================================================================
// Compilation — call compile.py to generate xclbin + insts
// ============================================================================

static bool ensure_compiled(ggml_backend_xdna_context * ctx,
                            const std::string & cache_key,
                            xdna_op_kind op_kind,
                            int64_t M, int64_t K, int64_t N,
                            const char * dtype_in, int num_cols) {
    std::string xclbin_path = ctx->cache_dir + "/" + cache_key + ".xclbin";
    std::string insts_path  = ctx->cache_dir + "/" + cache_key + ".insts";

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
                 "python3 %s gemv --N %ld --K %ld --dtype-in %s --dtype-out %s "
                 "--num-aie-columns %d --out %s 2>&1",
                 ctx->compile_script.c_str(),
                 (long)N, (long)K,
                 dtype_in, dtype_in,
                 num_cols,
                 xclbin_path.c_str());
        GGML_LOG_INFO("ggml-xdna: compiling GEMV K=%ld N=%ld (first run, will be cached)...\n",
                      (long)K, (long)N);
    } else {
        snprintf(cmd, sizeof(cmd),
                 "python3 %s gemm --M %ld --K %ld --N %ld --dtype-in %s --dtype-out %s "
                 "--num-aie-columns %d --out %s 2>&1",
                 ctx->compile_script.c_str(),
                 (long)M, (long)K, (long)N,
                 dtype_in, dtype_in,
                 num_cols,
                 xclbin_path.c_str());
        GGML_LOG_INFO("ggml-xdna: compiling GEMM %ldx%ldx%ld (first run, will be cached)...\n",
                      (long)M, (long)K, (long)N);
    }

    int ret = system(cmd);
    if (ret != 0) {
        GGML_LOG_ERROR("ggml-xdna: compilation failed (exit code %d)\n", ret);
        return false;
    }

    // Verify files were created
    std::ifstream xf(xclbin_path);
    std::ifstream inf(insts_path);
    if (!xf.good() || !inf.good()) {
        GGML_LOG_ERROR("ggml-xdna: compilation succeeded but output files missing\n");
        return false;
    }

    GGML_LOG_INFO("ggml-xdna: compilation complete, cached at %s\n", xclbin_path.c_str());
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

    // Load from disk
    std::string xclbin_path = ctx->cache_dir + "/" + cache_key + ".xclbin";
    std::string insts_path  = ctx->cache_dir + "/" + cache_key + ".insts";

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
        GGML_LOG_INFO("ggml-xdna: loaded kernel for %s\n", cache_key.c_str());
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

    int num_cols = 4;
    std::string cache_key = make_cache_key(XDNA_OP_GEMM, M, K, N, "bf16", num_cols);

    if (!ensure_compiled(ctx, cache_key, XDNA_OP_GEMM, M, K, N, "bf16", num_cols)) {
        GGML_LOG_ERROR("ggml-xdna: compile failed for %ldx%ldx%ld\n", (long)M, (long)K, (long)N);
        return;
    }

    xdna_kernel_entry * entry = get_or_load_kernel(ctx, cache_key, XDNA_OP_GEMM, M, K, N);
    if (!entry) {
        return;
    }

    try {
        size_t a_elems = M * K;
        size_t b_elems = N * K;
        size_t c_elems = M * N;
        size_t a_bytes = a_elems * sizeof(uint16_t);
        size_t b_bytes = b_elems * sizeof(uint16_t);
        size_t c_bytes = c_elems * sizeof(uint16_t);

        // Lazily allocate persistent A and C BOs on first dispatch for this kernel.
        if (!entry->a_bo) {
            entry->a_bo = std::make_unique<xrt::bo>(ctx->device, a_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(3));
        }
        if (!entry->c_bo) {
            entry->c_bo = std::make_unique<xrt::bo>(ctx->device, c_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(5));
        }

        if (src1->type == GGML_TYPE_F32) {
            f32_to_bf16((const float *)src1->data, (uint16_t *)entry->a_bo->map<void*>(), a_elems);
        } else {
            memcpy(entry->a_bo->map<void*>(), src1->data, a_bytes);
        }
        entry->a_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Get or build the cached transposed B (weight). Weight data is immutable
        // after model load, so the (transpose + bf16 convert + DMA sync) happens once
        // per (kernel, weight_ptr) pair.
        xrt::bo * b_bo_ptr = nullptr;
        {
            std::lock_guard<std::mutex> lock(*entry->b_bo_mutex);
            auto it = entry->b_bo_cache.find(src0->data);
            if (it == entry->b_bo_cache.end()) {
                xrt::bo new_b(ctx->device, b_bytes, xrt::bo::flags::host_only, entry->kernel.group_id(4));
                if (src0->type == GGML_TYPE_F32) {
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
                fprintf(stderr, "ggml-xdna: warm b_bo K=%ld N=%ld weight=%s (%zu cached for this kernel)\n",
                    (long)K, (long)N, src0->name, entry->b_bo_cache.size());
                fflush(stderr);
            } else {
                b_bo_ptr = &it->second;
            }
        }

        auto run = entry->kernel(3, entry->insts_bo, (uint32_t)entry->insts.size(),
                                  *entry->a_bo, *b_bo_ptr, *entry->c_bo);
        run.wait();

        entry->c_bo->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bf16_to_f32((const uint16_t *)entry->c_bo->map<void*>(), (float *)dst->data, c_elems);
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

    int num_cols = 4;
    std::string cache_key = make_cache_key(XDNA_OP_GEMV, 1, K, N, "bf16", num_cols);

    if (!ensure_compiled(ctx, cache_key, XDNA_OP_GEMV, 1, K, N, "bf16", num_cols)) {
        GGML_LOG_ERROR("ggml-xdna: GEMV compile failed for K=%ld N=%ld\n", (long)K, (long)N);
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
                fprintf(stderr, "ggml-xdna: warm gemv matrix K=%ld N=%ld weight=%s (%zu cached)\n",
                        (long)K, (long)N, src0->name, entry->b_bo_cache.size());
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
    struct ggml_cgraph sub = ggml_graph_view(cgraph, i0, i1);
    return ggml_backend_graph_compute(ctx->cpu_backend, &sub);
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *)backend->context;
    int n = cgraph->n_nodes;

    static const bool debug = getenv("XDNA_DEBUG") != NULL;
    if (debug) {
        int n_mulmat = 0, n_mulmat_disp = 0;
        for (int i = 0; i < n; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT) {
                n_mulmat++;
                if (xdna_node_npu_dispatchable(node)) n_mulmat_disp++;
            }
        }
        fprintf(stderr, "ggml-xdna: graph_compute n_nodes=%d mul_mat=%d npu_dispatchable=%d\n",
                n, n_mulmat, n_mulmat_disp);
        fflush(stderr);
    }

    // Sweep the graph. Walk nodes in order: collect runs of consecutive
    // CPU-bound nodes and delegate them as one cgraph view; break out to
    // dispatch each NPU-capable MUL_MAT individually. View ops are skipped
    // (included in the CPU run — CPU handles them as no-ops but keeping them
    // in-range preserves sched invariants).
    int cpu_run_start = -1;
    for (int i = 0; i < n; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if (xdna_node_npu_dispatchable(node)) {
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

    // Flush trailing CPU run.
    if (cpu_run_start >= 0) {
        ggml_status s = xdna_delegate_range(ctx, cgraph, cpu_run_start, n);
        if (s != GGML_STATUS_SUCCESS) return s;
    }

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
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    size_t ram = (pages > 0 && page_size > 0) ? (size_t)pages * (size_t)page_size : (8ULL << 30);
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
    // NPU shares host RAM with CPU — our buffers ARE plain host memory.
    // But we advertise as a device (not host) so the scheduler routes compute
    // work to us instead of treating us like CPU-owned storage.
    return false;
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
// Kernel invariant (aie_kernels/aie2p/mm.cc): 4x8x8 bf16_f32 MAC has r=4, so
// tile_m % (2*r) == 0 → tile_m >= 8 and divisible by 8. Minimum M = tile_m * 4 = 32.
// Cap N at 32768 to skip vocab projection (248320) — aiecc BD overflow at
// N-tiles per shim >~485 (248320 = 2^9*5*97, can't tile friendlier without
// N-chunking in the backend).
static bool xdna_shape_dispatchable(int64_t M, int64_t K, int64_t N) {
    if (M < 32 || K < 32 || N < 32 || N > 32768) return false;
    const int num_cols = 4;
    const int64_t tiles_m[] = {64, 32, 16, 8};
    const int64_t tiles_k[] = {64, 32, 16, 8};
    const int64_t tiles_n[] = {64, 32, 16, 8};
    bool m_ok = false, k_ok = false, n_ok = false;
    for (int i = 0; i < 4 && !m_ok; i++) if (M % (tiles_m[i] * 4)        == 0) m_ok = true;
    for (int i = 0; i < 4 && !k_ok; i++) if (K %  tiles_k[i]             == 0) k_ok = true;
    for (int i = 0; i < 4 && !n_ok; i++) if (N % (tiles_n[i] * num_cols) == 0) n_ok = true;
    return m_ok && k_ok && n_ok;
}

// GEMV (M=1 decode) dispatchability. IRON GEMV constraints:
//   K % kernel_vector_size == 0  (default 64)
//   N % num_cols == 0, per_col = N/num_cols >= 8, (per_col % tile_out) == 0 for
//   some tile_out <= per_col. With the candidate set in compile.py
//   select_gemv_tiles, any per_col >= 8 with per_col a power-of-two multiple works.
// Same vocab-proj N cap applies (BD-overflow territory).
static bool xdna_shape_dispatchable_gemv(int64_t K, int64_t N) {
    // GEMV is correct but currently slower than CPU for M=1 decode because
    // per-call XRT submit/wait overhead dominates the tiny GEMV compute
    // (2.7x regression on Qwen3.5-0.8B). Off by default; opt in via
    // XDNA_ENABLE_GEMV=1 once fused multi-op dispatch lands.
    static const bool gemv_enabled = getenv("XDNA_ENABLE_GEMV") != NULL;
    if (!gemv_enabled) return false;
    const int num_cols = 4;
    if (K < 64 || K % 64 != 0) return false;
    if (N < num_cols || N % num_cols != 0) return false;
    const int64_t per_col = N / num_cols;
    if (per_col < 8) return false;
    if (N > 32768) return false;  // BD-overflow cap (same as GEMM)
    return true;
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
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
            if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_BF16) return false;
            return true;
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
            return true;

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
