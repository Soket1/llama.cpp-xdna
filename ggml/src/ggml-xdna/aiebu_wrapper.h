// SPDX-License-Identifier: MIT
// Thin C API around aiebu's low-level aiebu::assembler::process. All heavy
// aiebu/boost/ELFIO includes are isolated in aiebu_wrapper.cpp so they don't
// pollute ggml-xdna.cpp's compile.
//
// The public aiebu_assembler wrapper crashes on static-init when statically
// linked into a DLL on MSVC; we bypass it. FFLM uses the same lower-level
// path (verified via symbol scan of llama_npu.dll: aiebu_assembler absent,
// aiebu::assembler::process present).
#ifndef GGML_XDNA_AIEBU_WRAPPER_H_
#define GGML_XDNA_AIEBU_WRAPPER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Assemble an IRON transaction-binary buffer (e.g. post_attn_fused_main.insts)
// into a complete XRT-loadable ELF using buffer_type::aie2_transaction_blob
// (equivalent to aiebu-asm --target aie2txn).
//
// On success: *out_elf points to a buffer allocated by the wrapper; caller
// MUST release it via aiebu_free_buffer. *out_size receives the byte count.
// Returns 0 on success, nonzero on failure (error logged to stderr).
int aiebu_assemble_transaction(const void * insts, size_t insts_size,
                               void ** out_elf, size_t * out_size);

void aiebu_free_buffer(void * buf);

#ifdef __cplusplus
}
#endif

#endif // GGML_XDNA_AIEBU_WRAPPER_H_
