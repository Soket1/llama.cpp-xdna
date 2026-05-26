// SPDX-License-Identifier: MIT
// Isolated TU that pulls in all aiebu/boost/ELFIO internals so the rest of
// ggml-xdna doesn't have to. Exposes a C API declared in aiebu_wrapper.h.
//
// Why bypass the public aiebu_assembler wrapper:
//   - Statically linking aiebu_static.lib into ggml-xdna.dll and calling
//     aiebu::aiebu_assembler::aiebu_assembler(...) crashes (ACCESS_VIOLATION)
//     at runtime on MSVC, even with /MD CRT match. The lower-level
//     aiebu::assembler::process() does not crash. Verified that FFLM's
//     llama_npu.dll links aiebu_static.lib and uses the same lower-level
//     class (no aiebu_assembler symbols in their DLL).

#include "aiebu_wrapper.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "aiebu/aiebu_assembler.h"

// Full definitions of the unique_ptr members in assembler — required for
// the inline ~assembler() = default destructor to compile externally.
#include "elfwriter.h"           // class elf_writer
#include "encoder.h"             // class encoder
#include "preprocessor.h"        // class preprocessor
#include "preprocessor_input.h"  // class preprocessor_input

#include "assembler.h"           // class aiebu::assembler

extern "C" int aiebu_assemble_transaction(const void * insts, size_t insts_size,
                                          void ** out_elf, size_t * out_size) {
    if (!insts || !out_elf || !out_size) return 1;
    *out_elf = nullptr;
    *out_size = 0;
    try {
        const char * src = static_cast<const char *>(insts);
        std::vector<char> in(src, src + insts_size);
        aiebu::assembler a(aiebu::assembler::elf_type::aie2_transaction_blob);
        std::vector<char> elf = a.process(in);
        const size_t n = elf.size();
        void * buf = std::malloc(n);
        if (!buf) return 2;
        std::memcpy(buf, elf.data(), n);
        *out_elf = buf;
        *out_size = n;
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr,
                     "aiebu_wrapper: assemble_transaction threw: %s\n",
                     e.what());
        return 3;
    } catch (...) {
        std::fprintf(stderr,
                     "aiebu_wrapper: assemble_transaction threw unknown\n");
        return 4;
    }
}

extern "C" void aiebu_free_buffer(void * buf) {
    if (buf) std::free(buf);
}
