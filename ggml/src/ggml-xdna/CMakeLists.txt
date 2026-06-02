# Check for XRT (Xilinx Runtime) — required for XDNA NPU access
find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    HINTS
        /opt/xilinx/xrt/include
        $ENV{XILINX_XRT}/include
)

find_library(XRT_CORE_LIB
    NAMES xrt_core
    HINTS
        /opt/xilinx/xrt/lib
        $ENV{XILINX_XRT}/lib
)

find_library(XRT_COREUTIL_LIB
    NAMES xrt_coreutil
    HINTS
        /opt/xilinx/xrt/lib
        $ENV{XILINX_XRT}/lib
)

if (XRT_INCLUDE_DIR AND XRT_COREUTIL_LIB)
    message(STATUS "XRT found, Include: ${XRT_INCLUDE_DIR}")
    message(STATUS "XRT found, Libraries: ${XRT_COREUTIL_LIB}")

    # Optional: link AMD aiebu for transaction-binary emission (DDR_PATCH /
    # MERGE_SYNC), the same path FastFlowLM uses. Built locally into
    # third_party/aiebu/install. OFF by default.
    option(GGML_XDNA_USE_AIEBU "Link the aiebu assembler library" OFF)

    # Heavy aiebu includes are isolated in aiebu_wrapper.cpp so they don't
    # pollute ggml-xdna.cpp's compile when aiebu is enabled.
    set(GGML_XDNA_SRCS ggml-xdna.cpp)
    if (GGML_XDNA_USE_AIEBU)
        list(APPEND GGML_XDNA_SRCS aiebu_wrapper.cpp)
    endif()
    ggml_add_backend_library(ggml-xdna ${GGML_XDNA_SRCS})

    target_include_directories(ggml-xdna SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${XRT_COREUTIL_LIB} ggml-base ggml-cpu)

    if (GGML_XDNA_USE_AIEBU)
        set(AIEBU_ROOT "${CMAKE_SOURCE_DIR}/third_party/aiebu/install" CACHE PATH "aiebu install prefix")
        set(AIEBU_SRC  "${CMAKE_SOURCE_DIR}/third_party/aiebu/src/cpp" CACHE PATH "aiebu source tree")
        find_package(AIEBU REQUIRED PATHS "${AIEBU_ROOT}/share/cmake/AIEBU" NO_DEFAULT_PATH)
        # CRITICAL: aiebu paths must come BEFORE XRT_INCLUDE_DIR. The XRT
        # Windows SDK ships an outdated copy of aiebu/aiebu_assembler.h
        # (119 lines, no file_artifact); compiling against it produces
        # confusing "file_artifact: not a member of aiebu" errors.
        target_include_directories(ggml-xdna SYSTEM BEFORE PRIVATE
            "${AIEBU_ROOT}/include"
            "${AIEBU_SRC}/include"
            # FFLM bypasses the public aiebu_assembler wrapper (which crashes
            # on static-init in a DLL) and calls aiebu::assembler::process
            # directly. Subdirs added since internal headers use unqualified
            # peer includes.
            "${AIEBU_SRC}/assembler"
            "${AIEBU_SRC}/common"
            "${AIEBU_SRC}/encoder"
            "${AIEBU_SRC}/preprocessor"
            "${AIEBU_SRC}/elf"
            "${AIEBU_SRC}/analyzer"
            "${AIEBU_SRC}"             # "common/regex_wrapper.h", "json/..."
            "${AIEBU_SRC}/ELFIO"
            "${CMAKE_SOURCE_DIR}/third_party/boost/boost_1_88_0"
        )
        target_link_libraries(ggml-xdna PRIVATE "${AIEBU_ROOT}/lib/aiebu_static.lib")
        target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_USE_AIEBU)
        message(STATUS "ggml-xdna: aiebu enabled at ${AIEBU_ROOT}")
    endif()
    
    if (MSVC)
        target_compile_options(ggml-xdna PRIVATE /std:c++17 /Zc:__cplusplus)
    else()
        target_compile_features(ggml-xdna PRIVATE cxx_std_17)
    endif()

    # Phase 9 Day 1 spike: standalone XRT async-dispatch benchmark.
    # Not part of the backend; links against XRT directly.
    add_executable(xrt_async_spike tools/xrt_async_spike.cpp)
    target_include_directories(xrt_async_spike SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(xrt_async_spike PRIVATE ${XRT_COREUTIL_LIB})
    if (MSVC)
        target_compile_options(xrt_async_spike PRIVATE /std:c++17 /Zc:__cplusplus)
    else()
        target_compile_features(xrt_async_spike PRIVATE cxx_std_17)
    endif()

    # Generic MLIR_AIE xclbin replay/dispatch harness (standalone).
    add_executable(xclbin_replay tools/xclbin_replay.cpp)
    target_include_directories(xclbin_replay SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(xclbin_replay PRIVATE ${XRT_COREUTIL_LIB})
    if (MSVC)
        target_compile_options(xclbin_replay PRIVATE /std:c++17 /Zc:__cplusplus)
    else()
        target_compile_features(xclbin_replay PRIVATE cxx_std_17)
    endif()
else()
    message(FATAL_ERROR "XRT not found. Install AMD XRT and source /opt/xilinx/xrt/setup.sh\n"
                        "  XRT_INCLUDE_DIR: ${XRT_INCLUDE_DIR}\n"
                        "  XRT_COREUTIL_LIB: ${XRT_COREUTIL_LIB}")
endif()
