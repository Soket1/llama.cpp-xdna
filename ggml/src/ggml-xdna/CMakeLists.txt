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

    ggml_add_backend_library(ggml-xdna
                             ggml-xdna.cpp
                            )

    target_include_directories(ggml-xdna SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${XRT_COREUTIL_LIB} ggml-base ggml-cpu)
    
    if (MSVC)
        target_compile_options(ggml-xdna PRIVATE /std:c++17 /Zc:__cplusplus)
    else()
        target_compile_features(ggml-xdna PRIVATE cxx_std_17)
    endif()
else()
    message(FATAL_ERROR "XRT not found. Install AMD XRT and source /opt/xilinx/xrt/setup.sh\n"
                        "  XRT_INCLUDE_DIR: ${XRT_INCLUDE_DIR}\n"
                        "  XRT_COREUTIL_LIB: ${XRT_COREUTIL_LIB}")
endif()
