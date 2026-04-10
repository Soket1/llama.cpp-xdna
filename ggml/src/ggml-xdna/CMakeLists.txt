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

if (XRT_INCLUDE_DIR AND XRT_CORE_LIB AND XRT_COREUTIL_LIB)
    message(STATUS "XRT found, Include: ${XRT_INCLUDE_DIR}")
    message(STATUS "XRT found, Libraries: ${XRT_CORE_LIB} ${XRT_COREUTIL_LIB}")

    ggml_add_backend_library(ggml-xdna
                             ggml-xdna.cpp
                            )

    target_include_directories(ggml-xdna SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${XRT_CORE_LIB} ${XRT_COREUTIL_LIB})
else()
    message(FATAL_ERROR "XRT not found. Install AMD XRT and source /opt/xilinx/xrt/setup.sh\n"
                        "  XRT_INCLUDE_DIR: ${XRT_INCLUDE_DIR}\n"
                        "  XRT_CORE_LIB: ${XRT_CORE_LIB}\n"
                        "  XRT_COREUTIL_LIB: ${XRT_COREUTIL_LIB}")
endif()
