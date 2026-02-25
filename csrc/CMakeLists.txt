# NOTES: this CMake is only for debugging; for setup, please use Torch extension
cmake_minimum_required(VERSION 3.10)
project(deep_ep)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

if(DEFINED ENV{USE_ROCM} OR USE_ROCM)
    enable_language(CXX C HIP)

    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation path")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
    endif()

    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
    # list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_BFLOAT16_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF2_OPERATORS__=1)
    list(APPEND HIP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    list(APPEND HIP_CXX_FLAGS -mavx2)
    list(APPEND HIP_CXX_FLAGS -mf16c)
    list(APPEND HIP_CXX_FLAGS -mfma)

    set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})

    find_package(HIP REQUIRED)
    find_package(pybind11 REQUIRED)
    find_package(Torch REQUIRED)
    find_package(ROCSHMEM REQUIRED HINTS /root/rocshmem/lib/cmake/rocshmem)

    set(CMAKE_CXX_STANDARD 17)

    set(ROCSHMEM_INCLUDE_DIR /root/rocshmem/include)
    set(NVSHMEM_LIB_DIR /root/rocshmem/lib)

    include_directories(${ROCM_PATH}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${ROCSHMEM_INCLUDE_DIR})
    link_directories(${TORCH_INSTALL_PREFIX}/lib ${ROCM_PATH}/lib ${NVSHMEM_LIB_DIR})
else()
    enable_language(CXX C CUDA)

    set(CUDA_SEPARABLE_COMPILATION ON)
    list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
    list(APPEND CUDA_NVCC_FLAGS "-O3")
    list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

    set(USE_SYSTEM_NVTX on)
    set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
    set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

    find_package(CUDAToolkit REQUIRED)
    find_package(pybind11 REQUIRED)
    find_package(Torch REQUIRED)
    find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_ROOT_DIR}/lib/cmake/nvshmem)

    add_library(nvshmem ALIAS nvshmem::nvshmem)
    add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
    add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CUDA_STANDARD 17)

    include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NVSHMEM_INCLUDE_DIR})
    link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NVSHMEM_LIB_DIR})
endif()

add_subdirectory(kernels)

# Link CPP and CUDA together
pybind11_add_module(deep_ep_cpp deep_ep.cpp)
target_link_libraries(deep_ep_cpp PRIVATE ${EP_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
