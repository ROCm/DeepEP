#pragma once

#define NUM_MAX_NVL_PEERS 8
#define NUM_MAX_RDMA_PEERS 20

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull  // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull  // 20G cycles ~= 10s
#endif

#ifdef USE_ROCM
#define NUM_WAIT_CYCLES_TIMES_64 16
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900  // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__     // NOLINT(*-reserved-identifier)
#endif

// Define __CUDACC_RDC__ to ensure proper extern declarations for NVSHMEM device symbols
#ifndef DISABLE_NVSHMEM
#ifndef __CUDACC_RDC__
#define __CUDACC_RDC__  // NOLINT(*-reserved-identifier)
#endif
#endif

namespace deep_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif
#define DISABLE_SM90_FEATURES 1
#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;  // int32_t or int64_t
#undef INT_BITS_T
#undef INT_BITS_T2

}  // namespace deep_ep

#ifdef USE_ROCM
static constexpr int32_t kWarpSize = 64;
// For ROCm equals to half the wave size or Nvidia warp size
static constexpr int32_t kEmulatedWarpSize = kWarpSize / 2;
static constexpr uint64_t kFullWarpMask = 0xffffffffffffffff;
static constexpr uint64_t kFirstHalfMask = 0x00000000ffffffff;
static constexpr uint64_t kSecondHalfMask = 0xffffffff00000000;
#else
static constexpr int32_t kWarpSize = 32;
// For Nvidia matches the actual warp size
static constexpr int32_t kEmulatedWarpSize = kWarpSize;
static constexpr uint32_t kFullWarpMask = 0xffffffff;
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

// Remove Torch restrictions for HIP
#ifdef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_OPERATORS__
#endif

#ifndef USE_ROCM
#include <cuda_bf16.h>
#else
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#endif
#include <cuda_runtime.h>

#include <cstdint>

#ifndef USE_ROCM
#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif
#endif

namespace deep_ep {

#ifndef TOPK_IDX_BITS
#define TOPK_IDX_BITS 64
#endif

#define INT_BITS_T2(bits) int##bits##_t
#define INT_BITS_T(bits) INT_BITS_T2(bits)
typedef INT_BITS_T(TOPK_IDX_BITS) topk_idx_t;  // int32_t or int64_t
#undef INT_BITS_T
#undef INT_BITS_T2

}  // namespace deep_ep

#ifndef DISABLE_NVSHMEM
#include <infiniband/mlx5dv.h>
#ifndef USE_ROCM
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#else
#include <rocshmem/rocshmem.hpp>
#include <rocshmem/rocshmem_RMA_X.hpp>
#endif // !USE_ROCM
#endif // !DISABLE_NVSHMEM
