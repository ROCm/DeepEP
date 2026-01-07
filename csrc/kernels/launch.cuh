#pragma once

#include "configs.cuh"

#if defined(USE_ROCM)
#define GPU_R_16BF HIP_R_16BF
#define GPU_R_32F HIP_R_32F
using gpu_bfloat16_t = hip_bfloat16;
#else
#define GPU_R_16BF CUDA_R_16BF
#define GPU_R_32F CUDA_R_32F
using gpu_bfloat16_t = nv_bfloat16;
#endif

// ROCm helper functions and structures
#if defined(USE_ROCM)
namespace rocm::experimental {
typedef struct {
  dim3 num_sms;
  dim3 num_threads;
  unsigned int shared_mem_bytes;
  hipStream_t stream;
} hipLaunchConfig_t;

// Compile time void** kernelArgs array fill with variadic arguments
template <typename T> void fill_kernel_args(void **f, size_t idx, T &&arg) {
  f[idx] = static_cast<void *>(std::addressof(arg));
}

template <typename Head, typename... Tail>
void fill_kernel_args(void **f, size_t idx, Head &&head, Tail &&...tail) {
  f[idx] = static_cast<void *>(std::addressof(head));
  fill_kernel_args(f, idx + 1, std::forward<Tail>(tail)...);
}
} // namespace rocm::experimental

#endif

#ifndef SETUP_LAUNCH_CONFIG
#if defined(USE_ROCM)
// The code below is a workaround for ROCm. All the proposed overhead
// is to match current macro signatures and should be reworked once
// cudaLaunchKernelExt() hip alternative is live.
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                      \
  rocm::experimental::hipLaunchConfig_t cfg = {(num_sms), (num_threads), 0,    \
                                               stream};

#else
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                      \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0};  \
  cudaLaunchAttribute attr[1];                                                 \
  attr[0].id = cudaLaunchAttributeCooperative;                                 \
  attr[0].val.cooperative = 1;                                                 \
  cfg.attrs = attr;                                                            \
    cfg.numAttrs = 1
#endif // #if defined(USE_ROCM)
#endif // #ifndef SETUP_LAUNCH_CONFIG

#ifndef LAUNCH_KERNEL
#if defined(USE_ROCM)
template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL(T &&config, Kern &&kernel, Args &&...args) {
  constexpr size_t k_num_kernel_args = sizeof...(args);
  void *kernel_args[k_num_kernel_args];
  rocm::experimental::fill_kernel_args(kernel_args, 0,
                                       std::forward<Args>(args)...);
  CUDA_CHECK(hipLaunchCooperativeKernel(
      std::forward<Kern>(kernel), config->num_sms, config->num_threads,
      kernel_args, config->shared_mem_bytes, config->stream));
}

template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL_NON_COOPERATIVE(T &&config, Kern &&kernel,
                                          Args &&...args) {
  *kernel<<<config->num_sms, config->num_threads, config->shared_mem_bytes,
            config->stream>>>(std::forward<Args>(args)...);
}

#else
#define LAUNCH_KERNEL(config, kernel, ...)                                     \
  CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif // #if defined(USE_ROCM)
#endif // #ifndef LAUNCH_KERNEL


#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                \
    EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
    cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif


#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
     } while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false && "Unsupported ranks"); \
    } while (false)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case GPU_R_16BF: case_macro(gpu_bfloat16_t); \
        case GPU_R_32F:  case_macro(float); \
        default: EP_HOST_ASSERT(false && "Unsupported type"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2560: case_macro(2560); \
        case 5120: case_macro(5120); \
        case 4096: case_macro(4096); \
        case 7168: case_macro(7168); \
        default: EP_HOST_ASSERT(false && "Unsupported hidden"); \
    } while (false)
