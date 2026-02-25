#pragma once

#include "exception.cuh"

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                                                     \
    {                                                                                                                                 \
        constexpr int kLoopStride = kWarpSize * (UNROLL_FACTOR);                                                                             \
        typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)];                          \
        auto __src = (SRC);                                                                                                           \
        auto __dst = (DST);                                                                                                           \
        for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) {                                      \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize ); \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) ST_FUNC(__dst + __i + __j * kWarpSize , unrolled_values[__j]);  \
        }                                                                                                                             \
        {                                                                                                                             \
            int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID);                                                                  \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * kWarpSize < (N)) {                                                                                           \
                    unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize);                                                           \
                }                                                                                                                     \
            }                                                                                                                         \
            _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) {                                                       \
                if (__i + __j * kWarpSize < (N)) {                                                                                           \
                    ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]);                                                            \
                }                                                                                                                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }
#define UNROLLED_WARP_COPY_EMULATED(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = kEmulatedWarpSize * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * kEmulatedWarpSize); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * kEmulatedWarpSize, unrolled_values[__j]); \
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += kEmulatedWarpSize) \
        ST_FUNC(__dst + __i, LD_FUNC(__src + __i)); \
}
// HELPER FUNCTIONS #####################################################################################
#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

template <typename T>
DEVICE_INLINE T shfl_xor(
    const T val,
    int laneMask,
    int width = kWarpSize,
    uint64_t shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) 
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(shfl_sync_mask, val, laneMask, width);
#endif
}

DEVICE_INLINE int shfl_sync(
    const int val,
    int srcLane = 0,
    int width = kWarpSize,
    uint64_t shfl_sync_mask = kFullWarpMask) {  // Let compiler deduce type
#if defined(USE_ROCM)
  return __shfl(val, srcLane, width);
#else
  return __shfl_sync(shfl_sync_mask, val, srcLane, width);
#endif
}

#ifdef USE_ROCM
DEVICE_INLINE int __any_sync(uint64_t mask, int predicate) {
  uint64_t predicate_bit_pattern = __ballot(predicate);
  return (predicate_bit_pattern & mask) > 0;
}

DEVICE_INLINE int __all_sync(uint64_t mask, int predicate) {
    uint64_t predicate_bit_pattern = __ballot(predicate);
    return (~predicate_bit_pattern & mask) == 0;
}
#endif

DEVICE_INLINE void syncwarp() {
#ifdef USE_ROCM
__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
__builtin_amdgcn_wave_barrier();
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

//NOTE: This method will be tested for performance
//   // Performance - replace a block level __syncthreads with per CU
//   // __threadfence_block. It is a fine replacement for __syncwarp on AMD GPUs,
//   // it is because a. memory fencing: __threadfence_block ops. at CU level,
//   // same as __syncwarp at SM b. threads re-converge: wavefront run in
//   // lockstep, no need __syncwarp re-converge
//   __threadfence_block();
#else
  __syncwarp();
#endif
}
// ######################################################################################################

namespace deep_ep {

template <int kBytes>
struct VecInt {};
template <>
struct VecInt<1> {
    using vec_t = int8_t;
};
template <>
struct VecInt<2> {
    using vec_t = int16_t;
};
template <>
struct VecInt<4> {
    using vec_t = int;
};
template <>
struct VecInt<8> {
    using vec_t = int64_t;
};
template <>
struct VecInt<16> {
    using vec_t = int4;
};

template <typename FuncT>
struct PatternVisitor {
    FuncT func;

    __device__ __host__ explicit PatternVisitor(FuncT&& func) : func(std::forward<FuncT>(func)) {}

    __device__ __host__ auto operator[](const uint32_t& i) { return func(i); }
};

__device__ __forceinline__ void trap() {
#ifdef USE_ROCM
    abort();
#else
    asm("trap;");
#endif
}

__device__ __forceinline__ void memory_fence() {
#ifdef USE_ROCM
    __threadfence_system();
#else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_gpu() {
#ifdef USE_ROCM
    __threadfence();
#else
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

__device__ __forceinline__ void memory_fence_cta() {
#ifdef USE_ROCM
    __threadfence_block();
#else
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(const int* ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ void st_release_sys_global(const int* ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ void st_release_cta(const int* ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
    asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int ret;
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    uint64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}
__device__ __forceinline__ int64_t ld_acquire_sys_global(const int64_t* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int64_t ret;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ int ld_acquire_global(const int* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else
    int ret;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
#ifdef USE_ROCM
    return __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int ret;
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
#endif
}

__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
#ifdef USE_ROCM
    return __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
#endif
}

__device__ __forceinline__ int ld_acquire_cta(const int* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
    int ret;
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
#endif
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ int ld_volatile_global(const int* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(reinterpret_cast<const volatile int*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int ret;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ float ld_volatile_global(const float* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(reinterpret_cast<const volatile float*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    float ret;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(reinterpret_cast<const volatile int64_t*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int64_t ret;
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t* ptr) {
#ifdef USE_ROCM
    return __hip_atomic_load(reinterpret_cast<const volatile uint64_t*>(ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    int64_t ret;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t* ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    uint16_t ret;
    // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return static_cast<uint8_t>(ret);
#endif
}

template <>
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    int ret;
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#endif
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    int64_t ret;
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#endif
}

template <>
__device__ __forceinline__ float ld_nc_global(const float* ptr) {
#ifdef USE_ROCM
    return __builtin_nontemporal_load(ptr);
#else
    float ret;
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
#endif
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2* ptr) {
#ifdef USE_ROCM
    int2 ret;
    ret.x = __builtin_nontemporal_load(&(ptr->x));
    ret.y = __builtin_nontemporal_load(&(ptr->y));
    return ret;
#else
    int2 ret;
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
    return ret;
#endif
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
#ifdef USE_ROCM
    int4 ret;
    ret.x = __builtin_nontemporal_load(&(ptr->x));
    ret.y = __builtin_nontemporal_load(&(ptr->y));
    ret.z = __builtin_nontemporal_load(&(ptr->z));
    ret.w = __builtin_nontemporal_load(&(ptr->w));
    return ret;
#else
    int4 ret;
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t* ptr, uint8_t val) {
#ifdef USE_ROCM
    uint8_t* non_const_ptr = const_cast<uint8_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t* ptr, uint16_t val) {
#ifdef USE_ROCM
    uint16_t* non_const_ptr = const_cast<uint16_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t* ptr, uint32_t val) {
#ifdef USE_ROCM
    uint32_t* non_const_ptr = const_cast<uint32_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int* ptr, int val) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int4* ptr, int4 val) {
#ifdef USE_ROCM
    int4* non_const_ptr = const_cast<int4*>(ptr);
    non_const_ptr->x = val.x;
    non_const_ptr->y = val.y;
    non_const_ptr->z = val.z;
    non_const_ptr->w = val.w;
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

__device__ __forceinline__ void st_na_release(const int* ptr, int val) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint32_t* ptr, uint32_t val) {
#ifdef USE_ROCM
    uint32_t* non_const_ptr = const_cast<uint32_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint64_t* ptr, uint64_t val) {
#ifdef USE_ROCM
    uint64_t* non_const_ptr = const_cast<uint64_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const int64_t* ptr, int64_t val) {
#ifdef USE_ROCM
    int64_t* non_const_ptr = const_cast<int64_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#endif
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t* ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int* ptr, const int& value) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t* ptr, const int64_t& value) {
#ifdef USE_ROCM
    int64_t* non_const_ptr = const_cast<int64_t*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
#ifdef USE_ROCM
    float* non_const_ptr = const_cast<float*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
#endif
}

template <>
__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
#ifdef USE_ROCM
    int4* non_const_ptr = const_cast<int4*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#endif
}

__device__ __forceinline__ float log2f_approx(const float& x) {
#ifdef USE_ROCM
    return __log2f(x);
#else
    float ret;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
#endif
}

__device__ __forceinline__ float exp2f_approx(const float& x) {
#ifdef USE_ROCM
    return __builtin_amdgcn_exp2f(x);
#else
    float ret;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
#endif
}

__forceinline__ __device__ int get_lane_id() {
#ifdef USE_ROCM
    return threadIdx.x % warpSize;
#else
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
#endif
}

__device__ __forceinline__ uint32_t elect_one_sync() {

#ifdef USE_ROCM
    return get_lane_id() == 0;
#else
#ifndef DISABLE_SM90_FEATURES
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    return get_lane_id() == 0;
#endif
#endif
}

// TMA PTX instructions
#ifndef DISABLE_SM90_FEATURES

__device__ __forceinline__ void fence_barrier_init() {
#ifdef USE_ROCM
    // no-op (NVIDIA-only)
#else
    asm volatile("fence.mbarrier_init.release.cluster; \n" ::);
#endif
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar_ptr, uint32_t arrive_count) {
#ifdef USE_ROCM
    (void)mbar_ptr;
    (void)arrive_count;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::"r"(arrive_count), "r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void mbarrier_inval(uint64_t* mbar_ptr) {
#ifdef USE_ROCM
    (void)mbar_ptr;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mbar_int_ptr));
#endif
}

template <bool kWithMultiStages = false>
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar_ptr, uint32_t& phase, int stage_idx = 0) {
#ifdef USE_ROCM
    (void)mbar_ptr;
    (void)stage_idx;
    __syncthreads();
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    const auto& wait = kWithMultiStages ? (phase >> stage_idx) & 1 : phase;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::"r"(mbar_int_ptr),
        "r"(wait),
        "r"(0x989680));
    phase ^= kWithMultiStages ? (1 << stage_idx) : 1;
#endif
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx(uint64_t* mbar_ptr, int num_bytes) {
#ifdef USE_ROCM
    (void)mbar_ptr;
    (void)num_bytes;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::"r"(num_bytes), "r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar_ptr) {
#ifdef USE_ROCM
    (void)mbar_ptr;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0]; \n\t" ::"r"(mbar_int_ptr));
#endif
}

__device__ __forceinline__ void tma_store_fence() {
#ifdef USE_ROCM
    // no-op (NVIDIA-only)
#else
    asm volatile("fence.proxy.async.shared::cta;");
#endif
}

constexpr uint64_t kEvictFirst = 0x12f0000000000000;
constexpr uint64_t kEvictNormal = 0x1000000000000000;

__device__ __forceinline__ void tma_load_1d(
    const void* smem_ptr, const void* gmem_ptr, uint64_t* mbar_ptr, int num_bytes, bool evict_first = true) {
#ifdef USE_ROCM
    (void)smem_ptr;
    (void)gmem_ptr;
    (void)mbar_ptr;
    (void)num_bytes;
    (void)evict_first;
#else
    auto mbar_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar_ptr));
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr),
        "r"(num_bytes),
        "r"(mbar_int_ptr),
        "l"(cache_hint)
        : "memory");
#endif
}

__device__ __forceinline__ void tma_store_1d(const void* smem_ptr, const void* gmem_ptr, int num_bytes, bool evict_first = true) {
#ifdef USE_ROCM
    (void)smem_ptr;
    (void)gmem_ptr;
    (void)num_bytes;
    (void)evict_first;
#else
    auto smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    const auto cache_hint = evict_first ? kEvictFirst : kEvictNormal;
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(gmem_ptr),
                 "r"(smem_int_ptr),
                 "r"(num_bytes),
                 "l"(cache_hint)
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;");
#endif
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
#ifdef USE_ROCM
    (void)N;
#else
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
#endif
}

#endif

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_up(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align_down(dtype_t a, dtype_t b) {
    return a / b * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = ceil_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++i)
#ifdef USE_ROCM
        recv_int_values[i] = __shfl(send_int_values[i], src_lane_idx, 32);
#else
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], src_lane_idx);
#endif
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

#ifdef USE_ROCM 
constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 240.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 240.0f;
#else
constexpr float kFP8Margin = 1e-4;
constexpr float kFinfoAmaxE4M3 = 448.0f;
constexpr float kFinfoAmaxInvE4M3 = 1 / 448.0f;
#endif

__forceinline__ __device__ float fast_pow2(int x) {
    // We can ensure `-126 <= x and x <= 127`
    uint32_t bits_x = (x + 127) << 23;
    return *reinterpret_cast<float*>(&bits_x);
}

__forceinline__ __device__ int fast_log2_ceil(float x) {
    auto bits_x = *reinterpret_cast<uint32_t*>(&x);
    auto exp_x = (bits_x >> 23) & 0xff;
    auto man_bits = bits_x & ((1 << 23) - 1);
    return exp_x - 127 + (man_bits != 0);
}

__forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv, bool round_scale) {
    if (round_scale) {
        auto exp_scale_inv = fast_log2_ceil(amax * kFinfoAmaxInvE4M3);
        scale = fast_pow2(-exp_scale_inv);
        scale_inv = fast_pow2(exp_scale_inv);
    } else {
        scale_inv = amax * kFinfoAmaxInvE4M3;
        scale = kFinfoAmaxE4M3 / amax;
    }
}

template <bool kIsUE8M0, typename out_dtype_t = std::conditional_t<kIsUE8M0, uint8_t, float>>
__forceinline__ __device__ out_dtype_t extract_required_scale_format(float value) {
    if constexpr (kIsUE8M0) {
        return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
    } else {
        return value;
    }
}

#ifdef USE_ROCM
__device__ __forceinline__ int __all_sync(uint64_t mask, int predicate) {
    uint64_t predicate_bit_pattern = __ballot(predicate);
    return (~predicate_bit_pattern & mask) == 0;
}
#endif

template <int kNumRanks, bool kSyncOnly = false>
__forceinline__ __device__ void barrier_block(int** barrier_signal_ptrs, int rank) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // For non-sync-only cases, the memory operations by other threads in the block must be visible to the `sys` scope
    if constexpr (not kSyncOnly) {
        memory_fence();
        __syncthreads();
    }

    // Add self-ranks, sub other ranks
    if (thread_id < kNumRanks) {
        atomicAdd_system(barrier_signal_ptrs[rank] + thread_id, FINISHED_SUM_TAG);
        atomicSub_system(barrier_signal_ptrs[thread_id] + rank, FINISHED_SUM_TAG);
    }
    EP_DEVICE_ASSERT(kNumRanks <= blockDim.x);

    // Check timeout
    auto start_time = clock64();
    while (true) {
        auto value = thread_id < kNumRanks ? ld_volatile_global(barrier_signal_ptrs[rank] + thread_id) : 0;
        if (__all_sync(0xffffffff, value <= 0))
            break;

        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and thread_id < kNumRanks) {
            printf("DeepEP timeout check failed: rank = %d, thread = %d, value = %d)\n", rank, thread_id, value);
            trap();
        }
    }
    __syncthreads();
}

__forceinline__ __device__ int atomic_cas_cta_acquire(int* addr, int x, int y) {
#ifdef USE_ROCM
    int expected = x;
    __hip_atomic_compare_exchange_strong(addr, &expected, y, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    return expected;
#else
    int ret;
    asm volatile("atom.acquire.cta.shared::cta.cas.b32 %0, [%1], %2, %3;" : "=r"(ret) : "l"(addr), "r"(x), "r"(y) : "memory");
    return ret;
#endif
}

__forceinline__ __device__ int atomic_exch_cta_release(int* addr, int x) {
#ifdef USE_ROCM
    return __hip_atomic_exchange(addr, x, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
    int ret;
    asm volatile("atom.release.cta.shared::cta.exch.b32 %0, [%1], %2;" : "=r"(ret) : "l"(addr), "r"(x) : "memory");
    return ret;
#endif
}

__forceinline__ __device__ void acquire_lock(int* mutex) {
    // To make later memory operations valid, we must use `acquire` for memory semantics
    while (atomic_cas_cta_acquire(mutex, 0, 1) != 0)
        ;
}

__forceinline__ __device__ void release_lock(int* mutex) {
    // To make previous memory operations visible to other threads, we must use `release` for memory semantics
    atomic_exch_cta_release(mutex, 0);
}

// Operation functors
template <typename T>
struct ReduceSum {
    __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct ReduceMax {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};
template <typename T>
struct ReduceMin {
    __device__ T operator()(T a, T b) const { return a < b ? a : b; }
};
template <typename T>
struct ReduceAnd {
    __device__ T operator()(T a, T b) const { return a & b; }
};
template <typename T>
struct ReduceOr {
    __device__ T operator()(T a, T b) const { return a | b; }
};

// Unified reduction function
template <int kNumLanesPerGroup, bool kIntergroupReduce, typename T, typename Op>
__forceinline__ __device__ T warp_reduce(T value, Op op) {
#ifdef USE_ROCM
    EP_STATIC_ASSERT(kNumLanesPerGroup == 64 or kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or
                         kNumLanesPerGroup == 4 or kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
#else
    EP_STATIC_ASSERT(kNumLanesPerGroup == 32 or kNumLanesPerGroup == 16 or kNumLanesPerGroup == 8 or kNumLanesPerGroup == 4 or
                         kNumLanesPerGroup == 2 or kNumLanesPerGroup == 1,
                     "Invalid number of lanes");
#endif
    if constexpr (kIntergroupReduce) {
        if constexpr (kNumLanesPerGroup <= 1)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 1));
        if constexpr (kNumLanesPerGroup <= 2)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 2));
        if constexpr (kNumLanesPerGroup <= 4)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 4));
        if constexpr (kNumLanesPerGroup <= 8)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 8));
        if constexpr (kNumLanesPerGroup <= 16)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 16));
        if constexpr (kNumLanesPerGroup <= 32)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 32));
    } else {

        if constexpr (kNumLanesPerGroup >= 64)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 32));
        if constexpr (kNumLanesPerGroup >= 32)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 16));
        if constexpr (kNumLanesPerGroup >= 16)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 8));
        if constexpr (kNumLanesPerGroup >= 8)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 4));
        if constexpr (kNumLanesPerGroup >= 4)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 2));
        if constexpr (kNumLanesPerGroup >= 2)
            value = op(value, __shfl_xor_sync(kFullWarpMask, value, 1));
    }
    return value;
}
// Convenience aliases
template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_sum(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceSum<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_max(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMax<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_min(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceMin<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_and(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceAnd<T>{});
}

template <int kNumLanesPerGroup = kWarpSize, bool kIntergroupReduce = false, typename T>
__forceinline__ __device__ T warp_reduce_or(T value) {
    return warp_reduce<kNumLanesPerGroup, kIntergroupReduce, T>(value, ReduceOr<T>{});
}


//TODO::remove these functions
/////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_ROCM
__device__ __forceinline__ int ld_relaxed_sys_global(const int* ptr) {
    return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ int ld_relaxed_sys_global(const uint64_t* ptr) {
    return static_cast<int>(__hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ __forceinline__ int ld_relaxed_sys_global(const int64_t* ptr) {
    return static_cast<int>(__hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
}
#endif

__device__ __forceinline__ int atomic_add_relaxed_global(const int* ptr, int value) {
#ifdef USE_ROCM
    return __hip_atomic_fetch_add(const_cast<int*>(ptr), value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    int ret;
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
    return ret;
#endif
}

template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
    return cell_div<dtype_t>(a, b) * b;
}


__forceinline__ __device__ float half_warp_reduce_max(float value) {
#ifdef USE_ROCM
    if constexpr (kWarpSize == 64)
        value = max(value, __shfl_xor(value, 16, kWarpSize));
    value = max(value, __shfl_xor(value, 8, kWarpSize));
    value = max(value, __shfl_xor(value, 4, kWarpSize));
    value = max(value, __shfl_xor(value, 2, kWarpSize));
    value = max(value, __shfl_xor(value, 1, kWarpSize));
#else
    if constexpr (kWarpSize == 64)
        value = max(value, __shfl_xor_sync(kFullWarpMask, value, 16));
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, 8));
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, 4));
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, 2));
    value = max(value, __shfl_xor_sync(kFullWarpMask, value, 1));
#endif
    return value;
}

#ifdef USE_ROCM
__forceinline__ __device__ float quarter_warp_reduce_max(float value) {
    value = max(value, __shfl_xor(value, 8, kWarpSize));
    value = max(value, __shfl_xor(value, 4, kWarpSize));
    value = max(value, __shfl_xor(value, 2, kWarpSize));
    value = max(value, __shfl_xor(value, 1, kWarpSize));
    return value;
}
#endif

template <int kNumRanks>
__forceinline__ __device__ void move_fifo_slots(int& head) {
    head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int kNumRanks>
__device__ __forceinline__ bool not_finished(int* task, int expected) {
    bool result = false;
    auto lane_id = static_cast<int>(threadIdx.x) % kWarpSize;
    if (lane_id < kNumRanks) {
        result = ld_volatile_global(task + lane_id) != expected;
    }
#ifdef USE_ROCM
    return __any(result);
#else
    return __any_sync(kFullWarpMask, result);
#endif
}

template <int kNumRanks>
__forceinline__ __device__ void timeout_check(int** task_fifo_ptrs, int head, int rank, int expected, int tag = 0) {
    auto start_time = clock64();
    while (not_finished<kNumRanks>(task_fifo_ptrs[rank] + head, expected)) {
        auto elapsed_time = clock64() - start_time;
        if (elapsed_time > NUM_TIMEOUT_CYCLES and threadIdx.x == 0) {
            printf("DeepEP timeout check failed: %d (rank = %d)\n", tag, rank);
            trap();
        }
    }
}

template <int kNumRanks>
__forceinline__ __device__ void barrier_device(int** task_fifo_ptrs, int head, int rank, int tag = 0) {
    auto thread_id = static_cast<int>(threadIdx.x);
    EP_DEVICE_ASSERT(kNumRanks <= kWarpSize);

    if (thread_id < kNumRanks) {
        atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
        memory_fence();
        atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
    }
    timeout_check<kNumRanks>(task_fifo_ptrs, head, rank, 0, tag);
}


}  // namespace deep_ep
