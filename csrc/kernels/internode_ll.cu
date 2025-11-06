#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "buffer.cuh"
#include "utils.cuh"
#include "shmem_wrapper.cuh"
#include <cooperative_groups.h>
#include <rocshmem/rocshmem.hpp>
#include <iostream>
// low latency+RocSHMEM has issue with CTX.
#if defined(NIC_CX7)
  #define ROCM_DISABLE_CTX
#endif

namespace cg = cooperative_groups;
using namespace rocshmem;
namespace deep_ep {

namespace internode_ll {

__device__ void grid_barrier(int* global_counter, int num_blocks) {
    volatile int ret;
    __syncthreads();
    if (threadIdx.x == 0 ) {
        __threadfence();
	    ret = __hip_atomic_fetch_add( &global_counter[0], 1,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(global_counter,
                 __ATOMIC_RELAXED,
                 __HIP_MEMORY_SCOPE_AGENT) != num_blocks);
    }
    __syncthreads();
}


template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int64_t* clean_0, int num_clean_int_0,
                                         int64_t* clean_1, int num_clean_int_1) {
    // Barrier before cleaning (in case of unfinished chunked EP)
#ifdef USE_ROCM
    if (threadIdx.x == 0)
        internode::shmem_device_barrier_all();
#else
    nvshmemx_barrier_all_block();
#endif

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure low-latency mode work 
#ifdef USE_ROCM
    if (threadIdx.x == 0)
        internode::shmem_device_barrier_all();
#else
    nvshmemx_barrier_all_block();
#endif
}

void clean_low_latency_buffer(int64_t* clean_0,
                              int num_clean_int_0,
                              int64_t* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1);
}

template <bool kUseFP8, bool kMultinode, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * kWarpSize, 1) void
dispatch(void* packed_recv_x, float* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         int* global_atomic_counter,
         void* rdma_recv_x, int64_t* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int64_t* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    constexpr auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

#if !defined(ROCM_DISABLE_CTX)
    __shared__ internode::shmem_ctx_t ctx;
    if constexpr (kMultinode)
        EP_DEVICE_ASSERT(internode::shmem_wg_ctx_create(&ctx) == 0);
#endif

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
#ifdef USE_ROCM
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 240, kFP8AmaxInv = 1.0f / 240.0f;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__hip_fp8_storage_t) : sizeof(gpu_bfloat16_t));
#else
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__hip_fp8_storage_t) : sizeof(gpu_bfloat16_t));
#endif
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(gpu_bfloat16_t)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // Expert counts
    __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps ) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(gpu_bfloat16_t);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize % kNumPerChannels == 0, "Invalid vectorization");
        constexpr int num_threads = kNumWarpGroups * kNumWarpsPerGroup * kWarpSize;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index write
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr(kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<gpu_bfloat16_t*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }
#ifdef USE_ROCM
                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize / kNumPerChannels == 4, "Invalid vectorization");
                    amax = quarter_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
                    if (lane_id % 16 == 0)
#else
                    EP_STATIC_ASSERT(kNumElemsPerRead * kWarpSize / kNumPerChannels == 2, "Invalid vectorization");
                    amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
                    if (lane_id == 0 or lane_id == 16)
#endif
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
#ifdef USE_ROCM
                    auto fp8x2_values = reinterpret_cast<__hip_fp8x2_storage_t*>(&int2_value);
#else
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
#endif
                    #pragma unroll
                    for (int j = 0;j < kNumElemsPerRead;j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
#ifdef USE_ROCM
                        fp8x2_values[j / 2] = __hip_cvt_float2_to_fp8x2(fp32x2, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
#else
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
#endif
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
#ifdef USE_ROCM
            __syncthreads();
#else
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));
#endif
            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = shfl_sync(slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;
                if (dst_rank != rank) {

#ifdef USE_ROCM
                    if constexpr (!kMultinode) {
                        internode::shmemx_int8_put_nbi_warp(reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr), num_bytes_per_msg, dst_rank);
                    } else {
#if defined(ROCM_DISABLE_CTX)
                        internode::shmemx_int8_put_nbi_warp(reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr), num_bytes_per_msg, dst_rank);
#else //DISABLE_CTX
                        internode::shmem_ctx_schar_put_nbi_warp(ctx, reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(src_ptr), num_bytes_per_msg, dst_rank);
#endif
                    }
#else //USE_ROCM
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
#endif
                } else {
                    // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                    UNROLLED_WARP_COPY(4, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                // Increase counter after finishing
                syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
#ifndef USE_ROCM
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_local_experts);
#endif
            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id;i < num_next_clean_int;i += kWarpSize)
                next_clean[i] = 0;
            if constexpr (kMultinode){
                syncwarp();
                #pragma unroll 4
                for (int i = lane_id;i < num_experts;i += kWarpSize)
                    atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
            }
        }
        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * kNumWarpGroups;
        const auto expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);

        // Per lane count
        #pragma unroll 2
        for (int i = lane_id; i < num_tokens * num_topk; i += kWarpSize) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        // Warp reduce
        #pragma unroll 2
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                if constexpr (kMultinode){
                    atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
                }else{
                    atomic_add_relaxed_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
                }
            }
        }
    }

#if defined(NIC_IO) || defined(NIC_THOR2)
     if constexpr (kMultinode){
         if (thread_id == 0 ){
#if defined(ROCM_DISABLE_CTX)
                    internode::shmem_fence();
#else
                    internode::shmem_ctx_quiet(ctx);
#endif
        }
     }
#endif

     __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * kNumWarpGroups];

        // Wait local sends issued and send expert counts
        if constexpr(kMultinode){
            while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        }else{
            while (ld_volatile_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG);
        }
        if (dst_rank != rank) {
#ifdef USE_ROCM
           if constexpr (!kMultinode){
                rocshmem::rocshmem_long_p(rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1, dst_rank);
            }else{
#if defined(ROCM_DISABLE_CTX)
                internode::shmem_long_atomic_add( rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1, dst_rank);
#else
                internode::shmem_ctx_long_atomic_add(ctx, rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1, dst_rank);
#endif
            }
#else //CUDA
           nvshmemi_ibgda_amo_nonfetch_add(rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
#endif
        } else {
            st_na_release(reinterpret_cast<int *>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank), -num_tokens_sent - 1);
        }
#if defined(NIC_IO) || defined(NIC_THOR2)
     if constexpr (kMultinode){
         if (thread_id == 0 ){
#if defined(ROCM_DISABLE_CTX)
                    internode::shmem_fence();
#else
                    internode::shmem_ctx_quiet(ctx);
#endif
        }
     }
#endif
        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0) {
#if defined(USE_ROCM) && !defined(ROCM_DISABLE_CTX)
        if constexpr (kMultinode)
            internode::shmem_wg_ctx_destroy(&ctx);
#endif
        return;
    }

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE){
        grid_barrier(global_atomic_counter, num_sms);
    }
    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_x_scales = packed_recv_x_scales + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_scales;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumWarpGroups], shared_recv_token_begin_idx[kNumWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens, recv_token_begin_idx;
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        if (sub_warp_id == 0 and lane_id == 0) {
            auto start_time = clock64();
            if constexpr (kMultinode){
                while ((num_recv_tokens = ld_acquire_sys_global(reinterpret_cast<int64_t*>(rdma_recv_count + local_expert_idx * num_ranks + src_rank))) == 0){
                     if ((clock64() - start_time) >= NUM_TIMEOUT_CYCLES){
                         printf("dispatch recieve time out \n");
                    }
                }
            }else{
                while ((num_recv_tokens = ld_volatile_global(reinterpret_cast<int64_t*>(rdma_recv_count + local_expert_idx * num_ranks + src_rank))) == 0){
                         if ((clock64() - start_time) >= NUM_TIMEOUT_CYCLES){
                         printf("dispatch recieve single node time out \n");
                    }
                }
            }
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        }
#ifdef USE_ROCM
	__syncthreads();
#else
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(kNumWarpsPerGroup * 32));
#endif
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(8, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr(kUseFP8) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto dst_scales = reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
                const auto scale_stride = num_ranks * num_max_dispatch_tokens_per_rank;
                auto scale_0 = lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
                auto scale_1 = (lane_id + kWarpSize) < num_scales ? ld_nc_global(src_scales + lane_id + kWarpSize) : 0;
                lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0 : 0.0f;
                (lane_id + kWarpSize) < num_scales ? dst_scales[(lane_id + kWarpSize) * scale_stride] = scale_1 : 0.0f;
            }
        }
    }
#if !defined(ROCM_DISABLE_CTX)
    if constexpr (kMultinode)
        internode::shmem_wg_ctx_destroy(&ctx);
#endif
}

void dispatch(void* packed_recv_x,
              float* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int64_t* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int64_t* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases,
              int* global_atomic_counter = NULL) {

#ifdef USE_ROCM
    constexpr int kNumWarpsPerGroup = 8;
    constexpr int kNumWarpGroups = 2;
#else
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
#endif
    constexpr int kNumMaxTopK = 9;
    EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    bool kMultinode = (num_ranks > 8);
#define DISPATCH_LAUNCH_CASE(hidden) { \
  auto dispatch_func =                                                                      \
    use_fp8                                                                                 \
      ? ( kMultinode                                                                          \
            ? dispatch<true,  true,  kNumWarpGroups, kNumWarpsPerGroup, hidden>            \
            : dispatch<true,  false, kNumWarpGroups, kNumWarpsPerGroup, hidden> )          \
      : ( kMultinode                                                                          \
            ? dispatch<false, true,  kNumWarpGroups, kNumWarpsPerGroup, hidden>            \
            : dispatch<false, false, kNumWarpGroups, kNumWarpsPerGroup, hidden> );\
LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              global_atomic_counter, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, phases);} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kMultinode, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * kWarpSize, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int64_t* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* global_atomic_counter,
        int64_t* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int phases, bool zero_copy) {

#if !defined(ROCM_DISABLE_CTX)
    __shared__ internode::shmem_ctx_t ctx;
    if constexpr(kMultinode)
        EP_DEVICE_ASSERT(internode::shmem_wg_ctx_create(&ctx) == 0);
#endif
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / kWarpSize, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(gpu_bfloat16_t);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Message package
    // BF16 mode: always use BF16 for hidden data (ignoring the extra flag slot)
    constexpr size_t num_bytes_per_slot = sizeof(int4) + kHidden * sizeof(gpu_bfloat16_t);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id ;i < num_next_clean_int; i += kWarpSize)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        syncwarp();
        if (lane_id == 0){
            if constexpr (kMultinode){
                atomic_add_release_global(atomic_clean_flag, num_experts);
            }else{
                atomic_add_relaxed_global(atomic_clean_flag, num_experts);
            }
        }
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // Issue IBGDA send
        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row + 4);

            // Copy directly to local rank, or copy to buffer and issue RDMA
            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot + sizeof(int4);
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(4, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY(4, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                
                //nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, hidden * sizeof(gpu_bfloat16_t), dst_rank, local_expert_idx, lane_id, token_idx - offset);
                if constexpr (!kMultinode){
                    internode::shmemx_int8_put_nbi_warp(reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(buf_ptr), hidden * sizeof(gpu_bfloat16_t), dst_rank);
                }else{
#if defined(ROCM_DISABLE_CTX)
                    internode::shmemx_int8_put_nbi_warp(reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(buf_ptr), hidden * sizeof(gpu_bfloat16_t), dst_rank);
#else
                    internode::shmem_ctx_schar_put_nbi_warp(ctx,reinterpret_cast<signed char*>(dst_ptr), reinterpret_cast<signed char*>(buf_ptr), hidden * sizeof(gpu_bfloat16_t), dst_rank);
#endif
                }
                if constexpr (kMultinode){
#if defined(ROCM_DISABLE_CTX)
                    internode::shmem_fence();
#else
                    internode::shmem_ctx_quiet(ctx);
#endif
                }
        }

        // Put finishing flag
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
#ifdef USE_ROCM
        __syncthreads();
#else
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
#endif
        if (sub_warp_id == 0 and lane_id == 0) {
            if constexpr (kMultinode){
                while (ld_acquire_global(atomic_clean_flag) == 0);
            }else{
                while (ld_volatile_global(atomic_clean_flag) == 0);
            }

            if (dst_rank != rank) {
#ifdef USE_ROCM
                if constexpr (!kMultinode){
                    rocshmem::rocshmem_long_p(rdma_recv_flag + global_expert_idx, 1, dst_rank);
                } else {
#if defined(ROCM_DISABLE_CTX)
                    internode::shmem_long_atomic_add(rdma_recv_flag + global_expert_idx, 1, dst_rank);
#else
                    internode::shmem_ctx_long_atomic_add(ctx, rdma_recv_flag + global_expert_idx, 1, dst_rank);
#endif //DISABLE_CTX
                }
#else
                nvshmemi_ibgda_amo_nonfetch_add(rdma_recv_flag + global_expert_idx, 1, dst_rank, local_expert_idx);
#endif //USE_ROCM
            } else {
                st_na_release(reinterpret_cast<int*>(rdma_recv_flag + global_expert_idx), 1);
            }
            atomic_add_relaxed_global(atomic_clean_flag, -1);
                if constexpr (kMultinode){
#if defined(ROCM_DISABLE_CTX)
                    internode::shmem_fence();
#else
                    internode::shmem_ctx_quiet(ctx);
#endif
                }

        }
    }

    // Receiving phase
    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0){
#if !defined(ROCM_DISABLE_CTX)
        if constexpr (kMultinode)
            internode::shmem_wg_ctx_destroy(&ctx);
#endif

        return;
    }
    // Wait all ranks to arrive and notify PCIe usage
    if (responsible_expert_idx < num_experts) {
        // EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0){
            if constexpr (kMultinode){
                while (ld_acquire_sys_global(reinterpret_cast<int64_t*>(rdma_recv_flag + responsible_expert_idx)) == 0);
            }else{
                while (ld_volatile_global(reinterpret_cast<int64_t*>(rdma_recv_flag + responsible_expert_idx)) == 0);
            }
        }
    }
    grid_barrier(global_atomic_counter, num_sms);

    // Reduce tokens with FP8 cast
    EP_DEVICE_ASSERT(num_topk <= kWarpSize and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (kWarpSize * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens;token_idx += num_sms) {
            // Read top-k indices and weights
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                // Read from sources
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type + 4);

                // Reduce
                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<gpu_bfloat16_t*>(&x_vec);
                #pragma unroll 4
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            // Write results
            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<gpu_bfloat16_t*>(&combined_values);
            #pragma unroll 4
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<gpu_bfloat16_t>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
#if !defined(ROCM_DISABLE_CTX)
    if constexpr (kMultinode)
        internode::shmem_wg_ctx_destroy(&ctx);
#endif
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int64_t* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int64_t* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy,
             int* global_atomic_counter = NULL) {
#ifdef USE_ROCM
    constexpr int kNumWarpsPerGroup = 8;
    constexpr int kNumWarpGroups = 2;
#else
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
#endif
    constexpr int kNumMaxTopk = 9;

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    // Check workspace
    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);
    bool kMultinode = (num_ranks > 8);
#define COMBINE_LAUNCH_CASE(hidden) { \
auto combine_func = kMultinode ? combine<true, kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>: \
                                combine<false, kNumWarpGroups, kNumWarpsPerGroup, hidden, kNumMaxTopk>;\
LAUNCH_KERNEL_NON_COOPERATIVE(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              global_atomic_counter, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              phases, zero_copy);} break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * kWarpSize, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}


template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = num_sms * kNumThreads;
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads) {
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
    }
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 1024;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    if (sm_id == 0 && thread_id == 0) {
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
    }
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
}


} // namespace internode_ll

} // namespace deep_ep
