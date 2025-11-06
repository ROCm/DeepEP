#include <vector>
#include <cstring>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "shmem_wrapper.cuh"

namespace deep_ep {

namespace intranode {

template<int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

/*void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks) \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, task_fifo_ptrs, head, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, kWarpSize, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}*/

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream, int head = 0) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, head, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, kWarpSize, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {

shmem_team_t cpu_rdma_team = SHMEM_TEAM_INVALID;
shmem_team_config_t cpu_rdma_team_config;

std::vector<uint8_t> get_unique_id() {
    shmemx_uniqueid_t unique_id;
    shmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(shmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(shmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    shmemx_uniqueid_t root_unique_id;
    shmemx_init_attr_t attr;
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(shmemx_uniqueid_t));
    shmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    shmemx_init_attr(SHMEMX_INIT_WITH_UNIQUEID, &attr);

    // Create sub-RDMA teams
    // NOTES: if `num_ranks <= NUM_MAX_NVL_PEERS` then only low-latency kernels are used
    if (low_latency_mode and num_ranks > NUM_MAX_NVL_PEERS) {
        EP_HOST_ASSERT(cpu_rdma_team == SHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
        EP_HOST_ASSERT(shmem_team_split_strided(SHMEM_TEAM_WORLD, 
                                                rank % NUM_MAX_NVL_PEERS, 
                                                NUM_MAX_NVL_PEERS,
                                                num_ranks / NUM_MAX_NVL_PEERS, 
                                                &cpu_rdma_team_config, 
                                                0, 
                                                &cpu_rdma_team) == 0);
       //TODO::issue on ROCM: enable it for ROCM
#ifndef USE_ROCM
        EP_HOST_ASSERT(cpu_rdma_team != SHMEM_TEAM_INVALID);
#endif
    }
    // Normal operations use IBRC, while low-latency operations use IBGDA
    if (low_latency_mode) {
#ifndef USE_ROCM
        nvshmemi_device_host_state_t* dev_state_ptr = nullptr;
        CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_state_ptr), nvshmemi_device_state_d));
        bool ibgda_is_initialized = false;
        CUDA_CHECK(cudaMemcpy(&dev_state_ptr->ibgda_is_initialized, &ibgda_is_initialized, sizeof(bool), cudaMemcpyHostToDevice));
#endif
    }
    shmem_barrier_all();
    return shmem_my_pe();
}

void* alloc(size_t size, size_t alignment) {
    return shmem_align(alignment, size);
}

void free(void* ptr) {
    shmem_free(ptr);
}

void barrier() {
    shmem_barrier_all();
}

void finalize() {
    if (cpu_rdma_team != SHMEM_TEAM_INVALID) {
        shmem_team_destroy(cpu_rdma_team);
        cpu_rdma_team = SHMEM_TEAM_INVALID;
    }
    shmem_finalize();
}
} // namespace internode

} // namespace deep_ep
