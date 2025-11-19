#pragma once
/*
 * Temporary wrapper for for platform specific NVSHMEM and rocSHMEM functions.
 * Once hipify or hipify-torch fully supports this mapping, this file has to be
 * removed and according nvshmem* functions restored.
 */

#include "configs.cuh"

namespace deep_ep::internode {

// rocSHMEM wrapper
#if defined(USE_ROCM) && !FORCE_NVSHMEM_API
using shmem_team_t = rocshmem::rocshmem_team_t;
using shmem_team_config_t = rocshmem::rocshmem_team_config_t;
using shmemx_uniqueid_t = rocshmem::rocshmem_uniqueid_t;
using shmemx_init_attr_t = rocshmem::rocshmem_init_attr_t;
const shmem_team_t SHMEM_TEAM_INVALID = rocshmem::ROCSHMEM_TEAM_INVALID;
const shmem_team_t SHMEM_TEAM_WORLD = rocshmem::ROCSHMEM_TEAM_WORLD;
constexpr auto SHMEMX_INIT_WITH_UNIQUEID =
    rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID;

static inline const auto &shmemx_get_uniqueid = rocshmem::rocshmem_get_uniqueid;

static inline const auto &shmemx_set_attr_uniqueid_args =
    rocshmem::rocshmem_set_attr_uniqueid_args;

static inline const auto &shmemx_init_attr = rocshmem::rocshmem_init_attr;

static inline const auto &shmem_team_split_strided =
    rocshmem::rocshmem_team_split_strided;

static inline const auto &shmem_barrier_all = rocshmem::rocshmem_barrier_all;

static inline const auto &shmem_device_barrier_all = [] __device__() -> void {
    rocshmem::rocshmem_barrier_all();
};
static inline const auto &shmem_barrier = [] __device__(rocshmem::rocshmem_team_t team) {
    rocshmem::rocshmem_ctx_barrier(rocshmem::ROCSHMEM_CTX_DEFAULT, team);
};
static inline const auto &shmem_my_pe = rocshmem::rocshmem_my_pe;
static inline const auto &shmem_free = rocshmem::rocshmem_free;
// NOTE: No alternative for aligned allocation in rocSHMEM
static inline const auto &shmem_align = [](const size_t alignment,
                                           const size_t size) -> void * {
  return rocshmem::rocshmem_malloc(size);
};
static inline const auto &shmem_finalize = rocshmem::rocshmem_finalize;
static inline const auto &shmem_team_destroy = rocshmem::rocshmem_team_destroy;
static inline const auto &shmem_int_put_nbi = rocshmem::rocshmem_int_put_nbi;
static inline const auto &shmem_fence = rocshmem::rocshmem_fence;
static inline const auto &shmemx_int_put_nbi_warp =
    rocshmem::rocshmem_int_put_nbi_wave;
static inline const auto &shmemx_int8_put_nbi_warp =
    rocshmem::rocshmem_schar_put_nbi_wave;
static inline const auto &shmemx_int8_put_nbi =
    rocshmem::rocshmem_schar_put_nbi;
static inline const auto &shmem_signal_op_add =
    rocshmem::rocshmem_ulong_atomic_add;
static inline const auto &shmem_long_atomic_add =
    rocshmem::rocshmem_long_atomic_add;
static inline const auto &shmem_ibgda_amo_nonfetch_add = 
    rocshmem::rocshmem_uint64_atomic_add;

#if !defined(ROCM_DISABLE_CTX)
using shmem_ctx_t = rocshmem::rocshmem_ctx_t;
static inline const auto &shmem_wg_ctx_create = [] __device__(rocshmem::rocshmem_ctx_t *ctx) {
    return rocshmem::rocshmem_wg_ctx_create(ctx);
};
static inline const auto &shmem_wg_ctx_destroy =
    rocshmem::rocshmem_wg_ctx_destroy;
static inline const auto &shmem_ctx_quiet = rocshmem::rocshmem_ctx_quiet;
static inline const auto &shmem_ctx_ulong_atomic_add =
    rocshmem::rocshmem_ctx_ulong_atomic_add;
static inline const auto &shmem_ctx_long_atomic_add =
    rocshmem::rocshmem_ctx_long_atomic_add;
static inline const auto &shmem_ctx_schar_put_nbi_warp =
    rocshmem::rocshmem_ctx_schar_put_nbi_wave;
static inline const auto &shmem_ctx_int_put_nbi_warp =
    rocshmem::rocshmem_ctx_int_put_nbi_wave;
static inline const auto &shmem_ctx_schar_put_nbi =
    rocshmem::rocshmem_ctx_schar_put_nbi;
#endif
#else
// NVSHMEM wrapper
using shmem_team_t = nvshmem_team_t;
using shmem_team_config_t = nvshmem_team_config_t;
using shmemx_uniqueid_t = nvshmemx_uniqueid_t;
using shmemx_init_attr_t = nvshmemx_init_attr_t;
const shmem_team_t SHMEM_TEAM_INVALID = NVSHMEM_TEAM_INVALID;
const shmem_team_t SHMEM_TEAM_WORLD = NVSHMEM_TEAM_WORLD;
constexpr auto SHMEMX_INIT_WITH_UNIQUEID = NVSHMEMX_INIT_WITH_UNIQUEID;
static inline const auto &shmemx_get_uniqueid = nvshmemx_get_uniqueid;
static inline const auto &shmemx_set_attr_uniqueid_args =
    nvshmemx_set_attr_uniqueid_args;
static inline const auto &shmemx_init_attr = nvshmemx_init_attr;
static inline const auto &shmem_team_split_strided = nvshmem_team_split_strided;
static inline const auto &shmem_barrier_all = nvshmem_barrier_all;
static inline const auto &shmem_device_barrier_all = nvshmem_barrier_all;
static inline const auto &shmem_barrier = nvshmem_barrier;
static inline const auto &shmem_my_pe = nvshmem_my_pe;
static inline const auto &shmem_free = nvshmem_free;
static inline const auto &shmem_align = nvshmem_align;
static inline const auto &shmem_finalize = nvshmem_finalize;
static inline const auto &shmem_team_destroy = nvshmem_team_destroy;
static inline const auto &shmem_int_put_nbi = nvshmem_int_put_nbi;
static inline const auto &shmem_fence = nvshmem_fence;
static inline const auto &shmemx_int_put_nbi_warp = nvshmemx_int_put_nbi_warp;
static inline const auto &shmemx_int8_put_nbi_warp = nvshmemx_int8_put_nbi_warp;
static inline const auto &shmem_signal_op_add =
    [] __device__(uint64_t *sig_addr, uint64_t signal, int pe) {
      nvshmemx_signal_op(sig_addr, signal, NVSHMEM_SIGNAL_ADD, pe);
    };
#endif

} // namespace deep_ep::internode
