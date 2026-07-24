import argparse

import torch
import torch.distributed as dist

import deep_ep
import deep_ep_cpp
from utils import init_dist


NUM_EXPERTS = 4
NUM_TOPK = 2
HIDDEN = 7168


def make_batch(rank: int, iteration: int, microbatch: int):
    """Create unequal batches large enough to wrap the 128-slot queues."""
    if rank == 0:
        num_tokens = 2048 + (iteration * 17 + microbatch * 31) % 127
    else:
        num_tokens = 2176 + (iteration * 29 + microbatch * 43) % 131

    generator = torch.Generator(device="cuda")
    generator.manual_seed(100_000 * iteration + 1_000 * microbatch + rank)
    x = torch.randn(
        (num_tokens, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    if iteration % 2 == 0:
        # Each source sends to only one destination. The destination alternates
        # between the two shared-buffer sequences, leaving one rank group idle
        # while the other wraps its queues repeatedly.
        destination = rank if microbatch == 0 else 1 - rank
        topk_idx = torch.tensor(
            [destination * 2, destination * 2 + 1],
            dtype=deep_ep.topk_idx_t,
            device="cuda",
        ).expand(num_tokens, -1).contiguous()
    else:
        scores = torch.rand(
            (num_tokens, NUM_EXPERTS),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        topk_idx = scores.topk(
            NUM_TOPK, dim=-1, sorted=False
        ).indices.to(deep_ep.topk_idx_t)
    topk_weights = torch.rand(
        (num_tokens, NUM_TOPK),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    # DeepEP sends a token once to each distinct destination rank. An identity
    # expert therefore returns one copy for every destination rank selected.
    rank_idx = topk_idx // (NUM_EXPERTS // dist.get_world_size())
    expected_multiplier = torch.stack(
        [
            (rank_idx == destination).any(dim=1)
            for destination in range(dist.get_world_size())
        ],
        dim=1,
    ).sum(dim=1)
    return x, topk_idx, topk_weights, expected_multiplier


def dispatch(buffer: deep_ep.Buffer, config: deep_ep.Config, batch):
    x, topk_idx, topk_weights, expected_multiplier = batch
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        _,
    ) = buffer.get_dispatch_layout(
        topk_idx,
        NUM_EXPERTS,
        async_finish=False,
    )
    recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = buffer.dispatch(
        x=x,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        expert_alignment=1,
        config=config,
        async_finish=False,
    )
    return (
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        handle,
        expected_multiplier,
    )


def check_dispatch(
    receiver_rank: int,
    source_batches,
    recv_x: torch.Tensor,
    recv_topk_idx: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    handle,
):
    rank_prefix_matrix, _, _, recv_src_idx, _, _ = handle
    invalid_index = (
        NUM_EXPERTS // dist.get_world_size()
        if deep_ep_cpp.AITER_MOE
        else -1
    )
    local_expert_begin = receiver_rank * (
        NUM_EXPERTS // dist.get_world_size()
    )
    local_expert_end = local_expert_begin + (
        NUM_EXPERTS // dist.get_world_size()
    )

    expected_x = []
    expected_topk_idx = []
    expected_topk_weights = []
    start = 0
    for source_rank, source_batch in enumerate(source_batches):
        end = rank_prefix_matrix[source_rank][receiver_rank].item()
        source_indices = recv_src_idx[start:end].long()
        source_x, source_topk_idx, source_topk_weights, _ = source_batch

        expected_x.append(source_x[source_indices])
        source_indices_2d = source_indices[:, None].expand(-1, NUM_TOPK)
        selected_experts = source_topk_idx.gather(0, source_indices_2d)
        selected_weights = source_topk_weights.gather(0, source_indices_2d)
        is_local = (selected_experts >= local_expert_begin) & (
            selected_experts < local_expert_end
        )
        expected_topk_idx.append(
            torch.where(
                is_local,
                selected_experts - local_expert_begin,
                invalid_index,
            )
        )
        expected_topk_weights.append(
            torch.where(is_local, selected_weights, 0.0)
        )
        start = end

    assert start == recv_x.size(0)
    assert torch.equal(recv_x, torch.cat(expected_x))
    assert torch.equal(recv_topk_idx, torch.cat(expected_topk_idx))
    assert torch.equal(
        recv_topk_weights,
        torch.cat(expected_topk_weights),
    )


def test_loop(local_rank: int, num_processes: int, iterations: int):
    rank, num_ranks, group = init_dist(local_rank, num_processes)
    assert num_ranks == 2, "The ordering coverage requires two ranks"
    buffer = deep_ep.Buffer(
        group,
        int(512e6),
        0,
        low_latency_mode=False,
        num_qps_per_rank=1,
        explicitly_destroy=True,
    )
    dispatch_config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    combine_config = deep_ep.Buffer.get_combine_config(num_ranks)

    try:
        for iteration in range(iterations):
            batch0 = make_batch(rank, iteration, 0)
            batch1 = make_batch(rank, iteration, 1)

            # This is the order used when two vLLM microbatches share one
            # DeepEP buffer. It requires each producer's queue-tail publication
            # to make the corresponding payload visible to the consumer.
            (
                recv0,
                recv_topk_idx0,
                recv_topk_weights0,
                handle0,
                multiplier0,
            ) = dispatch(buffer, dispatch_config, batch0)
            (
                recv1,
                recv_topk_idx1,
                recv_topk_weights1,
                handle1,
                multiplier1,
            ) = dispatch(buffer, dispatch_config, batch1)
            out0, out_topk_weights0, _ = buffer.combine(
                x=recv0,
                handle=handle0,
                topk_weights=recv_topk_weights0,
                config=combine_config,
                async_finish=False,
            )
            out1, out_topk_weights1, _ = buffer.combine(
                x=recv1,
                handle=handle1,
                topk_weights=recv_topk_weights1,
                config=combine_config,
                async_finish=False,
            )
            torch.cuda.synchronize()

            source_batches0 = [
                make_batch(source_rank, iteration, 0)
                for source_rank in range(num_ranks)
            ]
            source_batches1 = [
                make_batch(source_rank, iteration, 1)
                for source_rank in range(num_ranks)
            ]
            check_dispatch(
                rank,
                source_batches0,
                recv0,
                recv_topk_idx0,
                recv_topk_weights0,
                handle0,
            )
            check_dispatch(
                rank,
                source_batches1,
                recv1,
                recv_topk_idx1,
                recv_topk_weights1,
                handle1,
            )

            expected0 = batch0[0] * multiplier0[:, None]
            expected1 = batch1[0] * multiplier1[:, None]
            expected_topk_weights0 = batch0[2]
            expected_topk_weights1 = batch1[2]
            for microbatch, output, expected, output_weights, expected_weights in (
                (
                    0,
                    out0,
                    expected0,
                    out_topk_weights0,
                    expected_topk_weights0,
                ),
                (
                    1,
                    out1,
                    expected1,
                    out_topk_weights1,
                    expected_topk_weights1,
                ),
            ):
                if not torch.equal(output, expected):
                    max_abs = (
                        (output - expected).abs().float().max().item()
                    )
                    raise AssertionError(
                        f"rank={rank} iteration={iteration} "
                        f"microbatch={microbatch} max_abs={max_abs}"
                    )
                if not torch.equal(output_weights, expected_weights):
                    max_abs = (
                        (output_weights - expected_weights)
                        .abs()
                        .max()
                        .item()
                    )
                    raise AssertionError(
                        f"rank={rank} iteration={iteration} "
                        f"microbatch={microbatch} "
                        f"topk_weights_max_abs={max_abs}"
                    )

        if rank == 0:
            print(
                f"[intranode] passed {iterations} shared-buffer iterations",
                flush=True,
            )
        dist.barrier(group)
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test shared-buffer intranode dispatch/combine ordering"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of two-microbatch sequences to run (default: 10)",
    )
    args = parser.parse_args()

    num_processes = 2
    torch.multiprocessing.spawn(
        test_loop,
        args=(num_processes, args.iterations),
        nprocs=num_processes,
    )
