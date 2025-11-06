# DeepEP
DeepEP is a high-performance communication library for Mixture-of-Experts (MoE) and expert parallelism (EP), providing throughput‑oriented and latency‑optimized GPU all‑to‑all (dispatch / combine) primitives with optional FP8 / BF16 support.

This edition adds preliminary AMD GPU (ROCm) enablement:
- xGMI / Infinity Fabric intranode + InfiniBand / RoCE RDMA internode with [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) kernels.
- Unified Python API with the upstream [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- Some NVIDIA-only optimizations (e.g., PTX loads) are not mirrored; AMD path focuses on correctness + baseline overlap.

Notice: 
- Implementation details may differ from the DeepSeek-V3 paper, and AMD behavior may lag NVIDIA in advanced tuning until subsequent releases.
- The Pytorch commit [e4adf5d](https://github.com/pytorch/pytorch/commit/e4adf5df39d9c472c7dcbac18efde29241e238f0) fixed a bug of the `-fgpu-rdc` flag in Pytorch ROCm submodule. Please use any Pytorch version later than this commit, e.g., nightly Pytorch. (If one is using a Pytorch version that is older than [e4adf5d](https://github.com/pytorch/pytorch/commit/e4adf5df39d9c472c7dcbac18efde29241e238f0), please apply the patch for Pytorch in [rocSHMEM Installation Guide](third-party/README.md))

## Quick start

### Requirements

- MI308X/MI300X GPU (will support more architectures or devices later)
- Python 3.8 and above
- ROCm 6.3.4
- PyTorch [e4adf5d](https://github.com/pytorch/pytorch/commit/e4adf5df39d9c472c7dcbac18efde29241e238f0) and above
- xGMI for intranode communication
- RDMA network for internode communication

### Download and install rocSHMEM dependency

DeepEP (AMD version) depends on [rocSHMEM](https://github.com/ROCm/rocSHMEM). Please take a look at [rocSHMEM Installation Guide](third-party/README.md) for instructions.

### Development

```bash
git clone https://github.com/ROCm/DeepEP
cd DeepEP

# export OMPI dir in the next command (e.g., it's $BUILD_DIR/ompi in third-party/README.md)
export OMPI_DIR=<ompi_dir>
python3 setup.py --variant rocm build develop --user

# To use DeepEP without MPI, please make sure rocSHMEM was built with this flag -DUSE_EXTERNAL_MPI=OFF
# Then install DeepEP using this command
python3 setup.py --variant rocm --disable-mpi build develop --user

# Run test cases
# NOTES: you may modify the `init_dist` function in `tests/utils.py`
# according to your own cluster settings, and launch into multiple nodes
python3 tests/test_intranode.py
python3 tests/test_internode.py
python3 tests/test_low_latency.py
```

### Installation

```bash
python3 setup.py --variant rocm build develop
```

Then, import `deep_ep` in your Python project, and enjoy!

## Network configurations

Currently, DeepEP (AMD version) supports Mellanox ConnectX-7 NIC (will support more NICs later).

## Interfaces and examples

DeepEP (AMD version) shares the same Python interface with [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP); please refer to the example in the [upstream README](https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#interfaces-and-examples) 

## Roadmap

- [x] MI350X support
- [ ] AMD Pensando AI NIC support
- [ ] Broadcom NIC support

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

The credits of the original innovation belong to [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP); please cite their paper:

```bibtex
@misc{deepep2025,
      title={DeepEP: an efficient expert-parallel communication library},
      author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepEP}},
}
```
