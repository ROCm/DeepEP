# Install rocSHMEM
## Hardware Prerequisites
- AMD MI308X/MI300X GPU
- Infiniband ConnectX-7 (IB or RoCE both acceptable, ideally would test both variants)
   - For more detailed requirements, see [rocSHMEM](https://github.com/ROCm/rocSHMEM) github repository


## Build and installation

```bash
# Obtain develop branch
git clone git@github.com:ROCm/rocSHMEM.git

# Build dependencies Open MPI/UCX (used for RO, and as a bootstrap mechanism otherwise)
export BUILD_DIR=$PWD
../rocSHMEM/scripts/install_dependencies.sh
export PATH=$PWD/ompi/bin:$PATH
export LD_LIBRARY_PATH=$PWD/ucx/lib:$PWD/ompi/lib:$LD_LIBRARY_PATH

# Build rocSHMEM library, library will be installed in $HOME/rocshmem
mkdir build.mnic && cd build.mnic
MPI_ROOT=$BUILD_DIR/ompi ../rocSHMEM/scripts/build_configs/gda_mlx5 --fresh \
  -DUSE_IPC=ON \
  -DGDA_BNXT=ON

# You may pass additional arguments to Cmake,
#   e.g., -DBUILD_LOCAL_GPU_TARGET_ONLY=ON
```
# pytorch patch 

Follow the below instruction for pytorch commits older than [e4adf5d](https://github.com/pytorch/pytorch/commit/e4adf5df39d9c472c7dcbac18efde29241e238f0).
```bash
TARGET="/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py"
# 1) Create the patch (use a here-doc—safer than many echo’s)
cat > /tmp/torch.patch <<'PATCH'
--- /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py
+++ /opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/cpp_extension.py
@@ -2114,7 +2114,7 @@
     if cflags is not None:
         for flag in cflags:
             if 'amdgpu-target' in flag or 'offload-arch' in flag:
-                return ['-fno-gpu-rdc']
+                return ['-fgpu-rdc']
     # Use same defaults as used for building PyTorch
     # Allow env var to override, just like during initial cmake build.
     _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
@@ -2127,7 +2127,7 @@
     else:
         archs = _archs.replace(' ', ';').split(';')
     flags = [f'--offload-arch={arch}' for arch in archs]
-    flags += ['-fno-gpu-rdc']
+    flags += ['-fgpu-rdc']
     return flags
 def _get_build_directory(name: str, verbose: bool) -> str:
PATCH
# 2) (Optional) sanity check the file exists
[[ -f "$TARGET" ]] || { echo "Missing: $TARGET"; exit 1; }
# 3) Dry run first (shows what would change; no write)
patch --dry-run "$TARGET" < /tmp/torch.patch
# 4) Apply with a backup of the original file (*.orig)
patch --backup -z .orig "$TARGET" < /tmp/torch.patch
# 5) Verify the changes
grep -n "fgpu-rdc" "$TARGET"
```

