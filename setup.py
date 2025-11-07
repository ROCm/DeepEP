import argparse
import os
import subprocess
import sys

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Wheel specific: the wheels only include the soname of the host library `libnvshmem_host.so.X`
def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')

if __name__ == "__main__":
    # Add argument parser for handling --variant flag
    parser = argparse.ArgumentParser(description="DeepEP setup configuration")
    parser.add_argument(
        "--variant",
        type=str,
        default="cuda",
        choices=["cuda", "rocm"],
        help="Architecture variant (cuda or rocm)",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose build")
    parser.add_argument("--enable_timer", action="store_true", help="Enable timer to debug time out in internode")
    parser.add_argument("--rocm-disable-ctx", action="store_true", help="Disable workgroup context optimization in internode")
    parser.add_argument("--enable-mpi", action="store_true", help="Enable MPI detection and configuration")
    parser.add_argument("--nic", type=str, default="cx7", choices=["cx7", "thor2", "io"], help="Target NIC architecture (e.g., cx7, thor2)")

    # Get the arguments to be parsed and separate setuptools arguments
    args, unknown_args = parser.parse_known_args()
    variant = args.variant
    debug = args.debug
    rocm_disable_ctx = args.rocm_disable_ctx
    enable_mpi = args.enable_mpi
    enable_timer = args.enable_timer
    nic_type = args.nic


    if variant != "rocm":
        disable_nvshmem = False
        nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
        nvshmem_host_lib = 'libnvshmem_host.so'
        if nvshmem_dir is None:
            try:
                nvshmem_dir = importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations[0]
                nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
                import nvidia.nvshmem as nvshmem  # noqa: F401
            except (ModuleNotFoundError, AttributeError, IndexError):
                print(
                    'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All internode and low-latency features are disabled\n'
                )
                disable_nvshmem = True
        else:
            disable_nvshmem = False    
            
        if not disable_nvshmem:
            assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'
            
    else:
        disable_nvshmem = False    


    # Reset sys.argv for setuptools to avoid conflicts
    sys.argv = [sys.argv[0]] + unknown_args

    print(f"Building for variant: {variant}")
    if (nic_type == "cx7" and rocm_disable_ctx == False):
        print("Warning: ctx is disabled for low latency and cx7!")

    if variant == "rocm":
        rocm_path = os.getenv("ROCM_HOME", "/opt/rocm")
        assert os.path.exists(rocm_path), f"Failed to find ROCm directory: {rocm_path}"
        os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"
        os.environ["CC"] = f"{rocm_path}/bin/hipcc"
        os.environ["CXX"] = f"{rocm_path}/bin/hipcc"
        os.environ["ROCM_HOME"] = rocm_path
        print(f'ROCm directory: {os.environ["ROCM_HOME"]}')

    shmem_variant_name = "NVSHMEM" if variant == "cuda" else "rocSHMEM"
    shmem_dir = (
        os.getenv("NVSHMEM_DIR", None)
        if variant == "cuda"
        else os.getenv("ROCSHMEM_DIR", f'{os.getenv("HOME")}/rocshmem')
    )
    assert shmem_dir is not None and os.path.exists(
        shmem_dir
    ), f"Failed to find {shmem_variant_name}"
    print(f"{shmem_variant_name} directory: {shmem_dir}")

    ompi_dir = None
    if variant == "rocm" and enable_mpi:
        # Attempt to auto-detect OpenMPI installation directory if OMPI_DIR not set.
        # The first existing candidate containing bin/mpicc will be used.
        ompi_dir_env = os.getenv("OMPI_DIR", "").strip()
        candidate_dirs = [
            ompi_dir_env if ompi_dir_env else None,
            "/opt/ompi",
            "/opt/openmpi",
            "/opt/rocm/ompi",
            "/usr/lib/x86_64-linux-gnu/openmpi",
            "/usr/lib/openmpi",
            "/usr/local/ompi",
            "/usr/local/openmpi",
        ]
        ompi_dir = None
        for d in candidate_dirs:
            if not d:
                continue
            mpicc_path = os.path.join(d, "bin", "mpicc")
            if os.path.exists(d) and os.path.exists(mpicc_path):
                ompi_dir = d
                break
        
        assert ompi_dir is not None, (
            f"Failed to find OpenMPI installation. "
            f"Searched: {', '.join([d for d in candidate_dirs if d])}. "
            f"Set OMPI_DIR environment variable or use --disable-mpi flag."
        )             
        print(f"Detected OpenMPI directory: {ompi_dir}")
    elif variant == "rocm" and not enable_mpi:
        print("MPI detection disabled for ROCm variant")
    elif variant == "cuda" and enable_mpi:
        print("MPI detection enabled for CUDA variant")
    else:
        print("MPI detection disabled for CUDA variant")

    # TODO: currently, we only support Hopper architecture, we may add Ampere support later
    if variant == "rocm":
        arch = os.getenv("PYTORCH_ROCM_ARCH")
        allowed_arch = {"gfx942", "gfx950"}
        if arch not in allowed_arch:
            raise EnvironmentError(
                f"Invalid PYTORCH_ROCM_ARCH='{arch}'. "
                f"Use one of: {', '.join(sorted(allowed_arch))}.")
    elif variant == "cuda":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"

    optimization_flag = "-O0" if debug else "-O3"
    debug_symbol_flags = ["-g", "-ggdb"] if debug else []
    define_macros = (
        ["-DUSE_ROCM=1", "-DDISABLE_SM90_FEATURES=1", "-fgpu-rdc",] if variant == "rocm" else []
    )
    if enable_timer:
        define_macros.append("-DENABLE_TIMER")
    if variant == "cuda" or rocm_disable_ctx:
        define_macros.append("-DROCM_DISABLE_CTX=1")
    if nic_type:
        nic_macro = f"-DNIC_{nic_type.upper()}=1"
        define_macros.append(nic_macro)
        print(f"Building with NIC Macro: {nic_macro}")
    cxx_flags = (
        [
            f"{optimization_flag}",
            "-Wno-deprecated-declarations",
            "-Wno-unused-variable",
            "-Wno-sign-compare",
            "-Wno-reorder",
            "-Wno-attributes",
        ]
        + debug_symbol_flags
        + define_macros
    )
    if variant == "cuda":
        nvcc_flags = [
            f"{optimization_flag}",
            "-Xcompiler",
            f"{optimization_flag}",
            "-rdc=true",
            "--ptxas-options=--register-usage-level=10",
            "--extended-lambda",
        ] + debug_symbol_flags
    elif variant == "rocm":
        nvcc_flags = [f"{optimization_flag}"] + debug_symbol_flags + define_macros

    include_dirs = ["csrc/", f"{shmem_dir}/include"]
    if variant == "rocm" and ompi_dir is not None:
        include_dirs.append(f"{ompi_dir}/include")

    sources = [
        "csrc/deep_ep.cpp",
        "csrc/kernels/runtime.cu",
        'csrc/kernels/layout.cu',
        "csrc/kernels/intranode.cu",
        "csrc/kernels/internode.cu",
        "csrc/kernels/internode_ll.cu",
    ]

    library_dirs = [f"{shmem_dir}/lib"]
    if variant == "rocm" and ompi_dir is not None:
        library_dirs.append(f"{ompi_dir}/lib")

    # Disable aggressive PTX instructions
    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "0")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        nvcc_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")


    # Bits of `topk_idx.dtype`, choices are 32 and 64
    if "TOPK_IDX_BITS" in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    shmem_lib_name = "nvshmem" if variant == "cuda" else "rocshmem"
    # Disable DLTO (default by PyTorch)
    nvcc_dlink = ["-dlink", f"-L{shmem_dir}/lib", f"-l{shmem_lib_name}"]
    extra_link_args = [f"-l:lib{shmem_lib_name}.a", f"-Wl,-rpath,{shmem_dir}/lib"]
    if variant == "cuda":
        extra_link_args.append("-l:nvshmem_bootstrap_uid.so")
    elif variant == "rocm":
        extra_link_args.extend(
            [
                "-fgpu-rdc",
                "--hip-link",
                "-lamdhip64",
                "-lhsa-runtime64",
                "-libverbs",
            ]
        )
        if enable_mpi:
            extra_link_args.extend(
                [
                    f"-l:libmpi.so",
                    f"-Wl,-rpath,{ompi_dir}/lib",
                ]
            )

    extra_compile_args = {
        "cxx": cxx_flags,
        "nvcc": nvcc_flags,
    }
    if variant == "cuda":
        extra_compile_args["nvcc_dlink"] = nvcc_dlink


    # Summary
    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > NVSHMEM path: {shmem_dir}')
    print(f' > Disable MPI: {disable_mpi}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except Exception as _:
        revision = ""

    setuptools.setup(
        name="deep_ep",
        version="1.2.1" + revision,
        packages=setuptools.find_packages(include=["deep_ep"]),
        ext_modules=[
            CUDAExtension(
                name="deep_ep_cpp",
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
