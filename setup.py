import argparse
import os
import subprocess
import sys

import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_nvshmem_host_lib_name(base_dir):
    path = Path(base_dir).joinpath('lib')
    for file in path.rglob('libnvshmem_host.so.*'):
        return file.name
    raise ModuleNotFoundError('libnvshmem_host.so not found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepEP setup configuration')
    parser.add_argument('--variant', type=str, default='cuda', choices=['cuda', 'rocm'],
                        help='Architecture variant (cuda or rocm)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose build')
    parser.add_argument('--enable_timer', action='store_true', help='Enable timer to debug time out in internode')
    parser.add_argument('--rocm-disable-ctx', action='store_true', help='Disable workgroup context optimization in internode')
    parser.add_argument('--rocm-explicit-ctx', action='store_true', help='Use per-expert explicit contexts instead of shared workgroup context')
    parser.add_argument('--enable-mpi', action='store_true', help='Enable MPI detection and configuration')
    parser.add_argument('--nic', type=str, default='cx7', choices=['cx7', 'thor2', 'io'], help='Target NIC architecture (e.g., cx7, thor2)')

    args, unknown_args = parser.parse_known_args()
    variant = args.variant
    debug = args.debug
    rocm_disable_ctx = args.rocm_disable_ctx
    rocm_explicit_ctx = args.rocm_explicit_ctx
    enable_mpi = args.enable_mpi
    enable_timer = args.enable_timer
    nic_type = args.nic

    sys.argv = [sys.argv[0]] + unknown_args

    print(f'Building for variant: {variant}')
    if nic_type == 'cx7' and not rocm_disable_ctx:
        print('Warning: ctx is disabled for low latency and cx7!')

    # --- ROCm environment ---
    if variant == 'rocm':
        rocm_path = os.getenv('ROCM_HOME', '/opt/rocm')
        assert os.path.exists(rocm_path), f'Failed to find ROCm directory: {rocm_path}'
        os.environ['TORCH_DONT_CHECK_COMPILER_ABI'] = '1'
        os.environ['CC'] = f'{rocm_path}/bin/hipcc'
        os.environ['CXX'] = f'{rocm_path}/bin/hipcc'
        os.environ['ROCM_HOME'] = rocm_path
        print(f'ROCm directory: {os.environ["ROCM_HOME"]}')

    # --- SHMEM library detection ---
    disable_nvshmem = False
    nvshmem_host_lib = 'libnvshmem_host.so'

    if variant == 'cuda':
        nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
        if nvshmem_dir is None:
            try:
                nvshmem_dir = importlib.util.find_spec('nvidia.nvshmem').submodule_search_locations[0]
                nvshmem_host_lib = get_nvshmem_host_lib_name(nvshmem_dir)
                import nvidia.nvshmem as nvshmem  # noqa: F401
            except (ModuleNotFoundError, AttributeError, IndexError):
                print(
                    'Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. '
                    'All internode and low-latency features are disabled\n'
                )
                disable_nvshmem = True
        if not disable_nvshmem:
            assert os.path.exists(nvshmem_dir), f'The specified NVSHMEM directory does not exist: {nvshmem_dir}'
            shmem_dir = nvshmem_dir
            print(f'NVSHMEM directory: {shmem_dir}')
        else:
            shmem_dir = None
    else:
        shmem_dir = os.getenv('ROCSHMEM_DIR', f'{os.getenv("HOME")}/rocshmem')
        assert shmem_dir is not None and os.path.exists(shmem_dir), f'Failed to find rocSHMEM at: {shmem_dir}'
        print(f'rocSHMEM directory: {shmem_dir}')

    # --- OpenMPI detection (ROCm) ---
    ompi_dir = None
    if variant == 'rocm' and enable_mpi:
        print('MPI detection enabled for ROCm variant')
        ompi_dir_env = os.getenv('OMPI_DIR', '').strip()
        candidate_dirs = [
            ompi_dir_env if ompi_dir_env else None,
            '/opt/ompi',
            '/opt/openmpi',
            '/opt/rocm/ompi',
            '/usr/lib/x86_64-linux-gnu/openmpi',
            '/usr/lib/openmpi',
            '/usr/local/ompi',
            '/usr/local/openmpi',
        ]
        for d in candidate_dirs:
            if not d:
                continue
            if os.path.exists(d) and os.path.exists(os.path.join(d, 'bin', 'mpicc')):
                ompi_dir = d
                break
        assert ompi_dir is not None, (
            f'Failed to find OpenMPI installation. '
            f'Searched: {", ".join([d for d in candidate_dirs if d])}. '
            f'Set OMPI_DIR environment variable or remove --enable-mpi flag.'
        )
        print(f'Detected OpenMPI directory: {ompi_dir}')
    elif variant == 'rocm':
        print('MPI detection disabled for ROCm variant')

    # --- Architecture ---
    if variant == 'rocm':
        allowed_arch = {'gfx942', 'gfx950'}
        arch_env = os.getenv('PYTORCH_ROCM_ARCH', '').strip()
        if not arch_env:
            arch_list = ['gfx942', 'gfx950']
            os.environ['PYTORCH_ROCM_ARCH'] = ';'.join(arch_list)
            print(f'PYTORCH_ROCM_ARCH not set; defaulting to \'{os.environ["PYTORCH_ROCM_ARCH"]}\'')
        else:
            raw_list = [a.strip() for a in arch_env.replace(',', ';').split(';') if a.strip()]
            keep = [a for a in raw_list if a in allowed_arch]
            if not keep:
                raise EnvironmentError(
                    f'Invalid PYTORCH_ROCM_ARCH=\'{arch_env}\'. '
                    f'DeepEP ROCm build supports only: {", ".join(sorted(allowed_arch))}.'
                )
            new_env = ';'.join(dict.fromkeys(keep))
            if new_env != arch_env:
                print(f'Filtering PYTORCH_ROCM_ARCH from \'{arch_env}\' to \'{new_env}\' (DeepEP supports only gfx942/gfx950)')
                os.environ['PYTORCH_ROCM_ARCH'] = new_env
    else:
        if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
            os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '8.0')
        else:
            os.environ['TORCH_CUDA_ARCH_LIST'] = os.getenv('TORCH_CUDA_ARCH_LIST', '9.0')

    # --- Compiler flags ---
    optimization_flag = '-O0' if debug else '-O3'
    debug_symbol_flags = ['-g', '-ggdb'] if debug else []

    define_macros = []
    if variant == 'rocm':
        define_macros.extend(['-DUSE_ROCM=1', '-fgpu-rdc'])
    if enable_timer:
        define_macros.append('-DENABLE_TIMER')
    if variant == 'cuda' or rocm_disable_ctx:
        define_macros.append('-DROCM_DISABLE_CTX=1')
    if rocm_explicit_ctx and variant == 'rocm':
        define_macros.append('-DROCM_EXPLICIT_CTX=1')
    if nic_type:
        nic_macro = f'-DNIC_{nic_type.upper()}=1'
        define_macros.append(nic_macro)
        print(f'Building with NIC Macro: {nic_macro}')

    cxx_flags = [
        optimization_flag,
        '-Wno-deprecated-declarations',
        '-Wno-unused-variable',
        '-Wno-sign-compare',
        '-Wno-reorder',
        '-Wno-attributes',
    ] + debug_symbol_flags + define_macros

    if variant == 'cuda':
        nvcc_flags = [
            optimization_flag, '-Xcompiler', optimization_flag,
            '--extended-lambda',
        ] + debug_symbol_flags

        if int(os.getenv('DISABLE_SM90_FEATURES', 0)):
            cxx_flags.append('-DDISABLE_SM90_FEATURES')
            nvcc_flags.append('-DDISABLE_SM90_FEATURES')
        else:
            nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])
    else:
        nvcc_flags = [optimization_flag] + debug_symbol_flags + define_macros

    # --- Sources & includes ---
    sources = [
        'csrc/deep_ep.cpp',
        'csrc/kernels/runtime.cu',
        'csrc/kernels/layout.cu',
        'csrc/kernels/intranode.cu',
    ]
    include_dirs = ['csrc/']
    if shmem_dir is not None:
        include_dirs.append(f'{shmem_dir}/include')
    if variant == 'rocm' and ompi_dir is not None:
        include_dirs.append(f'{ompi_dir}/include')

    if variant == 'cuda' and disable_nvshmem:
        cxx_flags.append('-DDISABLE_NVSHMEM')
        nvcc_flags.append('-DDISABLE_NVSHMEM')
        assert int(os.getenv('DISABLE_SM90_FEATURES', 0)), \
            'Internode/LL kernels require NVSHMEM when SM90 features are enabled'
    else:
        sources.extend(['csrc/kernels/internode.cu', 'csrc/kernels/internode_ll.cu'])

    # --- Disable aggressive PTX instructions ---
    if variant == 'cuda':
        if os.environ.get('TORCH_CUDA_ARCH_LIST', '').strip() != '9.0':
            assert int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', 1)) == 1
            os.environ['DISABLE_AGGRESSIVE_PTX_INSTRS'] = '1'

    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '0' if variant == 'rocm' else '1')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    # --- TOPK_IDX_BITS (optional) ---
    if 'TOPK_IDX_BITS' in os.environ:
        topk_idx_bits = int(os.environ['TOPK_IDX_BITS'])
        cxx_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')
        nvcc_flags.append(f'-DTOPK_IDX_BITS={topk_idx_bits}')

    # --- Library dirs & link flags ---
    library_dirs = []
    if shmem_dir is not None:
        library_dirs.append(f'{shmem_dir}/lib')
    if variant == 'rocm' and ompi_dir is not None:
        library_dirs.append(f'{ompi_dir}/lib')

    shmem_lib_name = 'nvshmem' if variant == 'cuda' else 'rocshmem'
    nvcc_dlink = []

    if variant == 'cuda':
        if disable_nvshmem:
            extra_link_args = ['-lcuda']
        else:
            nvcc_dlink = ['-dlink', f'-L{shmem_dir}/lib', f'-l{shmem_lib_name}']
            extra_link_args = [
                '-lcuda',
                f'-l:{nvshmem_host_lib}', '-l:libnvshmem_device.a',
                '-l:nvshmem_bootstrap_uid.so',
                f'-Wl,-rpath,{shmem_dir}/lib',
            ]
    else:
        extra_link_args = [
            f'-l:lib{shmem_lib_name}.a',
            f'-Wl,-rpath,{shmem_dir}/lib',
            '-fgpu-rdc',
            '--hip-link',
            '-lamdhip64',
            '-lhsa-runtime64',
            '-libverbs',
        ]
        arch_env = os.environ["PYTORCH_ROCM_ARCH"]
        extra_link_args.extend([f"--offload-arch={arch}" for arch in arch_env.split(";")])
        if enable_mpi:
            extra_link_args.extend([
                '-l:libmpi.so',
                f'-Wl,-rpath,{ompi_dir}/lib',
            ])

    # --- Compile args ---
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if variant == 'cuda' and len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # --- Summary ---
    print('Build summary:')
    print(f' > Variant: {variant}')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    if variant == 'cuda':
        print(f' > Arch list: {os.environ.get("TORCH_CUDA_ARCH_LIST", "N/A")}')
        print(f' > NVSHMEM path: {shmem_dir}')
    else:
        print(f' > ROCm arch: {os.environ.get("PYTORCH_ROCM_ARCH", "N/A")}')
        print(f' > rocSHMEM path: {shmem_dir}')
    print()

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.2.1' + revision,
        packages=setuptools.find_packages(include=['deep_ep']),
        ext_modules=[
            CUDAExtension(
                name='deep_ep_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        cmdclass={'build_ext': BuildExtension},
    )
