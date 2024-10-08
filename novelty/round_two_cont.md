Sparse matrices (scipy.sparse)
SciPy 2-D sparse array package for numeric data.

Note

This package is switching to an array interface, compatible with NumPy arrays, from the older matrix interface. We recommend that you use the array objects (bsr_array, coo_array, etc.) for all new work.

When using the array interface, please note that:

x * y no longer performs matrix multiplication, but element-wise multiplication (just like with NumPy arrays). To make code work with both arrays and matrices, use x @ y for matrix multiplication.

Operations such as sum, that used to produce dense matrices, now produce arrays, whose multiplication behavior differs similarly.

Sparse arrays currently must be two-dimensional. This also means that all slicing operations on these objects must produce two-dimensional results, or they will result in an error. This will be addressed in a future version.

The construction utilities (eye, kron, random, diags, etc.) have not yet been ported, but their results can be wrapped into arrays:

A = csr_array(eye(3))
Contents
Sparse array classes
bsr_array(arg1[, shape, dtype, copy, blocksize])

Block Sparse Row format sparse array.

coo_array(arg1[, shape, dtype, copy])

A sparse array in COOrdinate format.

csc_array(arg1[, shape, dtype, copy])

Compressed Sparse Column array.

csr_array(arg1[, shape, dtype, copy])

Compressed Sparse Row array.

dia_array(arg1[, shape, dtype, copy])

Sparse array with DIAgonal storage.

dok_array(arg1[, shape, dtype, copy])

Dictionary Of Keys based sparse array.

lil_array(arg1[, shape, dtype, copy])

Row-based LIst of Lists sparse array.

sparray()

This class provides a base class for all sparse arrays.

Sparse matrix classes
bsr_matrix(arg1[, shape, dtype, copy, blocksize])

Block Sparse Row format sparse matrix.

coo_matrix(arg1[, shape, dtype, copy])

A sparse matrix in COOrdinate format.

csc_matrix(arg1[, shape, dtype, copy])

Compressed Sparse Column matrix.

csr_matrix(arg1[, shape, dtype, copy])

Compressed Sparse Row matrix.

dia_matrix(arg1[, shape, dtype, copy])

Sparse matrix with DIAgonal storage.

dok_matrix(arg1[, shape, dtype, copy])

Dictionary Of Keys based sparse matrix.

lil_matrix(arg1[, shape, dtype, copy])

Row-based LIst of Lists sparse matrix.

spmatrix()

This class provides a base class for all sparse matrix classes.

Functions
Building sparse arrays:

diags_array(diagonals, /, *[, offsets, ...])

Construct a sparse array from diagonals.

eye_array(m[, n, k, dtype, format])

Identity matrix in sparse array format

random_array(shape, *[, density, format, ...])

Return a sparse array of uniformly random numbers in [0, 1)

block_array(blocks, *[, format, dtype])

Build a sparse array from sparse sub-blocks

Building sparse matrices:

eye(m[, n, k, dtype, format])

Sparse matrix with ones on diagonal

identity(n[, dtype, format])

Identity matrix in sparse format

diags(diagonals[, offsets, shape, format, dtype])

Construct a sparse matrix from diagonals.

spdiags(data, diags[, m, n, format])

Return a sparse matrix from diagonals.

bmat(blocks[, format, dtype])

Build a sparse array or matrix from sparse sub-blocks

random(m, n[, density, format, dtype, ...])

Generate a sparse matrix of the given shape and density with randomly distributed values.

rand(m, n[, density, format, dtype, ...])

Generate a sparse matrix of the given shape and density with uniformly distributed values.

Building larger structures from smaller (array or matrix)

kron(A, B[, format])

kronecker product of sparse matrices A and B

kronsum(A, B[, format])

kronecker sum of square sparse matrices A and B

block_diag(mats[, format, dtype])

Build a block diagonal sparse matrix or array from provided matrices.

tril(A[, k, format])

Return the lower triangular portion of a sparse array or matrix

triu(A[, k, format])

Return the upper triangular portion of a sparse array or matrix

hstack(blocks[, format, dtype])

Stack sparse matrices horizontally (column wise)

vstack(blocks[, format, dtype])

Stack sparse arrays vertically (row wise)

Save and load sparse matrices:

save_npz(file, matrix[, compressed])

Save a sparse matrix or array to a file using .npz format.

load_npz(file)

Load a sparse array/matrix from a file using .npz format.

Sparse tools:

find(A)

Return the indices and values of the nonzero elements of a matrix

Identifying sparse arrays:

use isinstance(A, sp.sparse.sparray) to check whether an array or matrix.

use A.format == ‘csr’ to check the sparse format

Identifying sparse matrices:

issparse(x)

Is x of a sparse array or sparse matrix type?

isspmatrix(x)

Is x of a sparse matrix type?

isspmatrix_csc(x)

Is x of csc_matrix type?

isspmatrix_csr(x)

Is x of csr_matrix type?

isspmatrix_bsr(x)

Is x of a bsr_matrix type?

isspmatrix_lil(x)

Is x of lil_matrix type?

isspmatrix_dok(x)

Is x of dok_array type?

isspmatrix_coo(x)

Is x of coo_matrix type?

isspmatrix_dia(x)

Is x of dia_matrix type?

Submodules
csgraph

Compressed sparse graph routines (scipy.sparse.csgraph)

linalg

Sparse linear algebra (scipy.sparse.linalg)

Exceptions
SparseEfficiencyWarning

SparseWarning

Usage information
There are seven available sparse array types:

csc_array: Compressed Sparse Column format

csr_array: Compressed Sparse Row format

bsr_array: Block Sparse Row format

lil_array: List of Lists format

dok_array: Dictionary of Keys format

coo_array: COOrdinate format (aka IJV, triplet format)

dia_array: DIAgonal format

To construct an array efficiently, use either dok_array or lil_array. The lil_array class supports basic slicing and fancy indexing with a similar syntax to NumPy arrays. As illustrated below, the COO format may also be used to efficiently construct arrays. Despite their similarity to NumPy arrays, it is strongly discouraged to use NumPy functions directly on these arrays because NumPy may not properly convert them for computations, leading to unexpected (and incorrect) results. If you do want to apply a NumPy function to these arrays, first check if SciPy has its own implementation for the given sparse array class, or convert the sparse array to a NumPy array (e.g., using the toarray method of the class) first before applying the method.

To perform manipulations such as multiplication or inversion, first convert the array to either CSC or CSR format. The lil_array format is row-based, so conversion to CSR is efficient, whereas conversion to CSC is less so.

All conversions among the CSR, CSC, and COO formats are efficient, linear-time operations.

Matrix vector product
To do a vector product between a sparse array and a vector simply use the array dot method, as described in its docstring:

import numpy as np
from scipy.sparse import csr_array
A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
v = np.array([1, 0, -1])
A.dot(v)
array([ 1, -3, -1], dtype=int64)
Warning

As of NumPy 1.7, np.dot is not aware of sparse arrays, therefore using it will result on unexpected results or errors. The corresponding dense array should be obtained first instead:

np.dot(A.toarray(), v)
array([ 1, -3, -1], dtype=int64)
but then all the performance advantages would be lost.

The CSR format is especially suitable for fast matrix vector products.

Example 1
Construct a 1000x1000 lil_array and add some values to it:

from scipy.sparse import lil_array
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
A = lil_array((1000, 1000))
A[0, :100] = rand(100)
A.setdiag(rand(1000))
Now convert it to CSR format and solve A x = b for x:

A = A.tocsr()
b = rand(1000)
x = spsolve(A, b)
Convert it to a dense array and solve, and check that the result is the same:

x_ = solve(A.toarray(), b)
Now we can compute norm of the error with:

err = norm(x-x_)
err < 1e-10
True
It should be small :)

Example 2
Construct an array in COO format:

from scipy import sparse
from numpy import array
I = array([0,3,1,0])
J = array([0,3,1,2])
V = array([4,5,7,9])
A = sparse.coo_array((V,(I,J)),shape=(4,4))
Notice that the indices do not need to be sorted.

Duplicate (i,j) entries are summed when converting to CSR or CSC.

I = array([0,0,1,3,1,0,0])
J = array([0,2,1,3,1,0,0])
V = array([1,1,1,1,1,1,1])
B = sparse.coo_array((V,(I,J)),shape=(4,4)).tocsr()
This is useful for constructing finite-element stiffness and mass matrices.

Further details
CSR column indices are not necessarily sorted. Likewise for CSC row indices. Use the .sorted_indices() and .sort_indices() methods when sorted indices are required (e.g., when passing data to other libraries).

previous

czt_points

next

Compressed sparse graph routines (scipy.sparse.csgraph)

Python Dependencies
NumPy/SciPy-compatible API in CuPy v13 is based on NumPy 1.26 and SciPy 1.11, and has been tested against the following versions:

NumPy: v1.22 / v1.23 / v1.24 / v1.25 / v1.26 / v2.0

SciPy (optional): v1.7 / v1.8 / v1.9 / v1.10 / v1.11

Required only when copying sparse matrices from GPU to CPU (see Sparse matrices (cupyx.scipy.sparse).)

Optuna (optional): v3.x

Required only when using Automatic Kernel Parameters Optimizations (cupyx.optimizing).

Note

SciPy and Optuna are optional dependencies and will not be installed automatically.

Note

Before installing CuPy, we recommend you to upgrade setuptools and pip:

$ python -m pip install -U setuptools pip
Additional CUDA Libraries
Part of the CUDA features in CuPy will be activated only when the corresponding libraries are installed.

cuTENSOR: v2.0

The library to accelerate tensor operations. See Environment variables for the details.

NCCL: v2.16 / v2.17

The library to perform collective multi-GPU / multi-node computations.

cuDNN: v8.8

The library to accelerate deep neural network computations.

cuSPARSELt: v0.2.0

The library to accelerate sparse matrix-matrix multiplication.

Installing CuPy
Installing CuPy from PyPI
Wheels (precompiled binary packages) are available for Linux and Windows. Package names are different depending on your CUDA Toolkit version.

CUDA

Command

v11.2 ~ 11.8 (x86_64 / aarch64)

pip install cupy-cuda11x

v12.x (x86_64 / aarch64)

pip install cupy-cuda12x

Note

To enable features provided by additional CUDA libraries (cuTENSOR / NCCL / cuDNN), you need to install them manually. If you installed CuPy via wheels, you can use the installer command below to setup these libraries in case you don’t have a previous installation:

$ python -m cupyx.tools.install_library --cuda 11.x --library cutensor
Note

Append --pre -U -f https://pip.cupy.dev/pre options to install pre-releases (e.g., pip install cupy-cuda11x --pre -U -f https://pip.cupy.dev/pre).

When using wheels, please be careful not to install multiple CuPy packages at the same time. Any of these packages and cupy package (source installation) conflict with each other. Please make sure that only one CuPy package (cupy or cupy-cudaXX where XX is a CUDA version) is installed:

$ pip freeze | grep cupy
Installing CuPy from Conda-Forge
Conda is a cross-language, cross-platform package management solution widely used in scientific computing and other fields. The above pip install instruction is compatible with conda environments. Alternatively, for both Linux (x86_64, ppc64le, aarch64-sbsa) and Windows once the CUDA driver is correctly set up, you can also install CuPy from the conda-forge channel:

$ conda install -c conda-forge cupy
and conda will install a pre-built CuPy binary package for you, along with the CUDA runtime libraries (cudatoolkit for CUDA 11 and below, or cuda-XXXXX for CUDA 12 and above). It is not necessary to install CUDA Toolkit in advance.

If you aim at minimizing the installation footprint, you can install the cupy-core package:

$ conda install -c conda-forge cupy-core
which only depends on numpy. None of the CUDA libraries will be installed this way, and it is your responsibility to install the needed dependencies yourself, either from conda-forge or elsewhere. This is equivalent of the cupy-cudaXX wheel installation.

Conda has a built-in mechanism to determine and install the latest version of cudatoolkit or any other CUDA components supported by your driver. However, if for any reason you need to force-install a particular CUDA version (say 11.8), you can do:

$ conda install -c conda-forge cupy cuda-version=11.8
Note

cuDNN, cuTENSOR, and NCCL are available on conda-forge as optional dependencies. The following command can install them all at once:

$ conda install -c conda-forge cupy cudnn cutensor nccl
Each of them can also be installed separately as needed.

Note

If you encounter any problem with CuPy installed from conda-forge, please feel free to report to cupy-feedstock, and we will help investigate if it is just a packaging issue in conda-forge’s recipe or a real issue in CuPy.

Note

If you did not install CUDA Toolkit by yourself, for CUDA 11 and below the nvcc compiler might not be available, as the cudatoolkit package from conda-forge does not include the nvcc compiler toolchain. If you would like to use it from a local CUDA installation, you need to make sure the version of CUDA Toolkit matches that of cudatoolkit to avoid surprises. For CUDA 12 and above, nvcc can be installed on a per-conda environment basis via

$ conda install -c conda-forge cuda-nvcc

Installing CuPy from Source
Use of wheel packages is recommended whenever possible. However, if wheels cannot meet your requirements (e.g., you are running non-Linux environment or want to use a version of CUDA / cuDNN / NCCL not supported by wheels), you can also build CuPy from source.

Note

CuPy source build requires g++-6 or later. For Ubuntu 18.04, run apt-get install g++. For Ubuntu 16.04, CentOS 6 or 7, follow the instructions here.

Note

When installing CuPy from source, features provided by additional CUDA libraries will be disabled if these libraries are not available at the build time. See Installing cuDNN and NCCL for the instructions.

Note

If you upgrade or downgrade the version of CUDA Toolkit, cuDNN, NCCL or cuTENSOR, you may need to reinstall CuPy. See Reinstalling CuPy for details.

You can install the latest stable release version of the CuPy source package via pip.

$ pip install cupy
If you want to install the latest development version of CuPy from a cloned Git repository:

$ git clone --recursive https://github.com/cupy/cupy.git
$ cd cupy
$ pip install .
Note

Cython 0.29.22 or later is required to build CuPy from source. It will be automatically installed during the build process if not available.

Uninstalling CuPy
Use pip to uninstall CuPy:

$ pip uninstall cupy
Note

If you are using a wheel, cupy shall be replaced with cupy-cudaXX (where XX is a CUDA version number).

Note

If CuPy is installed via conda, please do conda uninstall cupy instead.

Upgrading CuPy
Just use pip install with -U option:

$ pip install -U cupy
Note

If you are using a wheel, cupy shall be replaced with cupy-cudaXX (where XX is a CUDA version number).

Reinstalling CuPy
To reinstall CuPy, please uninstall CuPy and then install it. When reinstalling CuPy, we recommend using --no-cache-dir option as pip caches the previously built binaries:

$ pip uninstall cupy
$ pip install cupy --no-cache-dir
Note

If you are using a wheel, cupy shall be replaced with cupy-cudaXX (where XX is a CUDA version number).

Using CuPy inside Docker
We are providing the official Docker images. Use NVIDIA Container Toolkit to run CuPy image with GPU. You can login to the environment with bash, and run the Python interpreter:

$ docker run --gpus all -it cupy/cupy /bin/bash
Or run the interpreter directly:

$ docker run --gpus all -it cupy/cupy /usr/bin/python3
FAQ
pip fails to install CuPy
Please make sure that you are using the latest setuptools and pip:

$ pip install -U setuptools pip
Use -vvvv option with pip command. This will display all logs of installation:

$ pip install cupy -vvvv
If you are using sudo to install CuPy, note that sudo command does not propagate environment variables. If you need to pass environment variable (e.g., CUDA_PATH), you need to specify them inside sudo like this:

$ sudo CUDA_PATH=/opt/nvidia/cuda pip install cupy
If you are using certain versions of conda, it may fail to build CuPy with error g++: error: unrecognized command line option ‘-R’. This is due to a bug in conda (see conda/conda#6030 for details). If you encounter this problem, please upgrade your conda.

Installing cuDNN and NCCL
We recommend installing cuDNN and NCCL using binary packages (i.e., using apt or yum) provided by NVIDIA.

If you want to install tar-gz version of cuDNN and NCCL, we recommend installing it under the CUDA_PATH directory. For example, if you are using Ubuntu, copy *.h files to include directory and *.so* files to lib64 directory:

$ cp /path/to/cudnn.h $CUDA_PATH/include
$ cp /path/to/libcudnn.so* $CUDA_PATH/lib64
The destination directories depend on your environment.

If you want to use cuDNN or NCCL installed in another directory, please use CFLAGS, LDFLAGS and LD_LIBRARY_PATH environment variables before installing CuPy:

$ export CFLAGS=-I/path/to/cudnn/include
$ export LDFLAGS=-L/path/to/cudnn/lib
$ export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
Working with Custom CUDA Installation
If you have installed CUDA on the non-default directory or multiple CUDA versions on the same host, you may need to manually specify the CUDA installation directory to be used by CuPy.

CuPy uses the first CUDA installation directory found by the following order.

CUDA_PATH environment variable.

The parent directory of nvcc command. CuPy looks for nvcc command from PATH environment variable.

/usr/local/cuda

For example, you can build CuPy using non-default CUDA directory by CUDA_PATH environment variable:

$ CUDA_PATH=/opt/nvidia/cuda pip install cupy
Note

CUDA installation discovery is also performed at runtime using the rule above. Depending on your system configuration, you may also need to set LD_LIBRARY_PATH environment variable to $CUDA_PATH/lib64 at runtime.

CuPy always raises cupy.cuda.compiler.CompileException
If CuPy raises a CompileException for almost everything, it is possible that CuPy cannot detect CUDA installed on your system correctly. The following are error messages commonly observed in such cases.

nvrtc: error: failed to load builtins

catastrophic error: cannot open source file "cuda_fp16.h"

error: cannot overload functions distinguished by return type alone

error: identifier "__half_raw" is undefined

Please try setting LD_LIBRARY_PATH and CUDA_PATH environment variable. For example, if you have CUDA installed at /usr/local/cuda-9.2:

$ export CUDA_PATH=/usr/local/cuda-9.2
$ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
Also see Working with Custom CUDA Installation.

Build fails on Ubuntu 16.04, CentOS 6 or 7
In order to build CuPy from source on systems with legacy GCC (g++-5 or earlier), you need to manually set up g++-6 or later and configure NVCC environment variable.

On Ubuntu 16.04:

$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
$ sudo apt update
$ sudo apt install g++-6
$ export NVCC="nvcc --compiler-bindir gcc-6"
On CentOS 6 / 7:

$ sudo yum install centos-release-scl
$ sudo yum install devtoolset-7-gcc-c++
$ source /opt/rh/devtoolset-7/enable
$ export NVCC="nvcc --compiler-bindir gcc"
Using CuPy on AMD GPU (experimental)
CuPy has an experimental support for AMD GPU (ROCm).

Requirements
AMD GPU supported by ROCm

ROCm: v4.3 / v5.0
See the ROCm Installation Guide for details.

The following ROCm libraries are required:

$ sudo apt install hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl
Environment Variables
When building or running CuPy for ROCm, the following environment variables are effective.

ROCM_HOME: directory containing the ROCm software (e.g., /opt/rocm).

Docker
You can try running CuPy for ROCm using Docker.

$ docker run -it --device=/dev/kfd --device=/dev/dri --group-add video cupy/cupy-rocm
Installing Binary Packages
Wheels (precompiled binary packages) are available for Linux (x86_64). Package names are different depending on your ROCm version.

ROCm

Command

v4.3

$ pip install cupy-rocm-4-3

v5.0

$ pip install cupy-rocm-5-0

Building CuPy for ROCm From Source
To build CuPy from source, set the CUPY_INSTALL_USE_HIP, ROCM_HOME, and HCC_AMDGPU_TARGET environment variables. (HCC_AMDGPU_TARGET is the ISA name supported by your GPU. Run rocminfo and use the value displayed in Name: line (e.g., gfx900). You can specify a comma-separated list of ISAs if you have multiple GPUs of different architectures.)

$ export CUPY_INSTALL_USE_HIP=1
$ export ROCM_HOME=/opt/rocm
$ export HCC_AMDGPU_TARGET=gfx906
$ pip install cupy
Note

If you don’t specify the HCC_AMDGPU_TARGET environment variable, CuPy will be built for the GPU architectures available on the build host. This behavior is specific to ROCm builds; when building CuPy for NVIDIA CUDA, the build result is not affected by the host configuration.

Limitations
The following features are not available due to the limitation of ROCm or because that they are specific to CUDA:

CUDA Array Interface

cuTENSOR

Handling extremely large arrays whose size is around 32-bit boundary (HIP is known to fail with sizes 2**32-1024)

Atomic addition in FP16 (cupy.ndarray.scatter_add and cupyx.scatter_add)

Multi-GPU FFT and FFT callback

Some random number generation algorithms

Several options in RawKernel/RawModule APIs: Jitify, dynamic parallelism

Per-thread default stream

The following features are not yet supported:

Sparse matrices (cupyx.scipy.sparse)

cuDNN (hipDNN)

Hermitian/symmetric eigenvalue solver (cupy.linalg.eigh)

Polynomial roots (uses Hermitian/symmetric eigenvalue solver)

Splines in cupyx.scipy.interpolate (make_interp_spline, spline modes of RegularGridInterpolator/interpn), as they depend on sparse matrices.

The following features may not work in edge cases (e.g., some combinations of dtype):

Note

We are investigating the root causes of the issues. They are not necessarily CuPy’s issues, but ROCm may have some potential bugs.

cupy.ndarray.__getitem__ (#4653)

cupy.ix_ (#4654)

Some polynomial routines (#4758, #4759)

cupy.broadcast (#4662)

cupy.convolve (#4668)

cupy.correlate (#4781)

Some random sampling routines (cupy.random, #4770)

cupy.linalg.einsum

cupyx.scipy.ndimage and cupyx.scipy.signal (#4878, #4879, #4880)

previous

Overview

next

User Guide

 On this page
Installation
Requirements
Installing CuPy
Uninstalling CuPy
Upgrading CuPy
Reinstalling CuPy
Using CuPy inside Docker
FAQ
Using CuPy on AMD GPU (experimental)
Requirements
Environment Variables
Docker
Installing Binary Packages
Building CuPy for ROCm From Source
Limitations
