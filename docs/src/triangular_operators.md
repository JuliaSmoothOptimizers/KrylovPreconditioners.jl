# [Triangular operators](@id triangular_operators)

```@docs
TriangularOperator
update!(::AbstractTriangularOperator, ::Any)
```

## Nvidia GPUs

Sparse matrices have a specific storage on Nvidia GPUs (`CuSparseMatrixCSC`, `CuSparseMatrixCSR` or `CuSparseMatrixCOO`):

```julia
using CUDA, CUDA.CUSPARSE
using SparseArrays
using KrylovPreconditioners

if CUDA.functional()
  # CPU Arrays
  A_cpu = sprand(100, 100, 0.3)

  # GPU Arrays
  A_csc_gpu = CuSparseMatrixCSC(A_cpu)
  A_csr_gpu = CuSparseMatrixCSR(A_cpu)
  A_coo_gpu = CuSparseMatrixCOO(A_cpu)

  # Triangular operators
  op_csc = TriangularOperator(A_csc_gpu; uplo='L', diag='U', nrhs=1, transa='N')
  op_csr = TriangularOperator(A_csr_gpu; uplo='U', diag='N', nrhs=1, transa='T')
  op_coo = TriangularOperator(A_coo_gpu; uplo='L', diag='N', nrhs=5, transa='N')
end
```

## AMD GPUs

Sparse matrices have a specific storage on AMD GPUs (`ROCSparseMatrixCSC`, `ROCSparseMatrixCSR` or `ROCSparseMatrixCOO`):

```julia
using AMDGPU, AMDGPU.rocSPARSE
using SparseArrays
using KrylovPreconditioners

if AMDGPU.functional()
  # CPU Arrays
  A_cpu = sprand(200, 100, 0.3)

  # GPU Arrays
  A_csc_gpu = ROCSparseMatrixCSC(A_cpu)
  A_csr_gpu = ROCSparseMatrixCSR(A_cpu)
  A_coo_gpu = ROCSparseMatrixCOO(A_cpu)

  # Triangular operators
  op_csc = TriangularOperator(A_csc_gpu; uplo='L', diag='U', nrhs=1, transa='N')
  op_csr = TriangularOperator(A_csr_gpu; uplo='L', diag='U', nrhs=1, transa='T')
  op_coo = TriangularOperator(A_coo_gpu; uplo='L', diag='U', nrhs=5, transa='N')
end
```

## Intel GPUs

Sparse matrices have a specific storage on Intel GPUs (`oneSparseMatrixCSR`):

```julia
using oneAPI, oneAPI.oneMKL
using SparseArrays
using KrylovPreconditioners

if oneAPI.functional()
  # CPU Arrays
  A_cpu = sprand(T, 20, 10, 0.3)

  # GPU Arrays
  A_csr_gpu = oneSparseMatrixCSR(A_cpu)

  # Triangular operator
  op_csr = TriangularOperator(A_csr_gpu; uplo='L', diag='U', nrhs=1, transa='N')
end
```
