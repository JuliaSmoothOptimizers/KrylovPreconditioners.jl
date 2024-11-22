# [Krylov operators](@id krylov_operators)

```@docs
KrylovOperator
```

## Nvidia GPUs

Sparse matrices have a specific storage on Nvidia GPUs (`CuSparseMatrixCSC`, `CuSparseMatrixCSR` or `CuSparseMatrixCOO`):

```julia
using CUDA, CUDA.CUSPARSE
using SparseArrays
using KrylovPreconditioners

if CUDA.functional()
  # CPU Arrays
  A_cpu = sprand(200, 100, 0.3)

  # GPU Arrays
  A_csc_gpu = CuSparseMatrixCSC(A_cpu)
  A_csr_gpu = CuSparseMatrixCSR(A_cpu)
  A_coo_gpu = CuSparseMatrixCOO(A_cpu)

  # Krylov operators
  op_csc = KrylovOperator(A_csc_gpu; nrhs=1, transa='N')
  op_csr = KrylovOperator(A_csr_gpu; nrhs=1, transa='T')
  op_coo = KrylovOperator(A_coo_gpu; nrhs=5, transa='N')
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

  # Krylov operators
  op_csc = KrylovOperator(A_csc_gpu; nrhs=1, transa='N')
  op_csr = KrylovOperator(A_csr_gpu; nrhs=1, transa='T')
  op_coo = KrylovOperator(A_coo_gpu; nrhs=5, transa='N')
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
  A_cpu = sprand(Float32, 20, 10, 0.3)

  # GPU Arrays
  A_csr_gpu = oneSparseMatrixCSR(A_cpu)

  # Krylov operator
  op_csr = KrylovOperator(A_csr_gpu; nrhs=1, transa='N')
end
```
