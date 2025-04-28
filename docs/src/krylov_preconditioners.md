# [Preconditioners](@id preconditioners)

```@docs
kp_ic0
kp_ilu0
kp_block_jacobi
```

## Nvidia GPUs

```julia
using CUDA, CUDA.CUSPARSE
using SparseArrays
using KrylovPreconditioners

if CUDA.functional()
  # CPU Arrays
  A_cpu = sprand(100, 100, 0.3) + I

  # GPU Arrays
  A_csc_gpu = CuSparseMatrixCSC(A_cpu)
  A_csr_gpu = CuSparseMatrixCSR(A_cpu)

  # Krylov operators
  P_csc = kp_ilu0(A_csc_gpu)
  P_csr = kp_ilu0(A_csr_gpu)
end
```

## AMD GPUs

```julia
using AMDGPU, AMDGPU.rocSPARSE
using SparseArrays
using KrylovPreconditioners

if AMDGPU.functional()
  # CPU Arrays
  A_cpu = sprand(200, 200, 0.05)
  A_cpu = A_cpu' * A_cpu + I

  # GPU Arrays
  A_csc_gpu = ROCSparseMatrixCSC(A_cpu)
  A_csr_gpu = ROCSparseMatrixCSR(A_cpu)

  # Krylov operators
  P_csc = kp_ic0(A_csc_gpu)
  P_csr = kp_ic0(A_csr_gpu)
end
```
