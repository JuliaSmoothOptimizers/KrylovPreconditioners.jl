module KrylovPreconditionersCuBLASExt
using LinearAlgebra
using SparseArrays
using CUDACore
using CUDACore: @sync
using cuSPARSE
using cuBLAS
const CUSPARSE = cuSPARSE
const CUBLAS = cuBLAS

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("CUDA/block_jacobi.jl")

end
