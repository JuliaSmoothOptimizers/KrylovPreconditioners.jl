module KrylovPreconditionersCuSPARSEExt
using LinearAlgebra
using SparseArrays
using CUDACore
using cuSPARSE
const CUSPARSE = cuSPARSE
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!, mul!
import Base: size, eltype, unsafe_convert

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("CUDA/ic0.jl")
include("CUDA/ilu0.jl")
include("CUDA/operators.jl")
include("CUDA/scaling.jl")

end
