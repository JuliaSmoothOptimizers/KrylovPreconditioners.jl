module KrylovPreconditionersCUDAExt
using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!, mul!
import Base: size, eltype, unsafe_convert

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("CUDA/ic0.jl")
include("CUDA/ilu0.jl")
include("CUDA/blockjacobi.jl")
include("CUDA/operators.jl")

end
