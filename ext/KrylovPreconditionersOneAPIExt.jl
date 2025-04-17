module KrylovPreconditionersOneAPIExt
using LinearAlgebra
using SparseArrays
using oneAPI
using oneAPI: global_queue, sycl_queue, context, device
using oneAPI.oneMKL
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!, mul!
import Base: size, eltype, unsafe_convert

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("oneAPI/blockjacobi.jl")
include("oneAPI/operators.jl")

end
