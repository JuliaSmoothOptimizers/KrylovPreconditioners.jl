module KrylovPreconditionersoneAPIExt
using LinearAlgebra
using SparseArrays
using oneAPI
using oneAPI.oneMKL
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!, mul!
import Base: size, eltype, unsafe_convert

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("CUDA/blockjacobi.jl")

end
