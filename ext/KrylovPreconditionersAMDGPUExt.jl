module KrylovPreconditionersAMDGPUExt
using LinearAlgebra
using SparseArrays
using AMDGPU
using AMDGPU.rocSPARSE, AMDGPU.rocSOLVER
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!, mul!
import Base: size, eltype, unsafe_convert

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("AMDGPU/ic0.jl")
include("AMDGPU/ilu0.jl")
include("AMDGPU/blockjacobi.jl")
include("AMDGPU/operators.jl")
include("AMDGPU/scaling.jl")

end
