module KrylovPreconditionersAMDGPUExt
using LinearAlgebra
using AMDGPU
using AMDGPU.rocSPARSE
using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!
using SparseArrays

using KrylovPreconditioners
const KP = KrylovPreconditioners
using KernelAbstractions
const KA = KernelAbstractions

include("AMDGPU/ic0.jl")
include("AMDGPU/ilu0.jl")
include("AMDGPU/blockjacobi.jl")

end
