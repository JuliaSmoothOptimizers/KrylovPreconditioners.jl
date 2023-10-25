module KrylovPreconditioners

using LinearAlgebra, SparseArrays

using Adapt
using KernelAbstractions
using AMDGPU, AMDGPU.rocSPARSE
using CUDA, CUDA.CUSPARSE
const KA = KernelAbstractions

using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!

abstract type AbstractKrylovPreconditioner end

# Preconditioners
include("ic0.jl")
include("ilu0.jl")
include("blockjacobi.jl")

# Scaling
# include(scaling.jl)

# Ordering
# include(ordering.jl)

end # module KrylovPreconditioners
