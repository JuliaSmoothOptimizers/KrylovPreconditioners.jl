module KrylovPreconditioners

using LinearAlgebra, SparseArrays

using Adapt
using KernelAbstractions
const KA = KernelAbstractions

using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!

abstract type AbstractKrylovPreconditioner end
export AbstractKrylovPreconditioner

abstract type AbstractKrylovOperator{T} end
export AbstractKrylovOperator

function KrylovOperator end
export KrylovOperator

# Preconditioners
include("ic0.jl")
include("ilu0.jl")
include("blockjacobi.jl")

# Scaling
# include(scaling.jl)

# Ordering
# include(ordering.jl)

end # module KrylovPreconditioners
