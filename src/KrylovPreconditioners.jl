module KrylovPreconditioners

using LinearAlgebra, SparseArrays

using Adapt
using KernelAbstractions
const KA = KernelAbstractions

using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!

abstract type AbstractKrylovPreconditioner end
export AbstractKrylovPreconditioner

# Operators
include("krylov_operators.jl")
include("triangular_operators.jl")

# Preconditioners
include("ic0.jl")
include("ilu0.jl")
include("blockjacobi.jl")
include("ilu/IncompleteLU.jl")

# Scaling
include("scaling.jl")
export scaling_csr!

# Ordering
# include(ordering.jl)

update!(op::AbstractKrylovPreconditioner, A) = error("update!() for $(typeof(op)) is not implemented.")

export update!, get_timer, reset_timer!

function get_timer(p::AbstractKrylovPreconditioner)
    return p.timer_update
end

function reset_timer!(p::AbstractKrylovPreconditioner)
    p.timer_update = 0.0
end

end # module KrylovPreconditioners
