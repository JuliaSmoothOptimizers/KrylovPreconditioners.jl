module KrylovPreconditioners

using LinearAlgebra, SparseArrays

using Adapt
using KernelAbstractions
const KA = KernelAbstractions

using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!

abstract type AbstractKrylovPreconditioner end
export AbstractKrylovPreconditioner

update!(p::AbstractKrylovPreconditioner, A::AbstractMatrix) = error("update!() for $(typeof(p)) is not implemented.")
export update!, get_timer, reset_timer!

function get_timer(p::AbstractKrylovPreconditioner)
    return p.timer_update
end

function reset_timer!(p::AbstractKrylovPreconditioner)
    p.timer_update = 0.0
end

abstract type AbstractKrylovOperator{T} end
export AbstractKrylovOperator

function KrylovOperator end
export KrylovOperator

function update_operator! end
export update_operator!

# Preconditioners
include("ic0.jl")
include("ilu0.jl")
include("blockjacobi.jl")

# Scaling
# include(scaling.jl)

# Ordering
# include(ordering.jl)

end # module KrylovPreconditioners
