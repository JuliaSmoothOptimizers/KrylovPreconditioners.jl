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

abstract type AbstractTriangularOperator{T} end
export AbstractTriangularOperator

update!(p::AbstractKrylovPreconditioner, A::SparseMatrixCSC) = error("update!() for $(typeof(p)) is not implemented.")
update!(p::AbstractKrylovPreconditioner, A) = error("update!() for $(typeof(p)) is not implemented.")
update!(p::AbstractKrylovOperator, A::SparseMatrixCSC) = error("update!() for $(typeof(p)) is not implemented.")
update!(p::AbstractKrylovOperator, A) = error("update!() for $(typeof(p)) is not implemented.")

export update!, get_timer, reset_timer!

function get_timer(p::AbstractKrylovPreconditioner)
    return p.timer_update
end

function reset_timer!(p::AbstractKrylovPreconditioner)
    p.timer_update = 0.0
end

function KrylovOperator end
export KrylovOperator

function TriangularOperator end
export TriangularOperator

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

end # module KrylovPreconditioners
