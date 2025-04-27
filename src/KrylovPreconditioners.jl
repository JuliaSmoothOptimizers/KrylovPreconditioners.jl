module KrylovPreconditioners

using LinearAlgebra, SparseArrays

using Adapt
using KernelAbstractions
const KA = KernelAbstractions

using LinearAlgebra: checksquare, BlasReal, BlasFloat
import LinearAlgebra: ldiv!

# Operators
include("krylov_operators.jl")
include("triangular_operators.jl")

# Preconditioners
include("block_jacobi.jl")
include("krylov_preconditioners.jl")
include("ilu/IncompleteLU.jl")

# Scaling
include("scaling.jl")
export scaling_csr!

# Ordering
# include(ordering.jl)

export update!, get_timer, reset_timer!

end # module KrylovPreconditioners
