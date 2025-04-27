export AbstractKrylovPreconditioner
export kp_ic0, kp_ilu0

abstract type AbstractKrylovPreconditioner end

function get_timer(p::AbstractKrylovPreconditioner)
    return p.timer_update
end

function reset_timer!(p::AbstractKrylovPreconditioner)
    p.timer_update = 0.0
end

update!(P::AbstractKrylovPreconditioner, A) = error("update!() for $(typeof(P)) is not implemented.")

"""
    P = kp_ic0(A)

Construct an incomplete Cholesky preconditioner with zero fill-in -- IC(0), to accelerate Krylov solvers on GPU architectures.

The preconditioner is compatible with sparse matrices in CSR or CSC format stored on NVIDIA and AMD GPUs.

#### Input argument

* `A`: The sparse Hermitian and positive definite matrix on the GPU to factorize incompletely.

#### Output argument

* `P`: preconditioner of type `AbstractKrylovPreconditioner`.
"""
function kp_ic0 end

kp_ic0(A) = error("kp_ic0 is not implemented for $(typeof(A))")

"""
    P = kp_ic0(A)

Construct an incomplete LU preconditioner with zero fill-in -- ILU(0), to accelerate Krylov solvers on GPU architectures.

The preconditioner is compatible with sparse matrices in CSR or CSC format stored on NVIDIA and AMD GPUs.

#### Input argument

* `A`: The square sparse matrix on the GPU to factorize incompletely.

#### Output argument

* `P`: preconditioner of type `AbstractKrylovPreconditioner`.
"""
function kp_ilu0 end

kp_ilu0(A) = error("kp_ilu0 is not implemented for $(typeof(A))")

"""
    P = kp_block_jacobi(A)

Construct a block-Jacobi preconditioner to accelerate Krylov solvers on GPU architectures.

The preconditioner is compatible with sparse matrices in CSR or CSC format stored on NVIDIA, AMD, and Intel GPUs.
It also works on CPU with the CSC format.

#### Input argument

* `A`: The square sparse matrix on the CPU or GPU from which diagonal blocks are extracted.

#### Output argument

* `P`: preconditioner of type `AbstractKrylovPreconditioner`.
"""
function kp_block_jacobi end

kp_block_jacobi(A) = error("kp_block_jacobi is not implemented for $(typeof(A))")
