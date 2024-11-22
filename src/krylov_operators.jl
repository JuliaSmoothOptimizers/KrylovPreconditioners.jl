export AbstractKrylovOperator, KrylovOperator

abstract type AbstractKrylovOperator{T} end

"""
    update!(op::AbstractKrylovOperator, A)

Update the sparse matrix `A` associated with the given `AbstractKrylovOperator` without the need to reallocate buffers 
or repeat the structural analysis phase for detecting parallelism for sparse matrix-vector or matrix-matrix products.
`A` and the operator `op` must have the same sparsity pattern, enabling efficient reuse of existing resources.

#### Input arguments

* `op`: The Krylov operator to update;
* `A`: The new sparse matrix to associate with the operator.
"""
update!(op::AbstractKrylovOperator, A) = error("update!() for $(typeof(op)) is not implemented.")

"""
    KrylovOperator(A; nrhs::Int=1, transa::Char='N')

Create a Krylov operator to accelerate sparse matrix-vector or matrix-matrix products on GPU architectures.
The operator is compatible with sparse matrices stored on NVIDIA, AMD, and Intel GPUs.

#### Input arguments

* `A`: The sparse matrix on the GPU that serves as the operator for matrix-vector or matrix-matrix products;
* `nrhs`: Specifies the number of columns for the right-hand sides. Defaults to `1` for standard matrix-vector products;
* `transa`: Determines how the matrix `A` is applied during the products; `'N'` for no transposition, `'T'` for transpose, and `'C'` for conjugate transpose.

#### Output argument

* `op`: An instance of `AbstractKrylovOperator` representing the Krylov operator for the specified sparse matrix and parameters.
"""
function KrylovOperator end
