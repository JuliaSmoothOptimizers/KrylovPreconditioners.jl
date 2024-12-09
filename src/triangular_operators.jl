export AbstractTriangularOperator, TriangularOperator

abstract type AbstractTriangularOperator{T} end

"""
    update!(op::AbstractTriangularOperator, A)

Update the sparse matrix `A` associated with the given `AbstractTriangularOperator` without the need to reallocate buffers 
or repeat the structural analysis phase for detecting parallelism for sparse triangular solves.
`A` and the operator `op` must have the same sparsity pattern, enabling efficient reuse of existing resources.

#### Input arguments

* `op`: The triangular operator to update;
* `A`: The new sparse matrix to associate with the operator.
"""
update!(op::AbstractTriangularOperator, A) = error("update!() for $(typeof(op)) is not implemented.")

"""
    TriangularOperator(A, uplo::Char, diag::Char; nrhs::Int=1, transa::Char='N')

Create a triangular operator for efficient solution of sparse triangular systems on GPU architectures. 
Supports sparse matrices stored on NVIDIA, AMD, and Intel GPUs.

#### Input arguments

* `A`: A sparse matrix on the GPU representing the triangular system to be solved;
* `uplo`: Specifies whether the triangular matrix `A` is upper triangular (`'U'`) or lower triangular (`'L'`);
* `diag`: Indicates whether the diagonal is unit (`'U'`) or non-unit (`'N'`);
* `nrhs`: Specifies the number of columns for the right-hand side(s). Defaults to 1, corresponding to solving triangular systems with a single vector as the right-hand side;
* `transa`: Determines how the matrix `A` is applied during the triangle solves; `'N'` for no transposition, `'T'` for transpose, and `'C'` for conjugate transpose.

#### Output argument

* `op`: An instance of `AbstractTriangularOperator` representing the triangular operator for the specified sparse matrix and parameters.
"""
function TriangularOperator end
