export AbstractKrylovOperator, AbstractTriangularOperator
export KrylovOperator, TriangularOperator

abstract type AbstractKrylovOperator{T} end

abstract type AbstractTriangularOperator{T} end

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
