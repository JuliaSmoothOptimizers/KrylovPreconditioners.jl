mutable struct INTEL_KrylovOperator{T} <: AbstractKrylovOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    matrix::oneSparseMatrixCSR{T}
end

eltype(A::INTEL_KrylovOperator{T}) where T = T
size(A::INTEL_KrylovOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(oneSparseMatrixCSR{T}), :BlasFloat),)
    @eval begin
        function KP.KrylovOperator(A::$SparseMatrixType; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            if nrhs == 1
                oneMKL.sparse_optimize_gemv!(transa, A)
            else
                oneMKL.sparse_optimize_gemm!(trans, 'N', nrhs, A)
            end
            return INTEL_KrylovOperator{T}(T, m, n, nrhs, transa, A)
        end

        function KP.update!(A::INTEL_KrylovOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            error("The update of an INTEL_KrylovOperator is not supported.")
        end
    end
end

function LinearAlgebra.mul!(y::oneVector{T}, A::INTEL_KrylovOperator{T}, x::oneVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    alpha = one(T)
    beta = zero(T)
    oneMKL.sparse_gemv!(A.transa, alpha, A.matrix, x, beta, y)
end

function LinearAlgebra.mul!(Y::oneMatrix{T}, A::INTEL_KrylovOperator{T}, X::oneMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    alpha = one(T)
    beta = zero(T)
    oneMKL.sparse_gemm!(A.transa, 'N', alpha, A.matrix, X, beta, Y)
end

mutable struct INTEL_TriangularOperator{T} <: AbstractTriangularOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    uplo::Char
    diag::Char
    transa::Char
    matrix::oneSparseMatrixCSR{T}
end

eltype(A::INTEL_TriangularOperator{T}) where T = T
size(A::INTEL_TriangularOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(oneSparseMatrixCSR{T}), :BlasFloat),)
    @eval begin
        function KP.TriangularOperator(A::$SparseMatrixType, uplo::Char, diag::Char; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            if nrhs == 1
                oneMKL.sparse_optimize_trsv!(uplo, transa, diag, A)
            else
                oneMKL.sparse_optimize_trsm!(uplo, transa, diag, nrhs, A)
            end
            return INTEL_TriangularOperator{T}(T, m, n, nrhs, uplo, diag, transa, A)
        end

        function KP.update!(A::INTEL_TriangularOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            return error("The update of an INTEL_TriangularOperator is not supported.")
        end
    end
end

function LinearAlgebra.ldiv!(y::oneVector{T}, A::INTEL_TriangularOperator{T}, x::oneVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    oneMKL.sparse_trsv!(A.uplo, A.transa, A.diag, one(T), A.matrix, x, y)
end

function LinearAlgebra.ldiv!(Y::oneMatrix{T}, A::INTEL_TriangularOperator{T}, X::oneMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    oneMKL.sparse_trsm!(A.uplo, A.transa, 'N', A.diag, one(T), A.matrix, X, Y)
end
