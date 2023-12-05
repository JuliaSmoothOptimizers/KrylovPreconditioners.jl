mutable struct NVIDIA_KrylovOperator{T} <: AbstractKrylovOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
end

eltype(A::NVIDIA_KrylovOperator{T}) where T = T
size(A::NVIDIA_KrylovOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T}), :BlasFloat),
                                     (:(CuSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function KP.KrylovOperator(A::$SparseMatrixType; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
            if nrhs == 1
                descX = CUSPARSE.CuDenseVectorDescriptor(T, n)
                descY = CUSPARSE.CuDenseVectorDescriptor(T, m)
                algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                return NVIDIA_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer)
            else
                descX = CUSPARSE.CuDenseMatrixDescriptor(T, n, nrhs)
                descY = CUSPARSE.CuDenseMatrixDescriptor(T, m, nrhs)
                algo = CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                CUSPARSE.cusparseSpMM_bufferSize(CUSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                if !(A isa CuSparseMatrixCOO)
                    CUSPARSE.cusparseSpMM_preprocess(CUSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T, algo, buffer)
                end
                return NVIDIA_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer)
            end
        end

        function KP.update!(A::NVIDIA_KrylovOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            descB = CUSPARSE.CuSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            return A
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::NVIDIA_KrylovOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    CUSPARSE.cusparseSpMV(CUSPARSE.handle(), A.transa, alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end

function LinearAlgebra.mul!(Y::CuMatrix{T}, A::NVIDIA_KrylovOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = CUSPARSE.CuDenseMatrixDescriptor(Y)
    descX = CUSPARSE.CuDenseMatrixDescriptor(X)
    algo = CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    CUSPARSE.cusparseSpMM(CUSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end

mutable struct NVIDIA_TriangularOperator{T,S} <: AbstractTriangularOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    descT::S
    buffer::CuVector{UInt8}
end

eltype(A::NVIDIA_TriangularOperator{T}) where T = T
size(A::NVIDIA_TriangularOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T}), :BlasFloat),
                                     (:(CuSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function KP.TriangularOperator(A::$SparseMatrixType, uplo::Char, diag::Char; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            alpha = Ref{T}(one(T))
            descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
            cusparse_uplo = Ref{CUSPARSE.cusparseFillMode_t}(uplo)
            cusparse_diag = Ref{CUSPARSE.cusparseDiagType_t}(diag)
            CUSPARSE.cusparseSpMatSetAttribute(descA, 'F', cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
            CUSPARSE.cusparseSpMatSetAttribute(descA, 'D', cusparse_diag, Csize_t(sizeof(cusparse_diag)))
            if nrhs == 1
                descT = CUSPARSE.CuSparseSpSVDescriptor()
                descX = CUSPARSE.CuDenseVectorDescriptor(T, n)
                descY = CUSPARSE.CuDenseVectorDescriptor(T, m)
                algo = CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                CUSPARSE.cusparseSpSV_bufferSize(CUSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo, descT, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                CUSPARSE.cusparseSpSV_analysis(CUSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo, descT, buffer)
                return NVIDIA_TriangularOperator{T,CUSPARSE.CuSparseSpSVDescriptor}(T, m, n, nrhs, transa, descA, descT, buffer)
            else
                descT = CUSPARSE.CuSparseSpSMDescriptor()
                descX = CUSPARSE.CuDenseMatrixDescriptor(T, n, nrhs)
                descY = CUSPARSE.CuDenseMatrixDescriptor(T, m, nrhs)
                algo = CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                CUSPARSE.cusparseSpSM_bufferSize(CUSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo, descT, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                CUSPARSE.cusparseSpSM_analysis(CUSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo, descT, buffer)
                return NVIDIA_TriangularOperator{T,CUSPARSE.CuSparseSpSMDescriptor}(T, m, n, nrhs, transa, descA, descT, buffer)
            end
        end

        function KP.update!(A::NVIDIA_TriangularOperator{T,CUSPARSE.CuSparseSpSVDescriptor}, B::$SparseMatrixType) where T <: $BlasFloat
            CUSPARSE.version() ≥ v"12.2" || error("This operation is only support by CUDA ≥ v12.3")
            descB = CUSPARSE.CuSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            CUSPARSE.cusparseSpSV_updateMatrix(CUSPARSE.handle(), A.descT, B.nzVal, 'G')
            return A
        end

        function KP.update!(A::NVIDIA_TriangularOperator{T,CUSPARSE.CuSparseSpSMDescriptor}, B::$SparseMatrixType) where T <: $BlasFloat
            return error("This operation will be supported in CUDA v12.4")
        end
    end
end

function LinearAlgebra.ldiv!(y::CuVector{T}, A::NVIDIA_TriangularOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    algo = CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    CUSPARSE.cusparseSpSV_solve(CUSPARSE.handle(), A.transa, alpha, A.descA, descX, descY, T, algo, A.descT)
end

function LinearAlgebra.ldiv!(Y::CuMatrix{T}, A::NVIDIA_TriangularOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = CUSPARSE.CuDenseMatrixDescriptor(Y)
    descX = CUSPARSE.CuDenseMatrixDescriptor(X)
    algo = CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    CUSPARSE.cusparseSpSM_solve(CUSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX, descY, T, algo, A.descT)
end
