mutable struct NVIDIA_KrylovOperator{T} <: AbstractKrylovOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::cuSPARSE.CuSparseMatrixDescriptor
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
            descA = cuSPARSE.CuSparseMatrixDescriptor(A, 'O')
            if nrhs == 1
                descX = cuSPARSE.CuDenseVectorDescriptor(T, n)
                descY = cuSPARSE.CuDenseVectorDescriptor(T, m)
                algo = cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                cuSPARSE.cusparseSpMV_bufferSize(cuSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                if cuSPARSE.version() ≥ v"12.3"
                    cuSPARSE.cusparseSpMV_preprocess(cuSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer)
                end
                return NVIDIA_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer)
            else
                descX = cuSPARSE.CuDenseMatrixDescriptor(T, n, nrhs)
                descY = cuSPARSE.CuDenseMatrixDescriptor(T, m, nrhs)
                algo = cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                cuSPARSE.cusparseSpMM_bufferSize(cuSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                if !(A isa CuSparseMatrixCOO)
                    cuSPARSE.cusparseSpMM_preprocess(cuSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T, algo, buffer)
                end
                return NVIDIA_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer)
            end
        end

        function KP.update!(A::NVIDIA_KrylovOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            descB = cuSPARSE.CuSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            return A
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::NVIDIA_KrylovOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = cuSPARSE.CuDenseVectorDescriptor(y)
    descX = cuSPARSE.CuDenseVectorDescriptor(x)
    algo = cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    cuSPARSE.cusparseSpMV(cuSPARSE.handle(), A.transa, alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end

function LinearAlgebra.mul!(Y::CuMatrix{T}, A::NVIDIA_KrylovOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = cuSPARSE.CuDenseMatrixDescriptor(Y)
    descX = cuSPARSE.CuDenseMatrixDescriptor(X)
    algo = cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    cuSPARSE.cusparseSpMM(cuSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end

mutable struct NVIDIA_TriangularOperator{T,S} <: AbstractTriangularOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::cuSPARSE.CuSparseMatrixDescriptor
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
            descA = cuSPARSE.CuSparseMatrixDescriptor(A, 'O')
            cusparse_uplo = Ref{cuSPARSE.cusparseFillMode_t}(uplo)
            cusparse_diag = Ref{cuSPARSE.cusparseDiagType_t}(diag)
            cuSPARSE.cusparseSpMatSetAttribute(descA, 'F', cusparse_uplo, Csize_t(sizeof(cusparse_uplo)))
            cuSPARSE.cusparseSpMatSetAttribute(descA, 'D', cusparse_diag, Csize_t(sizeof(cusparse_diag)))
            if nrhs == 1
                descT = cuSPARSE.CuSparseSpSVDescriptor()
                descX = cuSPARSE.CuDenseVectorDescriptor(T, n)
                descY = cuSPARSE.CuDenseVectorDescriptor(T, m)
                algo = cuSPARSE.CUSPARSE_SPSV_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                cuSPARSE.cusparseSpSV_bufferSize(cuSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo, descT, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                cuSPARSE.cusparseSpSV_analysis(cuSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo, descT, buffer)
                return NVIDIA_TriangularOperator{T,cuSPARSE.CuSparseSpSVDescriptor}(T, m, n, nrhs, transa, descA, descT, buffer)
            else
                descT = cuSPARSE.CuSparseSpSMDescriptor()
                descX = cuSPARSE.CuDenseMatrixDescriptor(T, n, nrhs)
                descY = cuSPARSE.CuDenseMatrixDescriptor(T, m, nrhs)
                algo = cuSPARSE.CUSPARSE_SPSM_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                cuSPARSE.cusparseSpSM_bufferSize(cuSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo, descT, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size[])
                cuSPARSE.cusparseSpSM_analysis(cuSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo, descT, buffer)
                return NVIDIA_TriangularOperator{T,cuSPARSE.CuSparseSpSMDescriptor}(T, m, n, nrhs, transa, descA, descT, buffer)
            end
        end

        function KP.update!(A::NVIDIA_TriangularOperator{T,cuSPARSE.CuSparseSpSVDescriptor}, B::$SparseMatrixType) where T <: $BlasFloat
            cuSPARSE.version() ≥ v"12.2" || error("This operation is only supported by CUDA ≥ v12.3")
            descB = cuSPARSE.CuSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            cuSPARSE.cusparseSpSV_updateMatrix(cuSPARSE.handle(), A.descT, B.nzVal, 'G')
            return A
        end

        function KP.update!(A::NVIDIA_TriangularOperator{T,cuSPARSE.CuSparseSpSMDescriptor}, B::$SparseMatrixType) where T <: $BlasFloat
            cuSPARSE.version() ≥ v"12.3" || error("This operation is only supported by CUDA ≥ v12.4")
            descB = cuSPARSE.CuSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            cuSPARSE.cusparseSpSM_updateMatrix(cuSPARSE.handle(), A.descT, B.nzVal, 'G')
            return A
        end
    end
end

function LinearAlgebra.ldiv!(y::CuVector{T}, A::NVIDIA_TriangularOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = cuSPARSE.CuDenseVectorDescriptor(y)
    descX = cuSPARSE.CuDenseVectorDescriptor(x)
    algo = cuSPARSE.CUSPARSE_SPSV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    cuSPARSE.cusparseSpSV_solve(cuSPARSE.handle(), A.transa, alpha, A.descA, descX, descY, T, algo, A.descT)
end

function LinearAlgebra.ldiv!(Y::CuMatrix{T}, A::NVIDIA_TriangularOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = cuSPARSE.CuDenseMatrixDescriptor(Y)
    descX = cuSPARSE.CuDenseMatrixDescriptor(X)
    algo = cuSPARSE.CUSPARSE_SPSM_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    cuSPARSE.cusparseSpSM_solve(cuSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX, descY, T, algo, A.descT)
end
