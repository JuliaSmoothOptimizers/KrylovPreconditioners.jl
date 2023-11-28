mutable struct CUDA_KrylovOperator{T}
    m::Int
    n::Int
    nrhs::Int
    type::T
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
end

eltype(A::CUDA_KrylovOperator{T}) where T = T
size(A::CUDA_KrylovOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T}), :BlasFloat))
    @eval begin
        function KP.KrylovOperator(A::$SparseMatrixType; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            if nrhs == 1
                alpha = Ref{T}(one(T))
                beta = Ref{T}(zero(T))
                descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
                descX = CUSPARSE.CuDenseVectorDescriptor(T, n)
                descY = CUSPARSE.CuDenseVectorDescriptor(T, m)
                algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                CUSPARSE.cusparseSpMV_bufferSize(CUDA.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size)
                return CUDA_KrylovOperator{T}(m, n, nrhs, transa, descA, buffer)
            else
                alpha = Ref{T}(one(T))
                beta = Ref{T}(zeto(T))
                descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
                descX = CUSPARSE.CuDenseMatrixDescriptor(T, n, nrhs)
                descY = CUSPARSE.CuDenseMatrixDescriptor(T, m, nrhs)
                algo = CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
                buffer_size = Ref{Csize_t}()
                transb = 'N'
                CUSPARSE.cusparseSpMM_bufferSize(CUDA.handle(), transa, transb, alpha, descA, descX, beta, descY, T, algo, buffer_size)
                buffer = CuVector{UInt8}(undef, buffer_size)
                CUSPARSE.cusparseSpMM_preprocess(CUDA.handle(), transa, transb, alpha, descA, descX, beta, descY, T, algo, buffer)
                return CUDA_KrylovOperator{T}(m, n, nrhs, transa, descA, buffer)
            end
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::CUDA_KrylovOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch(""))
    (length(x) != A.n) && throw(DimensionMismatch(""))
    (A.nrhs != 1) && throw(DimensionMismatch(""))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    CUSPARSE.cusparseSpMV(CUDA.handle(), A.transa, one(T), A.descA, descX, zero(T), descY, T, A.algo, A.buffer)
end

function LinearAlgebra.mul!(Y::CuMatrix{T}, A::CUDA_KrylovOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch(""))
    (mX != A.n) && throw(DimensionMismatch(""))
    (nY == nX == A.nrhs) || throw(DimensionMismatch(""))
    descY = CUSPARSE.CuDenseVectorDescriptor(Y)
    descX = CUSPARSE.CuDenseVectorDescriptor(X)
    CUSPARSE.cusparseSpMM(CUDA.handle(), A.transa, 'N', one(T), A.descA, descX, zero(T), descY, T, A.algo, A.buffer)
end
