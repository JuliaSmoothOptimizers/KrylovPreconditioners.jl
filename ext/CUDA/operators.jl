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
size(A::CUDA_KrylovOperator) = (m, n)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasFloat))
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
            return CUDA_KrylovOperator{T}(m, n, rhs, transa, descA, buffer)
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
            return CUDA_KrylovOperator{T}(m, n, rhs, transa, descA, buffer)
        end
    end

    function LinearAlgebra.mul!(y::CuVector{T}, A::KrylovOperator{T}, x::CuVector{T}) where T<: BlasFloat
        descY = CUSPARSE.CuDenseVectorDescriptor(y)
        descX = CUSPARSE.CuDenseVectorDescriptor(x)
        CUSPARSE.cusparseSpMV(CUDA.handle(), A.transa, one(T), A.descA, descX, zero(T), descY, T, A.algo, A.buffer)
    end

    function LinearAlgebra.mul!(Y::CuMatrix{T}, A::KrylovOperator{T}, X::CuMatrix{T}) where T<: BlasFloat
        descY = CUSPARSE.CuDenseVectorDescriptor(y)
        descX = CUSPARSE.CuDenseVectorDescriptor(x)
        CUSPARSE.cusparseSpMM(CUDA.handle(), A.transa, 'N', one(T), A.descA, descX, zero(T), descY, T, A.algo, A.buffer)
    end
  end
end
