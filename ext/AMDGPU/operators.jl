mutable struct AMD_KrylovOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::rocSPARSE.ROCSparseMatrixDescriptor
    buffer_size::Ref{Csize_t}
    buffer::ROCVector{UInt8}
end

eltype(A::AMD_KrylovOperator{T}) where T = T
size(A::AMD_KrylovOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T}), :BlasFloat),
                                     (:(ROCSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function KP.KrylovOperator(A::$SparseMatrixType; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            if nrhs == 1
                alpha = Ref{T}(one(T))
                beta = Ref{T}(zero(T))
                descA = rocSPARSE.ROCSparseMatrixDescriptor(A, 'O')
                descX = rocSPARSE.ROCDenseVectorDescriptor(T, n)
                descY = rocSPARSE.ROCDenseVectorDescriptor(T, m)
                algo = rocSPARSE.rocSPARSE.rocsparse_spmv_alg_default
                buffer_size = Ref{Csize_t}()
                rocSPARSE.rocSPARSE.rocsparse_spmv(rocSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size, C_NULL)
                buffer = ROCVector{UInt8}(undef, buffer_size[])
                return AMD_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer_size, buffer)
            else
                alpha = Ref{T}(one(T))
                beta = Ref{T}(zeto(T))
                descA = rocSPARSE.ROCSparseMatrixDescriptor(A, 'O')
                descX = rocSPARSE.ROCDenseMatrixDescriptor(T, n, nrhs)
                descY = rocSPARSE.ROCDenseMatrixDescriptor(T, m, nrhs)
                algo = rocSPARSE.rocsparse_spmm_alg_default
                buffer_size = Ref{Csize_t}()
                transb = 'N'
                rocSPARSE.rocsparse_spmm(rocSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T,
                                         algo, rocSPARSE.rocsparse_spmm_stage_buffer_size, buffer_size, C_NULL)
                buffer = ROCVector{UInt8}(undef, buffer_size[])
                rocSPARSE.rocsparse_spmm(rocSPARSE.handle(), transa, 'N', alpha, descA, descX, beta, descY, T,
                                         algo, rocSPARSE.rocsparse_spmm_stage_preprocess, buffer_size, buffer)
                return AMD_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer_size, buffer)
            end
        end
    end
end

function LinearAlgebra.mul!(y::ROCVector{T}, A::AMD_KrylovOperator{T}, x::ROCVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch(""))
    (length(x) != A.n) && throw(DimensionMismatch(""))
    (A.nrhs != 1) && throw(DimensionMismatch(""))
    descY = rocSPARSE.ROCDenseVectorDescriptor(y)
    descX = rocSPARSE.ROCDenseVectorDescriptor(x)
    algo = rocSPARSE.rocsparse_spmv_alg_default
    rocSPARSE.rocsparse_spmv(rocSPARSE.handle(), A.transa, Ref{T}(one(T)), A.descA, descX,
                             Ref{T}(zero(T)), descY, T, algo, A.buffer_size, A.buffer)
end

function LinearAlgebra.mul!(Y::ROCMatrix{T}, A::AMD_KrylovOperator{T}, X::ROCMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch(""))
    (mX != A.n) && throw(DimensionMismatch(""))
    (nY == nX == A.nrhs) || throw(DimensionMismatch(""))
    descY = rocSPARSE.ROCDenseMatrixDescriptor(Y)
    descX = rocSPARSE.ROCDenseMatrixDescriptor(X)
    algo = rocSPARSE.rocsparse_spmm_alg_default
    rocSPARSE.rocsparse_spmm(rocSPARSE.handle(), A.transa, 'N', Ref{T}(one(T)), A.descA, descX,
                             Ref{T}(zero(T)), descY, T, algo, rocSPARSE.rocsparse_spmm_stage_compute, A.buffer_size, A.buffer)
end
