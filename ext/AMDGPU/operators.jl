mutable struct AMD_KrylovOperator{T} <: AbstractKrylovOperator{T}
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
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            descA = rocSPARSE.ROCSparseMatrixDescriptor(A, 'O')
            if nrhs == 1
                descX = rocSPARSE.ROCDenseVectorDescriptor(T, n)
                descY = rocSPARSE.ROCDenseVectorDescriptor(T, m)
                algo = rocSPARSE.rocSPARSE.rocsparse_spmv_alg_default
                buffer_size = Ref{Csize_t}()
                rocSPARSE.rocsparse_spmv(rocSPARSE.handle(), transa, alpha, descA, descX,
                                         beta, descY, T, algo, buffer_size, C_NULL)
                buffer = ROCVector{UInt8}(undef, buffer_size[])
                return AMD_KrylovOperator{T}(T, m, n, nrhs, transa, descA, buffer_size, buffer)
            else
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

        function KP.update!(A::AMD_KrylovOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            descB = rocSPARSE.ROCSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            return A
        end
    end
end

function LinearAlgebra.mul!(y::ROCVector{T}, A::AMD_KrylovOperator{T}, x::ROCVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = rocSPARSE.ROCDenseVectorDescriptor(y)
    descX = rocSPARSE.ROCDenseVectorDescriptor(x)
    algo = rocSPARSE.rocsparse_spmv_alg_default
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    rocSPARSE.rocsparse_spmv(rocSPARSE.handle(), A.transa, alpha, A.descA, descX,
                             beta, descY, T, algo, A.buffer_size, A.buffer)
end

function LinearAlgebra.mul!(Y::ROCMatrix{T}, A::AMD_KrylovOperator{T}, X::ROCMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = rocSPARSE.ROCDenseMatrixDescriptor(Y)
    descX = rocSPARSE.ROCDenseMatrixDescriptor(X)
    algo = rocSPARSE.rocsparse_spmm_alg_default
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    rocSPARSE.rocsparse_spmm(rocSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX,
                             beta, descY, T, algo, rocSPARSE.rocsparse_spmm_stage_compute, A.buffer_size, A.buffer)
end

mutable struct AMD_TriangularOperator{T} <: AbstractTriangularOperator{T}
    type::Type{T}
    m::Int
    n::Int
    nrhs::Int
    transa::Char
    descA::rocSPARSE.ROCSparseMatrixDescriptor
    buffer_size::Ref{Csize_t}
    buffer::ROCVector{UInt8}
end

eltype(A::AMD_TriangularOperator{T}) where T = T
size(A::AMD_TriangularOperator) = (A.m, A.n)

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T}), :BlasFloat),
                                     (:(ROCSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function KP.TriangularOperator(A::$SparseMatrixType, uplo::Char, diag::Char; nrhs::Int=1, transa::Char='N') where T <: $BlasType
            m,n = size(A)
            alpha = Ref{T}(one(T))
            descA = rocSPARSE.ROCSparseMatrixDescriptor(A, 'O')
            rocsparse_uplo = Ref{rocSPARSE.rocsparse_fill_mode}(uplo)
            rocsparse_diag = Ref{rocSPARSE.rocsparse_diag_type}(diag)
            rocSPARSE.rocsparse_spmat_set_attribute(descA, rocSPARSE.rocsparse_spmat_fill_mode, rocsparse_uplo, Csize_t(sizeof(rocsparse_uplo)))
            rocSPARSE.rocsparse_spmat_set_attribute(descA, rocSPARSE.rocsparse_spmat_diag_type, rocsparse_diag, Csize_t(sizeof(rocsparse_diag)))
            if nrhs == 1
                descX = rocSPARSE.ROCDenseVectorDescriptor(T, n)
                descY = rocSPARSE.ROCDenseVectorDescriptor(T, m)
                algo = rocSPARSE.rocsparse_spsv_alg_default
                buffer_size = Ref{Csize_t}()
                rocSPARSE.rocsparse_spsv(rocSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo,
                                         rocSPARSE.rocsparse_spsv_stage_buffer_size, buffer_size, C_NULL)
                buffer = ROCVector{UInt8}(undef, buffer_size[])
                rocSPARSE.rocsparse_spsv(rocSPARSE.handle(), transa, alpha, descA, descX, descY, T, algo,
                                         rocSPARSE.rocsparse_spsv_stage_preprocess, buffer_size, buffer)
                return AMD_TriangularOperator{T}(T, m, n, nrhs, transa, descA, buffer_size, buffer)
            else
                descX = rocSPARSE.ROCDenseMatrixDescriptor(T, n, nrhs)
                descY = rocSPARSE.ROCDenseMatrixDescriptor(T, m, nrhs)
                algo = rocSPARSE.rocsparse_spsm_alg_default
                buffer_size = Ref{Csize_t}()
                rocSPARSE.rocsparse_spsm(rocSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo,
                                         rocSPARSE.rocsparse_spsm_stage_buffer_size, buffer_size, C_NULL)
                buffer = ROCVector{UInt8}(undef, buffer_size[])
                rocSPARSE.rocsparse_spsm(rocSPARSE.handle(), transa, 'N', alpha, descA, descX, descY, T, algo,
                                         rocSPARSE.rocsparse_spsm_stage_preprocess, buffer_size, buffer)
                return AMD_TriangularOperator{T}(T, m, n, nrhs, transa, descA, buffer_size, buffer)
            end
        end

        function KP.update!(A::AMD_TriangularOperator{T}, B::$SparseMatrixType) where T <: $BlasFloat
            descB = rocSPARSE.ROCSparseMatrixDescriptor(B, 'O')
            A.descA = descB
            return A
        end
    end
end

function LinearAlgebra.ldiv!(y::ROCVector{T}, A::AMD_TriangularOperator{T}, x::ROCVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    (A.nrhs == 1) || throw(DimensionMismatch("A.nrhs != 1"))
    descY = rocSPARSE.ROCDenseVectorDescriptor(y)
    descX = rocSPARSE.ROCDenseVectorDescriptor(x)
    algo = rocSPARSE.rocsparse_spsv_alg_default
    alpha = Ref{T}(one(T))
    rocSPARSE.rocsparse_spsv(rocSPARSE.handle(), A.transa, alpha, A.descA, descX, descY, T,
                             algo, rocSPARSE.rocsparse_spsv_stage_compute, A.buffer_size, A.buffer)
end

function LinearAlgebra.ldiv!(Y::ROCMatrix{T}, A::AMD_TriangularOperator{T}, X::ROCMatrix{T}) where T <: BlasFloat
    mY, nY = size(Y)
    mX, nX = size(X)
    (mY != A.m) && throw(DimensionMismatch("mY != A.m"))
    (mX != A.n) && throw(DimensionMismatch("mX != A.n"))
    (nY == nX == A.nrhs) || throw(DimensionMismatch("nY != A.nrhs or nX != A.nrhs"))
    descY = rocSPARSE.ROCDenseMatrixDescriptor(Y)
    descX = rocSPARSE.ROCDenseMatrixDescriptor(X)
    algo = rocSPARSE.rocsparse_spsm_alg_default
    alpha = Ref{T}(one(T))
    rocSPARSE.rocsparse_spsm(rocSPARSE.handle(), A.transa, 'N', alpha, A.descA, descX, descY, T,
                             algo, rocSPARSE.rocsparse_spsm_stage_compute, A.buffer_size, A.buffer)
end
