import SparseArrays: nnz
import LinearAlgebra: ldiv!
import Base.\

export forward_substitution!, backward_substitution!
export adjoint_forward_substitution!, adjoint_backward_substitution!

"""
Returns the number of nonzeros of the `L` and `U`
factor combined.

Excludes the unit diagonal of the `L` factor,
which is not stored.
"""
nnz(F::ILUFactorization) = nnz(F.L) + nnz(F.U)

function ldiv!(F::ILUFactorization, y::AbstractVecOrMat)
    forward_substitution!(F, y)
    backward_substitution!(F, y)
end

function ldiv!(F::AdjointFactorization{<:Any,<:ILUFactorization}, y::AbstractVecOrMat)
    adjoint_forward_substitution!(F.parent, y)
    adjoint_backward_substitution!(F.parent, y)
end

function ldiv!(y::AbstractVector, F::ILUFactorization, x::AbstractVector)
    y .= x
    ldiv!(F, y)
end

function ldiv!(y::AbstractVector, F::AdjointFactorization{<:Any,<:ILUFactorization}, x::AbstractVector)
    y .= x
    ldiv!(F, y)
end

function ldiv!(y::AbstractMatrix, F::ILUFactorization, x::AbstractMatrix)
    y .= x
    ldiv!(F, y)
end

function ldiv!(y::AbstractMatrix, F::AdjointFactorization{<:Any,<:ILUFactorization}, x::AbstractMatrix)
    y .= x
    ldiv!(F, y)
end

(\)(F::ILUFactorization, y::AbstractVecOrMat) = ldiv!(F, copy(y))
(\)(F::AdjointFactorization{<:Any,<:ILUFactorization}, y::AbstractVecOrMat) = ldiv!(F, copy(y))

"""
Applies in-place backward substitution with the U factor of F, under the assumptions:

1. U is stored transposed / row-wise
2. U has no lower-triangular elements stored
3. U has (nonzero) diagonal elements stored.
"""
function backward_substitution!(F::ILUFactorization, y::AbstractVector)
    U = F.U
    @inbounds for col = U.n : -1 : 1

        # Substitutions
        for idx = U.colptr[col + 1] - 1 : -1 : U.colptr[col] + 1
            y[col] -= U.nzval[idx] * y[U.rowval[idx]]
        end

        # Final value for y[col]
        y[col] /= U.nzval[U.colptr[col]]
    end

    y
end

function backward_substitution!(F::ILUFactorization, y::AbstractMatrix)
    U = F.U
    p = size(y, 2)

    @inbounds for c = 1 : p
        @inbounds for col = U.n : -1 : 1

            # Substitutions
            for idx = U.colptr[col + 1] - 1 : -1 : U.colptr[col] + 1
                y[col,c] -= U.nzval[idx] * y[U.rowval[idx],c]
            end

            # Final value for y[col,c]
            y[col,c] /= U.nzval[U.colptr[col]]
        end
    end

    y
end

function backward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    backward_substitution!(F, v)
end

function backward_substitution!(v::AbstractMatrix, F::ILUFactorization, y::AbstractMatrix)
    v .= y
    backward_substitution!(F, v)
end

function adjoint_backward_substitution!(F::ILUFactorization, y::AbstractVector)
    L = F.L
    @inbounds for col = L.n - 1 : -1 : 1
        # Substitutions
        for idx = L.colptr[col + 1] - 1 : -1 : L.colptr[col]
            y[col] -= L.nzval[idx] * y[L.rowval[idx]]
        end
    end

    y
end

function adjoint_backward_substitution!(F::ILUFactorization, y::AbstractMatrix)
    L = F.L
    p = size(y, 2)
    @inbounds for c = 1 : p
        @inbounds for col = L.n - 1 : -1 : 1
            # Substitutions
            for idx = L.colptr[col + 1] - 1 : -1 : L.colptr[col]
                y[col,c] -= L.nzval[idx] * y[L.rowval[idx],c]
            end
        end
    end

    y
end

function adjoint_backward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    adjoint_backward_substitution!(F, v)
end

function adjoint_backward_substitution!(v::AbstractMatrix, F::ILUFactorization, y::AbstractMatrix)
    v .= y
    adjoint_backward_substitution!(F, v)
end

"""
Applies in-place forward substitution with the L factor of F, under the assumptions:

1. L is stored column-wise (unlike U)
2. L has no upper triangular elements
3. L has *no* diagonal elements
"""
function forward_substitution!(F::ILUFactorization, y::AbstractVector)
    L = F.L
    @inbounds for col = 1 : L.n - 1
        for idx = L.colptr[col] : L.colptr[col + 1] - 1
            y[L.rowval[idx]] -= L.nzval[idx] * y[col]
        end
    end

    y
end

function forward_substitution!(F::ILUFactorization, y::AbstractMatrix)
    L = F.L
    p = size(y, 2)
    @inbounds for c = 1 : p
        @inbounds for col = 1 : L.n - 1
            for idx = L.colptr[col] : L.colptr[col + 1] - 1
                y[L.rowval[idx],c] -= L.nzval[idx] * y[col,c]
            end
        end
    end

    y
end

function forward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    forward_substitution!(F, v)
end

function forward_substitution!(v::AbstractMatrix, F::ILUFactorization, y::AbstractMatrix)
    v .= y
    forward_substitution!(F, v)
end

function adjoint_forward_substitution!(F::ILUFactorization, y::AbstractVector)
    U = F.U
    @inbounds for col = 1 : U.n
        # Final value for y[col]
        y[col] /= U.nzval[U.colptr[col]]

        for idx = U.colptr[col] + 1 : U.colptr[col + 1] - 1
            y[U.rowval[idx]] -= U.nzval[idx] * y[col]
        end
    end

    y
end

function adjoint_forward_substitution!(F::ILUFactorization, y::AbstractMatrix)
    U = F.U
    p = size(y, 2)
    @inbounds for c = 1 : p
        @inbounds for col = 1 : U.n
            # Final value for y[col,c]
            y[col,c] /= U.nzval[U.colptr[col]]

            for idx = U.colptr[col] + 1 : U.colptr[col + 1] - 1
                y[U.rowval[idx],c] -= U.nzval[idx] * y[col,c]
            end
        end
    end

    y
end

function adjoint_forward_substitution!(v::AbstractVector, F::ILUFactorization, y::AbstractVector)
    v .= y
    adjoint_forward_substitution!(F, v)
end

function adjoint_forward_substitution!(v::AbstractMatrix, F::ILUFactorization, y::AbstractMatrix)
    v .= y
    adjoint_forward_substitution!(F, v)
end
