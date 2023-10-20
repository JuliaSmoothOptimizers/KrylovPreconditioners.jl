mutable struct NVIDIA_IC0{SM,DM}
  P::SM
  z::DM
end

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function ic0(A::$SparseMatrixType; nrhs::Int=1) where T <: $BlasType
      P = ic02(A)
      n = checksquare(A)
      z = nrhs == 1 ? CuVector{T}(undef, n) : CuMatrix{T}(undef, n, nrhs)
      return NVIDIA_IC0(P,z)
    end
  end
end

for ArrayType in (:(CuVector{T}), :(CuMatrix{T}))
  @eval begin
    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{<:$ArrayType,CuSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(ic.z, LowerTriangular(ic.P), x)   # Forward substitution with L
      ldiv!(y, LowerTriangular(ic.P)', ic.z)  # Backward substitution with Lᴴ
      return y
    end

    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{<:$ArrayType,CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(ic.z, UpperTriangular(ic.P)', x)  # Forward substitution with L
      ldiv!(y, UpperTriangular(ic.P), ic.z)   # Backward substitution with Lᴴ
      return y
    end
  end
end

mutable struct AMD_IC0{SM}
  P::SM
end

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function ic0(A::$SparseMatrixType) where T <: $BlasType
      P = ilu0(A, 'O')
      return AMD_IC0(P)
    end
  end
end

for ArrayType in (:(ROCVector{T}), :(ROCMatrix{T}))
  @eval begin
    function ldiv!(y::$ArrayType, ic::AMD_IC0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      copyto!(y, x)
      ldiv!(LowerTriangular(ic.P), y)   # Forward substitution with L
      ldiv!(LowerTriangular(ic.P)', y)  # Backward substitution with Lᴴ
      return y
    end

    function ldiv!(y::$ArrayType, ic::AMD_IC0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ldiv!(UpperTriangular(ic.P)', y)  # Forward substitution with L
      ldiv!(UpperTriangular(ic.P), y)   # Backward substitution with Lᴴ
      return y
    end
  end
end
