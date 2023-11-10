mutable struct AMD_IC0{SM}
  P::SM
end

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ic0(A::$SparseMatrixType) where T <: $BlasType
      P = rocSPARSE.ic0(A, 'O')
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
