mutable struct AMD_ILU0{SM}
  P::SM
end

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function kp_ilu0(A::$SparseMatrixType) where T <: $BlasType
      P = rocSPARSE.ilu0(A, 'O')
      return AMD_ILU0(P)
    end
  end
end

for ArrayType in (:(ROCVector{T}), :(ROCMatrix{T}))
  @eval begin
    function ldiv!(y::$ArrayType, ilu::AMD_ILU0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      copyto!(y, x)
      ldiv!(UnitLowerTriangular(ilu.P), y)  # Forward substitution with L
      ldiv!(UpperTriangular(ilu.P), y)      # Backward substitution with U
      return y
    end

    function ldiv!(y::$ArrayType, ilu::AMD_ILU0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ldiv!(LowerTriangular(ilu.P), y)      # Forward substitution with L
      ldiv!(UnitUpperTriangular(ilu.P), y)  # Backward substitution with U
      return y
    end
  end
end