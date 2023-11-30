mutable struct AMD_ILU0{SM}
  P::SM
  timer_update::Float64
end

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ilu0(A::$SparseMatrixType) where T <: $BlasType
      P = rocSPARSE.ilu0(A, 'O')
      return AMD_ILU0(P, 0.0)
    end

    function KP.update!(p::AMD_ILU0{$SparseMatrixType}, A::$SparseMatrixType) where T <: $BlasType
      p.P = rocSPARSE.ilu0(A, 'O')
    end
  end
end

for ArrayType in (:(ROCVector{T}), :(ROCMatrix{T}))
  @eval begin
    function ldiv!(ilu::AMD_ILU0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(UnitLowerTriangular(ilu.P), x)  # Forward substitution with L
      ldiv!(UpperTriangular(ilu.P), x)      # Backward substitution with U
      return x
    end

    function ldiv!(y::$ArrayType, ilu::AMD_ILU0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      copyto!(y, x)
      ilu.timer_update += @elapsed begin
      ldiv!(ilu, y)
      end
      return y
    end

    function ldiv!(ilu::AMD_ILU0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      ldiv!(LowerTriangular(ilu.P), x)      # Forward substitution with L
      ldiv!(UnitUpperTriangular(ilu.P), x)  # Backward substitution with U
      return x
    end

    function ldiv!(y::$ArrayType, ilu::AMD_ILU0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ilu.timer_update += @elapsed begin
      ldiv!(ilu, y)
      end
      return y
    end
  end
end
