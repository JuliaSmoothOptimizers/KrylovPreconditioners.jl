mutable struct AMD_IC0{SM}
  P::SM
  timer_update::Float64
end

for (SparseMatrixType, BlasType) in ((:(ROCSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(ROCSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ic0(A::$SparseMatrixType) where T <: $BlasType
      P = rocSPARSE.ic0(A, 'O')
      return AMD_IC0(P, 0.0)
    end

    function KP.update!(p::AMD_IC0{$SparseMatrixType}, A::$SparseMatrixType) where T <: $BlasType
      p.P = rocSPARSE.ic0(A, 'O')
    end
  end
end

for ArrayType in (:(ROCVector{T}), :(ROCMatrix{T}))
  @eval begin
    function ldiv!(ic::AMD_IC0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(LowerTriangular(ic.P), x)   # Forward substitution with L
      ldiv!(LowerTriangular(ic.P)', x)  # Backward substitution with Lᴴ
      return x
    end

    function ldiv!(y::$ArrayType, ic::AMD_IC0{ROCSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ic.timer_update += @elapsed begin
      copyto!(y, x)
      ldiv!(ic, y)
      end
      return y
    end

    function ldiv!(ic::AMD_IC0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      ldiv!(UpperTriangular(ic.P)', x)  # Forward substitution with L
      ldiv!(UpperTriangular(ic.P), x)   # Backward substitution with Lᴴ
      return x
    end

    function ldiv!(y::$ArrayType, ic::AMD_IC0{ROCSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      ic.timer_update += @elapsed begin
      copyto!(y, x)
      ldiv!(ic, y)
      end
      return y
    end
  end
end
