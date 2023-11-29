mutable struct NVIDIA_ILU0{SM} <: AbstractKrylovPreconditioner
  P::SM
end

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ilu0(A::$SparseMatrixType) where T <: $BlasType
      P = CUSPARSE.ilu02(A)
      n = checksquare(A)
      return NVIDIA_ILU0(P)
    end
  end
end

for ArrayType in (:(CuVector{T}), :(CuMatrix{T}))
  @eval begin
    function ldiv!(ilu::NVIDIA_ILU0{CuSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(UnitLowerTriangular(ilu.P), x)  # Forward substitution with L
      ldiv!(UpperTriangular(ilu.P), x)      # Backward substitution with U
      return x
    end

    function ldiv!(y::$ArrayType, ilu::NVIDIA_ILU0{CuSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      copyto!(y, x)
      ldiv!(ilu, y)
      return y
    end

    function ldiv!(ilu::NVIDIA_ILU0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      ldiv!(LowerTriangular(ilu.P), x)      # Forward substitution with L
      ldiv!(UnitUpperTriangular(ilu.P), x)  # Backward substitution with U
      return x
    end

    function ldiv!(y::$ArrayType, ilu::NVIDIA_ILU0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ldiv!(ilu, y)
      return y
    end
  end
end
