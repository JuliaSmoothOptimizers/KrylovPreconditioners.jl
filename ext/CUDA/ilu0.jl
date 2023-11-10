mutable struct NVIDIA_ILU0{SM,DM} <: AbstractKrylovPreconditioner
  P::SM
  z::DM
end

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ilu0(A::$SparseMatrixType; nrhs::Int=1) where T <: $BlasType
      P = CUSPARSE.ilu02(A)
      n = checksquare(A)
      z = nrhs == 1 ? CuVector{T}(undef, n) : CuMatrix{T}(undef, n, nrhs)
      return NVIDIA_ILU0(P,z)
    end
  end
end

for ArrayType in (:(CuVector{T}), :(CuMatrix{T}))
  @eval begin
    function ldiv!(y::$ArrayType, ilu::NVIDIA_ILU0{CuSparseMatrixCSR{T,Cint},<:$ArrayType}, x::$ArrayType) where T <: BlasFloat
      ldiv!(ilu.z, UnitLowerTriangular(ilu.P), x)  # Forward substitution with L
      ldiv!(y, UpperTriangular(ilu.P), ilu.z)      # Backward substitution with U
      return y
    end

    function ldiv!(y::$ArrayType, ilu::NVIDIA_ILU0{CuSparseMatrixCSC{T,Cint},<:$ArrayType}, x::$ArrayType) where T <: BlasReal
      ldiv!(ilu.z, LowerTriangular(ilu.P), x)      # Forward substitution with L
      ldiv!(y, UnitUpperTriangular(ilu.P), ilu.z)  # Backward substitution with U
      return y
    end
  end
end
