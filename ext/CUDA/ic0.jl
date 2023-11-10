mutable struct NVIDIA_IC0{SM,DM} <: AbstractKrylovPreconditioner
  P::SM
  z::DM
end

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ic0(A::$SparseMatrixType; nrhs::Int=1) where T <: $BlasType
      P = CUSPARSE.ic02(A)
      n = checksquare(A)
      z = nrhs == 1 ? CuVector{T}(undef, n) : CuMatrix{T}(undef, n, nrhs)
      return NVIDIA_IC0(P,z)
    end
  end
end

for ArrayType in (:(CuVector{T}), :(CuMatrix{T}))
  @eval begin
    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{CuSparseMatrixCSR{T,Cint},<:$ArrayType}, x::$ArrayType) where T <: BlasFloat
      ldiv!(ic.z, LowerTriangular(ic.P), x)   # Forward substitution with L
      ldiv!(y, LowerTriangular(ic.P)', ic.z)  # Backward substitution with Lᴴ
      return y
    end

    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{CuSparseMatrixCSC{T,Cint},<:$ArrayType}, x::$ArrayType) where T <: BlasFloat
      ldiv!(ic.z, UpperTriangular(ic.P)', x)  # Forward substitution with L
      ldiv!(y, UpperTriangular(ic.P), ic.z)   # Backward substitution with Lᴴ
      return y
    end
  end
end
