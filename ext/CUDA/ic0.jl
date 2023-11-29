mutable struct NVIDIA_IC0{SM} <: AbstractKrylovPreconditioner
  P::SM
end

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T,Cint}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T,Cint}), :BlasReal))
  @eval begin
    function KP.kp_ic0(A::$SparseMatrixType) where T <: $BlasType
      P = CUSPARSE.ic02(A)
      n = checksquare(A)
      return NVIDIA_IC0(P)
    end
  end
end

for ArrayType in (:(CuVector{T}), :(CuMatrix{T}))
  @eval begin
    function ldiv!(ic::NVIDIA_IC0{CuSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(LowerTriangular(ic.P), x)   # Forward substitution with L
      ldiv!(LowerTriangular(ic.P)', x)  # Backward substitution with Lᴴ
      return x
    end

    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{CuSparseMatrixCSR{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      copyto!(y, x)
      ldiv!(ic, y)
      return y
    end

    function ldiv!(ic::NVIDIA_IC0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(UpperTriangular(ic.P)', x)  # Forward substitution with L
      ldiv!(UpperTriangular(ic.P), x)   # Backward substitution with Lᴴ
      return x
    end

    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ldiv!(ic, y)
      return y
    end
  end
end
