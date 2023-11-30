mutable struct IC0Info
    info::CUSPARSE.csric02Info_t

    function IC0Info()
        info_ref = Ref{CUSPARSE.csric02Info_t}()
        CUSPARSE.cusparseCreateCsric02Info(info_ref)
        obj = new(info_ref[])
        finalizer(CUSPARSE.cusparseDestroyCsric02Info, obj)
        obj
    end
end

unsafe_convert(::Type{CUSPARSE.csric02Info_t}, info::IC0Info) = info.info

mutable struct NVIDIA_IC0{SM} <: AbstractKrylovPreconditioner
  desc::CUSPARSE.CuMatrixDescriptor
  buffer::CuVector{UInt8}
  info::IC0Info
  timer_update::Float64
  P::SM
end

for (bname, aname, sname, T) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
  @eval begin
    function KP.kp_ic0(A::CuSparseMatrixCSR{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
      info = IC0Info()
      buffer_size = Ref{Cint}()
      CUSPARSE.$bname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, buffer_size)
      buffer = CuVector{UInt8}(undef, buffer_size[])
      CUSPARSE.$aname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      posit = Ref{Cint}(1)
      CUSPARSE.cusparseXcsric02_zeroPivot(CUSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      return NVIDIA_IC0(desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::NVIDIA_IC0{CuSparseMatrixCSR{$T,Cint}}, A::CuSparseMatrixCSR{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(P.p), p.desc, p.P.nzVal, p.P.rowPtr, p.P.colVal, p.info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, p.buffer)
      return p
    end

    function KP.kp_ic0(A::CuSparseMatrixCSC{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
      info = IC0Info()
      buffer_size = Ref{Cint}()
      CUSPARSE.$bname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, buffer_size)
      buffer = CuVector{UInt8}(undef, buffer_size[])
      CUSPARSE.$aname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      posit = Ref{Cint}(1)
      CUSPARSE.cusparseXcsric02_zeroPivot(CUSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      return NVIDIA_IC0(desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::NVIDIA_IC0{CuSparseMatrixCSC{$T,Cint}}, A::CuSparseMatrixCSC{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(p.P), p.desc, p.P.nzVal, p.P.colPtr, p.P.rowVal, p.info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, p.buffer)
      return p
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
      ic.timer_update += @elapsed begin
      ldiv!(ic, y)
      end
      return y
    end

    function ldiv!(ic::NVIDIA_IC0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasFloat
      ldiv!(UpperTriangular(ic.P)', x)  # Forward substitution with L
      ldiv!(UpperTriangular(ic.P), x)   # Backward substitution with Lᴴ
      return x
    end

    function ldiv!(y::$ArrayType, ic::NVIDIA_IC0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ic.timer_update += @elapsed begin
      ldiv!(ic, y)
      end
      return y
    end
  end
end
