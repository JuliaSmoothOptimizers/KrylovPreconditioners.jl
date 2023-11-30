mutable struct ILU0Info
    info::CUSPARSE.csrilu02Info_t

    function ILU0Info()
        info_ref = Ref{CUSPARSE.csrilu02Info_t}()
        CUSPARSE.cusparseCreateCsrilu02Info(info_ref)
        obj = new(info_ref[])
        finalizer(CUSPARSE.cusparseDestroyCsrilu02Info, obj)
        obj
    end
end

unsafe_convert(::Type{CUSPARSE.csrilu02Info_t}, info::ILU0Info) = info.info

mutable struct NVIDIA_ILU0{SM} <: AbstractKrylovPreconditioner
  desc::CUSPARSE.cusparseMatDescr_t
  buffer::CuVector{UInt8}
  info::ILU0Info
  timer_update::Float64
  P::SM
end

for (bname, aname, sname, T) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
  @eval begin
    function KP.kp_ilu0(A::CuSparseMatrixCSR{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
      info = ILU0Info()
      buffer_size = Ref{Cint}()
      CUSPARSE.$bname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, buffer_size)
      buffer = CuVector{UInt8}(undef, buffer_size[])
      CUSPARSE.$aname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      posit = Ref{Cint}(1)
      CUSPARSE.cusparseXcsric02_zeroPivot(CUSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      return NVIDIA_ILU0(desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::NVIDIA_ILU0{CuSparseMatrixCSR{$T,Cint}}, A::CuSparseMatrixCSR{$T,Cint})
      p.P = CUSPARSE.ilu02(A)
    end

    function KP.kp_ilu0(A::CuSparseMatrixCSC{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', 'O')
      info = ILU0Info()
      buffer_size = Ref{Cint}()
      CUSPARSE.$bname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, buffer_size)
      buffer = CuVector{UInt8}(undef, buffer_size[])
      CUSPARSE.$aname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      posit = Ref{Cint}(1)
      CUSPARSE.cusparseXcsric02_zeroPivot(CUSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      CUSPARSE.$sname(CUSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, CUSPARSE.CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
      return NVIDIA_ILU0(desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::NVIDIA_ILU0{CuSparseMatrixCSC{$T,Cint}}, A::CuSparseMatrixCSC{$T,Cint})
      p.P = CUSPARSE.ilu02(A)
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
      ilu.timer_update += @elapsed begin
      ldiv!(ilu, y)
      end
      return y
    end

    function ldiv!(ilu::NVIDIA_ILU0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      ldiv!(LowerTriangular(ilu.P), x)      # Forward substitution with L
      ldiv!(UnitUpperTriangular(ilu.P), x)  # Backward substitution with U
      return x
    end

    function ldiv!(y::$ArrayType, ilu::NVIDIA_ILU0{CuSparseMatrixCSC{T,Cint}}, x::$ArrayType) where T <: BlasReal
      copyto!(y, x)
      ilu.timer_update += @elapsed begin
      ldiv!(ilu, y)
      end
      return y
    end
  end
end
