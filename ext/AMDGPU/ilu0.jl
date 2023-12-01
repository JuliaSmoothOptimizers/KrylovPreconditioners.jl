mutable struct AMD_ILU0{SM} <: AbstractKrylovPreconditioner
  n::Int
  desc::rocSPARSE.ROCMatrixDescriptor
  buffer::ROCVector{UInt8}
  info::MatInfo
  timer_update::Float64
  P::SM
end

for (bname, aname, sname, T) in ((:rocsparse_scsrilu0_buffer_size, :rocsparse_scsrilu0_analysis, :rocsparse_scsrilu0, :Float32),
                                 (:rocsparse_dcsrilu0_buffer_size, :rocsparse_dcsrilu0_analysis, :rocsparse_dcsrilu0, :Float64),
                                 (:rocsparse_ccsrilu0_buffer_size, :rocsparse_ccsrilu0_analysis, :rocsparse_ccsrilu0, :ComplexF32),
                                 (:rocsparse_zcsrilu0_buffer_size, :rocsparse_zcsrilu0_analysis, :rocsparse_zcsrilu0, :ComplexF64))
  @eval begin
    function KP.kp_ilu0(A::ROCSparseMatrixCSR{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
      info = MatInfo()
      buffer_size = Ref{Cint}()
      rocSPARSE.$bname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, buffer_size)
      buffer = ROCVector{UInt8}(undef, buffer_size[])
      rocSPARSE.$aname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, rocSPARSE.rocsparse_analysis_policy_force, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      posit = Ref{Cint}(1)
      rocSPARSE.rocsparse_csrilu0_zero_pivot(rocSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      rocSPARSE.$sname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      return AMD_ILU0(n, desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::AMD_ILU0{ROCSparseMatrixCSR{$T,Cint}}, A::ROCSparseMatrixCSR{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      rocSPARSE.$sname(rocSPARSE.handle(), p.n, nnz(p.P), p.desc, p.P.nzVal, p.P.rowPtr, p.P.colVal, p.info, rocSPARSE.rocsparse_solve_policy_auto, p.buffer)
      return p
    end

    function KP.kp_ilu0(A::ROCSparseMatrixCSC{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
      info = MatInfo()
      buffer_size = Ref{Cint}()
      rocSPARSE.$bname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, buffer_size)
      buffer = ROCVector{UInt8}(undef, buffer_size[])
      rocSPARSE.$aname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, rocSPARSE.rocsparse_analysis_policy_force, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      posit = Ref{Cint}(1)
      rocSPARSE.rocsparse_csrilu0_zero_pivot(rocSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      rocSPARSE.$sname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      return AMD_ILU0(n, desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::AMD_ILU0{ROCSparseMatrixCSC{$T,Cint}}, A::ROCSparseMatrixCSC{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      rocSPARSE.$sname(rocSPARSE.handle(), p.n, nnz(p.P), p.desc, p.P.nzVal, p.P.colPtr, p.P.rowVal, p.info, rocSPARSE.rocsparse_solve_policy_auto, p.buffer)
      return p
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
