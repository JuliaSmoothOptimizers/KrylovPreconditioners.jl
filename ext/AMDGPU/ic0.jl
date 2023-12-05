mutable struct AMD_IC0{SM} <: AbstractKrylovPreconditioner
  n::Int
  desc::rocSPARSE.ROCMatrixDescriptor
  buffer::ROCVector{UInt8}
  info::rocSPARSE.MatInfo
  timer_update::Float64
  P::SM
end

for (bname, aname, sname, T) in ((:rocsparse_scsric0_buffer_size, :rocsparse_scsric0_analysis, :rocsparse_scsric0, :Float32),
                                 (:rocsparse_dcsric0_buffer_size, :rocsparse_dcsric0_analysis, :rocsparse_dcsric0, :Float64),
                                 (:rocsparse_ccsric0_buffer_size, :rocsparse_ccsric0_analysis, :rocsparse_ccsric0, :ComplexF32),
                                 (:rocsparse_zcsric0_buffer_size, :rocsparse_zcsric0_analysis, :rocsparse_zcsric0, :ComplexF64))
  @eval begin
    function KP.kp_ic0(A::ROCSparseMatrixCSR{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
      info = rocSPARSE.MatInfo()
      buffer_size = Ref{Csize_t}()
      rocSPARSE.$bname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, buffer_size)
      buffer = ROCVector{UInt8}(undef, buffer_size[])
      rocSPARSE.$aname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info,
                       rocSPARSE.rocsparse_analysis_policy_force, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      posit = Ref{Cint}(1)
      rocSPARSE.rocsparse_csric0_zero_pivot(rocSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      rocSPARSE.$sname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.rowPtr, P.colVal, info, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      return AMD_IC0(n, desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::AMD_IC0{ROCSparseMatrixCSR{$T,Cint}}, A::ROCSparseMatrixCSR{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      rocSPARSE.$sname(rocSPARSE.handle(), p.n, nnz(p.P), p.desc, p.P.nzVal, p.P.rowPtr, p.P.colVal, p.info, rocSPARSE.rocsparse_solve_policy_auto, p.buffer)
      return p
    end

    function KP.kp_ic0(A::ROCSparseMatrixCSC{$T,Cint})
      P = copy(A)
      n = checksquare(P)
      desc = rocSPARSE.ROCMatrixDescriptor('G', 'L', 'N', 'O')
      info = rocSPARSE.MatInfo()
      buffer_size = Ref{Csize_t}()
      rocSPARSE.$bname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, buffer_size)
      buffer = ROCVector{UInt8}(undef, buffer_size[])
      rocSPARSE.$aname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info,
                       rocSPARSE.rocsparse_analysis_policy_force, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      posit = Ref{Cint}(1)
      rocSPARSE.rocsparse_csric0_zero_pivot(rocSPARSE.handle(), info, posit)
      (posit[] ≥ 0) && error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
      rocSPARSE.$sname(rocSPARSE.handle(), n, nnz(P), desc, P.nzVal, P.colPtr, P.rowVal, info, rocSPARSE.rocsparse_solve_policy_auto, buffer)
      return AMD_IC0(n, desc, buffer, info, 0.0, P)
    end

    function KP.update!(p::AMD_IC0{ROCSparseMatrixCSC{$T,Cint}}, A::ROCSparseMatrixCSC{$T,Cint})
      copyto!(p.P.nzVal, A.nzVal)
      rocSPARSE.$sname(rocSPARSE.handle(), p.n, nnz(p.P), p.desc, p.P.nzVal, p.P.colPtr, p.P.rowVal, p.info, rocSPARSE.rocsparse_solve_policy_auto, p.buffer)
      return p
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
