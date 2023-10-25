using AMDGPU, AMDGPU.rocSPARSE, AMDGPU.rocSOLVER

_get_type(J::ROCSparseMatrixCSR) = ROCArray{Float64, 1, AMDGPU.Mem.HIPBuffer}
include("gpu.jl")

@testset "AMD -- AMDGPU.jl" begin

  @test AMDGPU.functional()
  AMDGPU.allowscalar(false)

  @testset "IC(0)" begin
    @testset "ROCSparseMatrixCSC -- $FC" for FC in (Float64,)
      test_ic0(FC, ROCVector{FC}, ROCSparseMatrixCSC{FC})
    end
    @testset "ROCSparseMatrixCSR -- $FC" for FC in (Float64, ComplexF64)
      test_ic0(FC, ROCVector{FC}, ROCSparseMatrixCSR{FC})
    end
  end

  @testset "ILU(0)" begin
    @testset "ROCSparseMatrixCSC -- $FC" for FC in (Float64,)
      test_ilu0(FC, ROCVector{FC}, ROCSparseMatrixCSC{FC})
    end
    @testset "ROCSparseMatrixCSR -- $FC" for FC in (Float64, ComplexF64)
      test_ilu0(FC, ROCVector{FC}, ROCSparseMatrixCSR{FC})
    end
  end

  @testset "Block Jacobi preconditioner" begin
      test_preconditioner(CPU(), Array, SparseMatrixCSC)
      if CUDA.has_cuda()
          test_preconditioner(CUDABackend(), CuArray, CuSparseMatrixCSR)
      end
  end

end
