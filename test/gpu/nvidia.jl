using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using KernelAbstractions

_get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
include("gpu.jl")

@testset "Nvidia -- CUDA.jl" begin

  @test CUDA.functional()
  CUDA.allowscalar(false)

  @testset "IC(0)" begin
    @testset "CuSparseMatrixCSC -- $FC" for FC in (Float64,)
      test_ic0(FC, CuVector{FC}, CuSparseMatrixCSC{FC})
    end
    @testset "CuSparseMatrixCSR -- $FC" for FC in (Float64, ComplexF64)
      test_ic0(FC, CuVector{FC}, CuSparseMatrixCSR{FC})
    end
  end

  @testset "ILU(0)" begin
    @testset "CuSparseMatrixCSC -- $FC" for FC in (Float64,)
      test_ilu0(FC, CuVector{FC}, CuSparseMatrixCSC{FC})
    end
    @testset "CuSparseMatrixCSR -- $FC" for FC in (Float64, ComplexF64)
      test_ilu0(FC, CuVector{FC}, CuSparseMatrixCSR{FC})
    end
  end

  @testset "Block Jacobi preconditioner" begin
      test_preconditioner(CPU(), Array, SparseMatrixCSC)
      if CUDA.has_cuda()
          test_preconditioner(CUDABackend(), CuArray, CuSparseMatrixCSR)
      end
  end

end
