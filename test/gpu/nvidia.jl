using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER

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

  @testset "KrylovOperator" begin
    @testset "CuSparseMatrixCOO -- $FC" for FC in (Float64, ComplexF64)
      test_operator(FC, CuVector{FC}, CuMatrix{FC}, CuSparseMatrixCOO{FC})
    end
    @testset "CuSparseMatrixCSC -- $FC" for FC in (Float64, ComplexF64)
      test_operator(FC, CuVector{FC}, CuMatrix{FC}, CuSparseMatrixCSC{FC})
    end
    @testset "CuSparseMatrixCSR -- $FC" for FC in (Float64, ComplexF64)
      test_operator(FC, CuVector{FC}, CuMatrix{FC}, CuSparseMatrixCSR{FC})
    end
  end

  @testset "Block Jacobi preconditioner" begin
    test_block_jacobi(CUDABackend(), CuArray, CuSparseMatrixCSR)
  end

end
