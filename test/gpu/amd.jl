using AMDGPU, AMDGPU.rocSPARSE, AMDGPU.rocSOLVER

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

end
