using oneAPI, oneAPI.oneMKL

include("gpu.jl")

@testset "Intel -- oneAPI.jl" begin

  @test oneAPI.functional()
  oneAPI.allowscalar(false)

  @testset "Block Jacobi preconditioner" begin
    test_block_jacobi(oneAPIBackend(), oneArray, oneSparseMatrixCSR)
  end
end
