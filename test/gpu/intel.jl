using oneAPI, oneAPI.oneMKL

_get_type(J::oneSparseMatrixCSR) = oneArray{Float64, 1, oneAPI.oneL0.DeviceBuffer}
_is_csr(J::oneSparseMatrixCSR) = true
include("gpu.jl")

@testset "Intel -- oneAPI.jl" begin

  @test oneAPI.functional()
  oneAPI.allowscalar(false)

  @testset "Block Jacobi preconditioner" begin
    test_block_jacobi(oneAPIBackend(), oneArray, oneSparseMatrixCSR)
  end
end
