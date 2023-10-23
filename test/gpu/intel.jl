using oneAPI

include("gpu.jl")

@testset "Intel -- oneAPI.jl" begin

  @test oneAPI.functional()
  oneAPI.allowscalar(false)
end
