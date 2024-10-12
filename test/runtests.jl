using Test
using KrylovPreconditioners

@testset "KrylovPreconditioners" begin
    @testset "IncompleteLU.jl" begin
        include("ilu/ilu.jl")
    end
end
