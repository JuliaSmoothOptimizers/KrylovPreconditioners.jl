using AMDGPU
using CUDA
using oneAPI
using Test

@testset "KrylovPreconditioners" begin
if AMDGPU.functional()
    @info "Testing AMDGPU backend"
    @testset "Testing AMDGPU backend" begin
        include("gpu/amd.jl")
    end
end

if CUDA.functional()
    @info "Testing CUDA backend"
    @testset "Testing CUDA backend" begin
        include("gpu/nvidia.jl")
    end
end

if oneAPI.functional()
    @info "Testing oneAPI backend"
    @testset "Testing oneAPI backend" begin
        include("gpu/intel.jl")
    end
end
end
