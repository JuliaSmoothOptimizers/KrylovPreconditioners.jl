using AMDGPU
using CUDA
using KernelAbstractions

if CUDA.functional()
    include("gpu/nvidia.jl")
end
if AMDGPU.functional()
    include("gpu/amd.jl")
end
