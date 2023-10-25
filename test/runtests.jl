using AMDGPU
using CUDA
using KernelAbstractions

if CUDA.functional()
    include("gpu/nvidia.jl")
end
if AMDGPU.has_rocm_gpu()
    include("gpu/amd.jl")
end
