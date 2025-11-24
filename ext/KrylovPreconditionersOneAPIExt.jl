module KrylovPreconditionersOneAPIExt
using oneAPI

# Don't do anything if oneAPI isn't actually available.
if oneAPI.oneL0.NEO_jll.is_available() && oneAPI.oneL0.functional[]
    using oneAPI: global_queue, sycl_queue, context, device
    using LinearAlgebra
    using SparseArrays
    using oneAPI.oneMKL
    using LinearAlgebra: checksquare, BlasReal, BlasFloat
    import LinearAlgebra: ldiv!, mul!
    import Base: size, eltype, unsafe_convert

    using KrylovPreconditioners
    const KP = KrylovPreconditioners
    using KernelAbstractions
    const KA = KernelAbstractions

    include("oneAPI/block_jacobi.jl")
    include("oneAPI/operators.jl")
end

end
