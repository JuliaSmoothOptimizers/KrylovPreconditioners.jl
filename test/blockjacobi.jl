using KernelAbstractions
using CUDA
using CUDA.CUSPARSE
using Krylov
using KrylovPreconditioners
using LinearAlgebra
using SparseArrays
using Test

_get_type(J::SparseMatrixCSC) = Vector{Float64}
_get_type(J::CuSparseMatrixCSR) = CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}

function generate_random_system(n::Int, m::Int)
    # Add a diagonal term for conditionning
    A = randn(n, m) + 15I
    x♯ = randn(m)
    b = A * x♯
    # Be careful: all algorithms work with sparse matrix
    spA = sparse(A)
    return spA, b, x♯
end

function test_preconditioner(device, AT, SMT)
    println("Testing ($device, $AT, $SMT)")
    n, m = 100, 100
    A, b, x♯  = generate_random_system(n, m)
    # Transfer data to device
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT
    x = similar(b); r = similar(b)
    nblocks = 2
    precond = BlockJacobiKrylovPreconditioner(A, nblocks, device)
    KrylovPreconditioners.update(precond, A, device)

    S = _get_type(A)
    linear_solver = Krylov.BicgstabSolver(n, m, S)
    CUDA.allowscalar() do
        Krylov.bicgstab!(
            linear_solver, A, b;
            # N=precond,
            atol=1e-10,
            rtol=1e-10,
            verbose=0,
            history=true,
        )
    end
    n_iters = length(linear_solver.stats.residuals)
    @show x
    copyto!(x, linear_solver.x)
    @show r = b - A * x
    @show resid = norm(r) / norm(b)
    @show x
    @test(resid ≤ 1e-6)
    @test x ≈ x♯
    @test n_iters <= n
end

# @testset "Block Jacobi preconditioner" begin
    test_preconditioner(CPU(), Array, SparseMatrixCSC)
    if CUDA.has_cuda()
        test_preconditioner(CUDABackend(), CuArray, CuSparseMatrixCSR)
    end
    # if AMDGPU.has_rocm_gpu()
    #     test_preconditioner(ROCBackend(), ROCArray, ROCSparseMatrixCSR)
    # end
# end