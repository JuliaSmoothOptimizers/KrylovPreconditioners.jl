using SparseArrays, Random, Test
using LinearAlgebra, Krylov, KrylovPreconditioners
using KernelAbstractions

Random.seed!(666)

function test_ic0(FC, V, M)
  n = 100
  R = real(FC)
  A_cpu = rand(FC, n, n)
  A_cpu = A_cpu * A_cpu'
  A_cpu = sparse(A_cpu)
  b_cpu = rand(FC, n)

  A_gpu = M(A_cpu)
  b_gpu = V(b_cpu)
  P = kp_ic0(A_gpu)

  x_gpu, stats = cg(A_gpu, b_gpu, M=P, ldiv=true)
  r_gpu = b_gpu - A_gpu * x_gpu
  @test stats.niter ≤ 5
  @test norm(r_gpu) ≤ 1e-8
end

function test_ilu0(FC, V, M)
  n = 100
  R = real(FC)
  A_cpu = rand(FC, n, n)
  A_cpu = sparse(A_cpu)
  b_cpu = rand(FC, n)

  A_gpu = M(A_cpu)
  b_gpu = V(b_cpu)
  P = kp_ilu0(A_gpu)

  x_gpu, stats = gmres(A_gpu, b_gpu, N=P, ldiv=true)
  r_gpu = b_gpu - A_gpu * x_gpu
  @test stats.niter ≤ 5
  @test norm(r_gpu) ≤ 1e-8
end

_get_type(J::SparseMatrixCSC) = Vector{Float64}

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
            N=precond,
            atol=1e-10,
            rtol=1e-10,
            verbose=0,
            history=true,
        )
    end
    n_iters = length(linear_solver.stats.residuals)
    copyto!(x, linear_solver.x)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ 1e-6)
    @test x ≈ x♯
    @test n_iters <= n
end
