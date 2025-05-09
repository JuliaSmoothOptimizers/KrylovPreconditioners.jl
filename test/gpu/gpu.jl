using SparseArrays, Random, Test
using LinearAlgebra, Krylov, KrylovPreconditioners

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
  if (FC <: ComplexF64) && V.body.name.name == :ROCArray
    @test_broken norm(r_gpu) ≤ 1e-6
  else
    @test norm(r_gpu) ≤ 1e-8
  end

  A_gpu = M(A_cpu + 200*I)
  update!(P, A_gpu)
  x_gpu, stats = cg(A_gpu, b_gpu, M=P, ldiv=true)
  r_gpu = b_gpu - A_gpu * x_gpu
  @test stats.niter ≤ 5
  if (FC <: ComplexF64) && V.body.name.name == :ROCArray
    @test_broken norm(r_gpu) ≤ 1e-6
  else
    @test norm(r_gpu) ≤ 1e-8
  end
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

  A_gpu = M(A_cpu + 200*I)
  update!(P, A_gpu)
  x_gpu, stats = gmres(A_gpu, b_gpu, N=P, ldiv=true)
  r_gpu = b_gpu - A_gpu * x_gpu
  @test stats.niter ≤ 5
  @test norm(r_gpu) ≤ 1e-8
end

function test_operator(FC, V, DM, SM)
  m = 200
  n = 100
  A_cpu = rand(FC, n, n)
  A_cpu = sparse(A_cpu)
  b_cpu = rand(FC, n)

  A_gpu = SM(A_cpu)
  b_gpu = V(b_cpu)
  opA_gpu = KrylovOperator(A_gpu)

  x_gpu, stats = gmres(opA_gpu, b_gpu)
  r_gpu = b_gpu - A_gpu * x_gpu
  @test stats.solved
  @test norm(r_gpu) ≤ 1e-8

  A_cpu = rand(FC, m, n)
  A_cpu = sparse(A_cpu)
  A_gpu = SM(A_cpu)

  opA_gpu = KrylovOperator(A_gpu)
  for i = 1:5
    y_cpu = rand(FC, m)
    x_cpu = rand(FC, n)
    mul!(y_cpu, A_cpu, x_cpu)
    y_gpu = V(y_cpu)
    x_gpu = V(x_cpu)
    mul!(y_gpu, opA_gpu, x_gpu)
    @test collect(y_gpu) ≈ y_cpu
  end
  if V.body.name.name != :oneArray
    for j = 1:5
      y_cpu = rand(FC, m)
      x_cpu = rand(FC, n)
      A_cpu2 = A_cpu + j*I
      mul!(y_cpu, A_cpu2, x_cpu)
      y_gpu = V(y_cpu)
      x_gpu = V(x_cpu)
      A_gpu2 = SM(A_cpu2)
      update!(opA_gpu, A_gpu2)
      mul!(y_gpu, opA_gpu, x_gpu)
      @test collect(y_gpu) ≈ y_cpu
    end
  end

  nrhs = 3
  opA_gpu = KrylovOperator(A_gpu; nrhs)
  for i = 1:5
    Y_cpu = rand(FC, m, nrhs)
    X_cpu = rand(FC, n, nrhs)
    mul!(Y_cpu, A_cpu, X_cpu)
    Y_gpu = DM(Y_cpu)
    X_gpu = DM(X_cpu)
    mul!(Y_gpu, opA_gpu, X_gpu)
    @test collect(Y_gpu) ≈ Y_cpu
  end
  if V.body.name.name != :oneArray
    for j = 1:5
      Y_cpu = rand(FC, m, nrhs)
      X_cpu = rand(FC, n, nrhs)
      A_cpu2 = A_cpu + j*I
      mul!(Y_cpu, A_cpu2, X_cpu)
      Y_gpu = DM(Y_cpu)
      X_gpu = DM(X_cpu)
      A_gpu2 = SM(A_cpu2)
      update!(opA_gpu, A_gpu2)
      mul!(Y_gpu, opA_gpu, X_gpu)
      @test collect(Y_gpu) ≈ Y_cpu
    end
  end
end

function test_triangular(FC, V, DM, SM)
  n = 100
  for (uplo, diag, triangle) in [('L', 'U', UnitLowerTriangular),
                                 ('L', 'N', LowerTriangular    ),
                                 ('U', 'U', UnitUpperTriangular),
                                 ('U', 'N', UpperTriangular    )]
    A_cpu = rand(FC, n, n)
    A_cpu = uplo == 'L' ? tril(A_cpu) : triu(A_cpu)
    A_cpu = diag == 'U' ? A_cpu - Diagonal(A_cpu) + I : A_cpu
    A_cpu = sparse(A_cpu)
    b_cpu = rand(FC, n)

    A_gpu = SM(A_cpu)
    b_gpu = V(b_cpu)
    opA_gpu = TriangularOperator(A_gpu, uplo, diag)
    for i = 1:5
      y_cpu = rand(FC, n)
      x_cpu = rand(FC, n)
      ldiv!(y_cpu, triangle(A_cpu), x_cpu)
      y_gpu = V(y_cpu)
      x_gpu = V(x_cpu)
      ldiv!(y_gpu, opA_gpu, x_gpu)
      @test collect(y_gpu) ≈ y_cpu
    end
    if V.body.name.name != :oneArray
      for j = 1:5
        y_cpu = rand(FC, n)
        x_cpu = rand(FC, n)
        A_cpu2 = A_cpu + j*tril(A_cpu,-1) + j*triu(A_cpu,1)
        ldiv!(y_cpu, triangle(A_cpu2), x_cpu)
        y_gpu = V(y_cpu)
        x_gpu = V(x_cpu)
        A_gpu2 = SM(A_cpu2)
        update!(opA_gpu, A_gpu2)
        ldiv!(y_gpu, opA_gpu, x_gpu)
        @test collect(y_gpu) ≈ y_cpu
      end
    end

    nrhs = 3
    opA_gpu = TriangularOperator(A_gpu, uplo, diag; nrhs)
    for i = 1:5
      Y_cpu = rand(FC, n, nrhs)
      X_cpu = rand(FC, n, nrhs)
      ldiv!(Y_cpu, triangle(A_cpu), X_cpu)
      Y_gpu = DM(Y_cpu)
      X_gpu = DM(X_cpu)
      ldiv!(Y_gpu, opA_gpu, X_gpu)
      @test collect(Y_gpu) ≈ Y_cpu
    end
    if V.body.name.name != :oneArray
      for j = 1:5
        Y_cpu = rand(FC, n, nrhs)
        X_cpu = rand(FC, n, nrhs)
        A_cpu2 = A_cpu + j*tril(A_cpu,-1) + j*triu(A_cpu,1)
        ldiv!(Y_cpu, triangle(A_cpu2), X_cpu)
        Y_gpu = DM(Y_cpu)
        X_gpu = DM(X_cpu)
        A_gpu2 = SM(A_cpu2)
        update!(opA_gpu, A_gpu2)
        ldiv!(Y_gpu, opA_gpu, X_gpu)
        @test collect(Y_gpu) ≈ Y_cpu
      end
    end
  end
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

function test_block_jacobi(device, AT, SMT; test_update::Bool=true)
    m, n = 100, 100
    A, b, x♯  = generate_random_system(m, n)
    # Transfer data to device
    A = A |> SMT
    b = b |> AT
    x♯ = x♯ |> AT
    x = similar(b); r = similar(b)
    nblocks = 2
    if _is_csr(A)
      scaling_csr!(A, b, device)
    end
    precond = BlockJacobiPreconditioner(A, nblocks, device)
    test_update && update!(precond, A)

    S = _get_type(A)
    workspace = BicgstabWorkspace(m, n, S)
    bicgstab!(workspace, A, b; N=precond, atol=1e-10, rtol=1e-10, verbose=0, history=true)
    n_iters = Krylov.iteration_count(workspace)
    copyto!(x, workspace.x)
    r = b - A * x
    resid = norm(r) / norm(b)
    @test(resid ≤ 1e-6)
    @test x ≈ x♯
    @test n_iters ≤ n

    P = kp_block_jacobi(A)
end
