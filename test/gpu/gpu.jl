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
