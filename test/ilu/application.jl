using Test
using KrylovPreconditioners: ILUFactorization, forward_substitution!, backward_substitution!
using LinearAlgebra

@testset "Forward and backward substitutions" begin
    function test_fw_substitution(F::ILUFactorization)
        A = F.L
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        forward_substitution!(v, F, x)
        forward_substitution!(F, x)
        ldiv!(UnitLowerTriangular(A), y)

        @test v ≈ y
        @test x ≈ y

        x = rand(n, 5)
        y = copy(x)
        v = zeros(n, 5)

        forward_substitution!(v, F, x)
        forward_substitution!(F, x)
        ldiv!(UnitLowerTriangular(A), y)

        @test v ≈ y
        @test x ≈ y
    end

    function test_bw_substitution(F::ILUFactorization)
        A = F.U
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        backward_substitution!(v, F, x)
        backward_substitution!(F, x)
        ldiv!(UpperTriangular(A'), y)

        @test v ≈ y
        @test x ≈ y

        x = rand(n, 5)
        y = copy(x)
        v = zeros(n, 5)

        backward_substitution!(v, F, x)
        backward_substitution!(F, x)
        ldiv!(UpperTriangular(A'), y)

        @test v ≈ y
        @test x ≈ y
    end

    L = sparse(tril(rand(10, 10), -1))
    U = sparse(tril(rand(10, 10)) + 10I)
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)

    L = sparse(tril(tril(sprand(10, 10, .5), -1)))
    U = sparse(tril(sprand(10, 10, .5) + 10I))
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)

    L = spzeros(10, 10)
    U = spzeros(10, 10) + 10I
    F = ILUFactorization(L, U)
    test_fw_substitution(F)
    test_bw_substitution(F)
end

@testset "Adjoint -- Forward and backward substitutions" begin
    function test_adjoint_fw_substitution(F::ILUFactorization)
        A = F.U
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        adjoint_forward_substitution!(v, F, x)
        adjoint_forward_substitution!(F, x)
        ldiv!(LowerTriangular(A), y)

        @test v ≈ y
        @test x ≈ y

        x = rand(n, 5)
        x2 = copy(x)
        y = copy(x)
        v = zeros(n, 5)

        adjoint_forward_substitution!(v, F, x)
        adjoint_forward_substitution!(F, x)
        ldiv!(LowerTriangular(A), y)

        @test v ≈ y
        @test x ≈ y
    end

    function test_adjoint_bw_substitution(F::ILUFactorization)
        A = F.L
        n = size(A, 1)
        x = rand(n)
        y = copy(x)
        v = zeros(n)

        adjoint_backward_substitution!(v, F, x)
        adjoint_backward_substitution!(F, x)
        ldiv!(UnitLowerTriangular(A)', y)

        @test v ≈ y
        @test x ≈ y

        x = rand(n, 5)
        y = copy(x)
        v = zeros(n, 5)

        adjoint_backward_substitution!(v, F, x)
        adjoint_backward_substitution!(F, x)
        ldiv!(UnitLowerTriangular(A)', y)

        @test v ≈ y
        @test x ≈ y
    end

    L = sparse(tril(rand(10, 10), -1))
    U = sparse(tril(rand(10, 10)) + 10I)
    F = ILUFactorization(L, U)
    test_adjoint_fw_substitution(F)
    test_adjoint_bw_substitution(F)

    L = sparse(tril(tril(sprand(10, 10, .5), -1)))
    U = sparse(tril(sprand(10, 10, .5) + 10I))
    F = ILUFactorization(L, U)
    test_adjoint_fw_substitution(F)
    test_adjoint_bw_substitution(F)

    L = spzeros(10, 10)
    U = spzeros(10, 10) + 10I
    F = ILUFactorization(L, U)
    test_adjoint_fw_substitution(F)
    test_adjoint_bw_substitution(F)
end

@testset "ldiv!" begin
    function test_ldiv!(L, U)
        LU = ILUFactorization(L, U)
        x = rand(size(LU.L, 1))
        y = copy(x)
        z = copy(x)
        w = copy(x)

        ldiv!(LU, x)
        ldiv!(UnitLowerTriangular(LU.L), y)
        ldiv!(UpperTriangular(LU.U'), y)

        @test x ≈ y
        @test LU \ z == x

        ldiv!(w, LU, z)

        @test w == x

        x = rand(size(LU.L, 1), 5)
        y = copy(x)
        z = copy(x)
        w = copy(x)

        ldiv!(LU, x)
        ldiv!(UnitLowerTriangular(LU.L), y)
        ldiv!(UpperTriangular(LU.U'), y)

        @test x ≈ y
        @test LU \ z == x

        ldiv!(w, LU, z)

        @test w == x
    end

    test_ldiv!(tril(sprand(10, 10, .5), -1), tril(sprand(10, 10, .5) + 10I))
end

@testset "Adjoint -- ldiv!" begin
    function test_adjoint_ldiv!(L, U)
        LU = ILUFactorization(L, U)
        ALU = adjoint(LU)

        x = rand(size(LU.L, 1))
        y = copy(x)
        z = copy(x)
        w = copy(x)

        ldiv!(ALU, x)
        ldiv!(LowerTriangular(LU.U), y)
        ldiv!(UnitLowerTriangular(LU.L)', y)

        @test x ≈ y
        @test ALU \ z == x

        ldiv!(w, ALU, z)

        @test w == x

        x = rand(size(LU.L, 1), 5)
        y = copy(x)
        z = copy(x)
        w = copy(x)

        ldiv!(ALU, x)
        ldiv!(LowerTriangular(LU.U), y)
        ldiv!(UnitLowerTriangular(LU.L)', y)

        @test x ≈ y
        @test ALU \ z == x

        ldiv!(w, ALU, z)

        @test w == x
    end

    test_adjoint_ldiv!(tril(sprand(10, 10, .5), -1), tril(sprand(10, 10, .5) + 10I))
end

@testset "nnz" begin
    L = tril(sprand(10, 10, .5), -1)
    U = tril(sprand(10, 10, .5)) + 10I
    LU = ILUFactorization(L, U)
    @test nnz(LU) == nnz(L) + nnz(U)
end
