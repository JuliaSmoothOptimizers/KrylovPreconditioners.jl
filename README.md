# KrylovPreconditioners.jl

| **Documentation** | **CI** | **Coverage** | **Downloads** |
|:-----------------:|:------:|:------------:|:-------------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![downloads][downloads-img]][downloads-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/KrylovPreconditioners.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/KrylovPreconditioners.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/KrylovPreconditioners.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/KrylovPreconditioners.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/KrylovPreconditioners.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/KrylovPreconditioners.jl
[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FKrylovPreconditioners&query=total_requests&suffix=%2Fmonth&label=Downloads
[downloads-url]: https://juliapkgstats.com/pkg/KrylovPreconditioners

## How to Cite

If you use KrylovPreconditioners.jl in your work, please cite it using the format provided in [`CITATION.cff`](https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl/blob/main/CITATION.cff).

## How to Install

To get started with `KrylovPreconditioners.jl`, you can install it using Julia's package manager:

```julia
julia> ]
pkg> add KrylovPreconditioners
```

To use the package alongside `Krylov.jl`, simply import both packages:

```julia
using Krylov, KrylovPreconditioners
```

## Content

To enhance the performance of [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl), especially on GPUs, we recommend `KrylovPreconditioners.jl`.
This package provides a variety of preconditioning strategies that significantly improve convergence rates for Krylov solvers, making them more efficient for large-scale problems.
It also contains operators that improve the efficiency of sparse matrix-dense vector products and sparse triangular solves on different GPUs, ensuring better performance on modern hardware.

[KrylovPreconditioners.jl](https://github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl) is the best sidekick of [Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl). └(^o^ )Ｘ( ^o^)┘

