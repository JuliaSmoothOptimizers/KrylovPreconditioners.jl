using Documenter, KrylovPreconditioners

makedocs(
  modules = [KrylovPreconditioners],
  checkdocs = :exports,
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(assets = ["assets/style.css"],
                           ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "KrylovPreconditioners.jl",
  pages = ["Home" => "index.md",
           "Krylov operators" => "krylov_operators.md",
           "Triangular operators" => "triangular_operators.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/KrylovPreconditioners.jl.git",
  push_preview = true,
  devbranch = "main",
)
