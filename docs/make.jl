using Documenter, Preconditioning

makedocs(
  modules = [Preconditioning],
  doctest = true,
  linkcheck = true,
  strict = true,
  format = Documenter.HTML(assets = ["assets/style.css"],
                           ansicolor = true,
                           prettyurls = get(ENV, "CI", nothing) == "true",
                           collapselevel = 1),
  sitename = "Preconditioning.jl",
  pages = ["Home" => "index.md",
           "Reference" => "reference.md"
          ]
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/Preconditioning.jl.git",
  push_preview = true,
  devbranch = "main",
)
