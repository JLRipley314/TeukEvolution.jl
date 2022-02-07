using TeukEvolution
using Documenter

DocMeta.setdocmeta!(TeukEvolution, :DocTestSetup, :(using TeukEvolution); recursive=true)

makedocs(;
    modules=[TeukEvolution],
    authors="Justin L. Ripley",
    repo="https://github.com/JLRipley314/TeukEvolution.jl/blob/{commit}{path}#{line}",
    sitename="TeukEvolution.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JLRipley314.github.io/TeukEvolution.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JLRipley314/TeukEvolution.jl",
    devbranch="main",
)
