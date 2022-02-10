include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

@time TE.launch(ARGS[1])
