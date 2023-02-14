include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

# write the parameter file

params = Dict(
    "outdir" => "low_sp2",
    "nx" => 128,   # number of x radial grid points
    "ny" => 24,    # number of y collocation points
    "nt" => 80000, # number of time steps
    "ts" => 500,   # save every ts time steps
    "psi_spin" => +2, # spin-weight of linear evolution scalar
    "id_kind" => "gaussian",
    "runtype" => "linear_field",
    "m_vals" => [-2, 2],   # m angular values
    "id_l_ang" => [2, 2],
    "id_ru" => [3.0, 3.0],
    "id_rl" => [-3.0, -3.0],
    "id_width" => [6.0, 6.0],

    "ingoing" => true, # ingoing wave

    # format: for each m value: [real part, imaginary part]
    "id_amp" => [[0.0, 0.0], [0.4, 0.0]],
    "cl" => 1.0, # compactification scale
    "cfl" => 0.5, # CFL number
    "bhs" => 0.0, # black hole spin 
    "bhm" => 1, # black hole mass
    "precision" => Float64, # precision the code is compiled at
)

@time TE.launch(params)
