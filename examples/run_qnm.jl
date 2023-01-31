include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

# write the parameter file

params = Dict(
    "outdir" => "high2_a0.999_qnm_3",
    "nx" => 256,   # number of x radial grid points
    "ny" => 32,    # number of y collocation points
    "nt" => 160000, # number of time steps
    "ts" => 200,   # save every ts time steps
    "psi_spin" => -2, # spin-weight of linear evolution scalar
    #"id_kind" => "gaussian",
    "id_kind" => "qnm",
    "runtype" => "linear_field",
    "m_vals" => [2],   # m angular values
    "id_m" => 2,
    "id_amp" => 1.0,
    "id_filename" => "a0.999_l2_m2.h5",
    "id_overtone_n" => 3,
    # format: for each m value: [real part, imaginary part]
    "cl" => 1.0, # compactification scale
    "cfl" => 0.2, # CFL number
    "bhs" => 0.999, # black hole spin 
    "bhm" => 1.0, # black hole mass
    "precision" => Float64, # precision the code is compiled at
)

@time TE.launch(params)
