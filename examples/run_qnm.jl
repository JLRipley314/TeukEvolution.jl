include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

# write the parameter file

params = Dict(
    "outdir" => "low_qnm_8",
    "nx" => 128,   # number of x radial grid points
    "ny" => 24,    # number of y collocation points
    "nt" => 80000, # number of time steps
    "ts" => 100,   # save every ts time steps
    "psi_spin" => -2, # spin-weight of linear evolution scalar
    #"id_kind" => "gaussian",
    "id_kind" => "qnm",
    "runtype" => "linear_field",
    "m_vals" => [2],   # m angular values
    "id_m" => 2,
    "id_amp" => 1.0,
    "id_filename" => "a0.0_l2_m2.h5",
    "id_overtone_n" => 8,
    # format: for each m value: [real part, imaginary part]
    "cl" => 1.0, # compactification scale
    "cfl" => 0.5, # CFL number
    "bhs" => 0.0, # black hole spin 
    "bhm" => 1.0, # black hole mass
    "precision" => Float64, # precision the code is compiled at
)

@time TE.launch(params)
