include("../src/TeukEvolution.jl")

import .TeukEvolution as TE

# Launches a run for s=-2, l=2, m=2 mode initial data,
# for a spin a=0 blakc hole. 
# See the qnm directory for other example qnm initial
# data, all of which was generated with the 
# TeukolskyQNMFunctions.jl code.

params = Dict(
    "outdir" => "s-2_m0_n0",
    "nx" => 128,        # number of x radial grid points
    "ny" => 32,         # number of y collocation points
    "nt" => 160000,     # number of time steps
    "ts" => 400,        # save every ts time steps
    "psi_spin" => -2,   # spin-weight of linear evolution scalar
    "id_kind" => "qnm", # read from qnm files
    "m_vals" => [2],    # m angular values
    
    "runtype" => "linear_field", # no metric reconstruction
    
    "cl" => 1.0,  # compactification scale
    "cfl" => 0.2, # CFL number
    "bhs" => 0.0, # black hole spin 
    "bhm" => 1.0, # black hole mass
    
    "precision" => Float64, # precision the code is compiled at
    
    "id_m" => 2,          # angular number for qnm initial data  
    "id_amp" => 1.0,      # initial amplitude of the qnm (max abs of field)
    "id_overtone_n" => 0, # overtone number to run
    "id_filename" => "s-2_l2_m2_a0.0.h5" # file name specifies the l angular number and black hole spin
                                         # NOTE: make sure s,m,a match the rest of the input!!!
)

@time TE.launch(params)
