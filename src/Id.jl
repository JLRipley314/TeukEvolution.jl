module Id

include("Sphere.jl")
include("Radial.jl")

import .Sphere: swal
import .Radial: set_d1!

import Polynomials: ChebyshevT
import HDF5: h5read

export set_gaussian!, set_qnm!

"""
    set_gaussian!(
            f,
            p,
            spin::Integer,
            mv::Integer,
            l_ang::Integer,
            ru::Real, 
            rl::Real, 
            width::Real,
            amp::Complex,
            cl::Real,
            Rv::Vector{<:Real},
            Yv::Vector{<:Real}
           )::Nothing

Initial gaussian initial data for the linear evolution variable (Psi_{0,1,2,3,4}).
"""
function set_gaussian!(
    f,
    p,
    spin::Integer,
    mv::Integer,
    l_ang::Integer,
    ru::Real,
    rl::Real,
    width::Real,
    amp::Complex,
    cl::Real,
    Rv::Vector{<:Real},
    Yv::Vector{<:Real},
    Ingoing::Bool,
    dr::Float64,
    nx::Integer,
    ny::Integer,
)::Nothing
    @assert f.mv == mv
    @assert p.mv == mv

    nx, ny = f.nx, f.ny

    max_val = 0.0

    for j = 1:ny
        f.n[1,j] = 0.0
        p.n[1,j] = 0.0
        for i = 2:nx
            r = (cl^2) / Rv[i]

            bump = 0.0
            if ((r < ru) && (r > rl))
                bump = exp(-1.0 * width / (r - rl)) * exp(-2.0 * width / (ru - r))
            end

            f.n[i, j] = (((r - rl) / width)^2) * (((ru - r) / width)^2) * bump
            f.n[i, j] *= swal(spin, mv, l_ang, Yv[j])
            
            p.n[i, j] = 0.0
            
            max_val = max(abs(f.n[i, j]), max_val)
        end
    end


    if Ingoing==true
        f_rd1 = zeros(ComplexF64, nx, ny)
        set_d1!(f_rd1, f.n, dr)
        for j = 1:ny
            for i = 1:nx
                p.n[i,j] += - Rv[i] * Rv[i] / (2 + 4 * Rv[i]) * f_rd1[i,j]
            end
        end
    end
    ## rescale

    for j = 1:ny
        for i = 1:nx
            f.n[i, j] *= amp / max_val
            p.n[i, j] *= amp / max_val

            f.np1[i, j] = f.n[i, j]
            p.np1[i, j] = p.n[i, j]
        end
    end
    return nothing
end

"""
    set_qnm!(
            f,
            p,
            s::Integer,
            l::Integer,
            mv::Integer,
            n::Real,
            a::Real,
            amp::Real,
            Rv::Vector{<:Real},
            Yv::Vector{<:Real}
        )::Nothing

Initial qnm initial data for psi, read in from an HDF5 file
produced from 
https://github.com/JLRipley314/TeukolskyQNMFunctions.jl.
"""
function set_qnm!(
    f,
    p,
    spin::Integer,
    mv::Integer,
    n::Integer,
    filename::String,
    amp::Real,
    idm::Integer,
    Rv::Vector{<:Real},
    Yv::Vector{<:Real},
)::Nothing

    @assert f.mv == mv
    @assert p.mv == mv
    nx, ny = f.nx, f.ny
    h5f = h5read(dirname(pwd()) * "/qnm/" * filename, "[n=$(n)]")
    rpoly = ChebyshevT(h5f["radial_coef"])
    lpoly = h5f["angular_coef"]
    lmin = max(abs(spin), abs(mv))
    max_val = 0.0

    # only set the field if an evolution m matches the m mode in initial data
    # NOTE: we need to multiply the spin-weighted coefficients by (-1)^l due
    # as apprently there is a global parity inversion between
    # TeukolskyQNMFunctions.jl and TeukEvolution.jl
    if mv==idm
        for j = 1:ny
            for i = 1:nx
                f.n[i, j] = rpoly( (2 * Rv[i])/maximum(Rv) -1 )
                f.n[i, j] *= sum([
                    (-1)^l * lpoly[l+1] * swal(spin, mv, l + lmin, Yv[j]) for l = 0:(length(lpoly)-1)
                ])
                max_val = max(abs(f.n[i, j]), max_val)
            end
        end

        for j = 1:ny
            for i = 1:nx
                f.n[i, j] *= amp / max_val
                f.np1[i, j] = f.n[i, j]
            end
        end
        ## p = f,t = -iωf  
        omega = h5f["omega"][1]
        for j = 1:ny
            for i = 1:nx
                p.n[i, j] = -im * omega * f.n[i, j]
                p.np1[i, j] = p.n[i, j]
            end
        end

    end
    return nothing
end

end
