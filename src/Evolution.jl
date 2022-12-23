"""
General evolution for spin s field about Kerr
"""
module Evolution

include("Fields.jl")
include("Radial.jl")
include("Sphere.jl")
include("BackgroundNP.jl")
include("GHP.jl")

using .Fields: Field
import .Radial
import .Sphere
import .BackgroundNP: NP_0
import .GHP: GHP_ops

const half = 1.0 / 2.0
const third = 1.0 / 3.0
const sixth = 1.0 / 6.0

export Evo_lin_f, Initialize_Evo_lin_f, Evolve_lin_f!

struct Evo_lin_f
    A_pp::Array{Float64,2}
    A_pq::Array{Float64,2}
    B_pp::Array{ComplexF64,2}
    B_pq::Array{ComplexF64,2}
    B_pf::Array{ComplexF64,2}
    pre::Array{Float64,2}

    S_lapl::Array{Float64,2}
    S_fltr::Array{Float64,2}

    mv::Int64

    function Evo_lin_f(;
        Rvals::Vector{Float64},
        Cvals::Vector{Float64},
        Svals::Vector{Float64},
        bhm::Float64,
        bhs::Float64,
        mv::Int64,
        cl::Float64,
        spin::Int64,
    )
        nx = length(Rvals)
        ny = length(Cvals)

        A_pp = zeros(Float64, nx, ny)
        A_pq = zeros(Float64, nx, ny)

        B_pp = zeros(ComplexF64, nx, ny)
        B_pq = zeros(ComplexF64, nx, ny)
        B_pf = zeros(ComplexF64, nx, ny)

        S_lapl = zeros(Float64, ny, ny)
        S_fltr = zeros(Float64, ny, ny)

        pre = zeros(Float64, nx, ny)

        for j = 1:ny
            sy = Svals[j]
            for i = 1:nx
                R = Rvals[i]

                pre[i, j] =
                    1.0 / (
                        8 *
                        bhm *
                        (2 * (cl^2) * bhm - (bhs^2) * R) *
                        ((cl^2) + 2 * bhm * R) / (cl^4) - (bhs * sy)^2
                    )
            end
        end

        S_lapl = Sphere.swal_laplacian_matrix(ny, spin, mv)
        S_fltr = Sphere.swal_killtop_matrix(ny, spin, mv, 2)
        #Sphere.swal_filter_matrix(ny, spin, mv)

        for j = 1:ny
            cy = Cvals[j]
            for i = 1:nx
                R = Rvals[i]

                A_pp[i, j] =
                    (2 / (cl^4)) * (
                        (cl^6) +
                        (cl^2) * (bhs^2 - 8 * (bhm^2)) * (R^2) +
                        4 * (bhs^2) * bhm * (R^3)
                    )

                A_pq[i, j] =
                    ((R^2) / (cl^4)) * ((cl^4) - 2 * (cl^2) * bhm * R + (bhs^2) * (R^2))

                B_pp[i, j] = -(
                    2 * im * bhs * mv * ((cl^2) + 4 * bhm * R) / (cl^2) -
                    2 * (bhs^2) * (cl^2 + 6 * bhm * R) * R / (cl^4) - 4 * bhm * spin +
                    8 * (bhm^2) * (2 + spin) * R / (cl^2) +
                    2 * im * bhs * spin * cy
                )

                B_pq[i, j] =
                    (2 * R / (cl^4)) * (
                        2 * (bhs^2) * (R^2) + (1 + spin) * (cl^4) -
                        (im * bhs * mv + (3 + spin) * bhm) * ((cl^2) * R)
                    )

                B_pf[i, j] =
                    -(2 * R / (cl^4)) *
                    (im * bhs * (cl^2) * mv - (bhs^2) * R + (cl^2) * bhm * (1 + spin))

            end
        end

        return new(A_pp, A_pq, B_pp, B_pq, B_pf, pre, S_lapl, S_fltr, mv)
    end
end

function Initialize_Evo_lin_f(;
    Rvals::Vector{Float64},
    Cvals::Vector{Float64},
    Svals::Vector{Float64},
    Mvals::Vector{Int64},
    bhm::Float64,
    bhs::Float64,
    cl::Float64,
    spin::Int64,
)
    return Dict([
        (
            mv,
            Evo_lin_f(
                Rvals = Rvals,
                Cvals = Cvals,
                Svals = Svals,
                bhm = bhm,
                bhs = bhs,
                mv = mv,
                cl = cl,
                spin = spin,
            ),
        ) for mv in Mvals
    ])
end


"""
set_kp
"""
function set_kp(
    kp::Array{ComplexF64,2},
    f_rd1::Array{ComplexF64,2},
    f_rd2::Array{ComplexF64,2},
    sph_lap::Array{ComplexF64,2},
    p_rd1::Array{ComplexF64,2},
    f::Array{ComplexF64,2},
    p::Array{ComplexF64,2},
    pre::Array{Float64,2},
    A_pp::Array{Float64,2},
    A_pq::Array{Float64,2},
    B_pp::Array{ComplexF64,2},
    B_pq::Array{ComplexF64,2},
    B_pf::Array{ComplexF64,2},
)
    nx, ny = size(kp)
    for j = 1:ny
        for i = 1:nx
            kp[i, j] = pre[i, j] * (
                A_pp[i, j] * p_rd1[i, j] +
                A_pq[i, j] * f_rd2[i, j] +
                B_pp[i, j] * p[i, j] +
                B_pq[i, j] * f_rd1[i, j] +
                B_pf[i, j] * f[i, j]
                +
                sph_lap[i, j]
            )
        end
    end
    return nothing
end

"""
Fourth order Runge-Kutta evolution of linear field 
"""
function Evolve_lin_f!(lin_f, lin_p, Evo::Evo_lin_f, dr::Float64, dt::Float64)::Nothing
    @assert Evo.mv == lin_f.mv
    @assert Evo.mv == lin_p.mv

    ## simplify names for convenience

    f_n = lin_f.n
    f_tmp = lin_f.tmp
    f_np1 = lin_f.np1
    f_k = lin_f.k
    f_rd1 = lin_f.rad_d1
    f_rd2 = lin_f.rad_d2
    f_sph_lap = lin_f.sph_lap

    p_n = lin_p.n
    p_tmp = lin_p.tmp
    p_np1 = lin_p.np1
    p_k = lin_p.k
    p_rd1 = lin_p.rad_d1

    pre = Evo.pre
    A_pp = Evo.A_pp
    A_pq = Evo.A_pq
    B_pp = Evo.B_pp
    B_pq = Evo.B_pq
    B_pf = Evo.B_pf

    laplM = Evo.S_lapl
    fltrM = Evo.S_fltr

    nx, ny = size(f_n)

    ## step 1
    Radial.set_d1!(f_rd1, f_n, dr)
    Radial.set_d1!(p_rd1, p_n, dr)
    Radial.set_d2!(f_rd2, f_n, dr)

    Sphere.angular_matrix_mult!(f_sph_lap, f_n, laplM)

    set_kp(p_k, f_rd1, f_rd2, f_sph_lap, p_rd1, f_n, p_n, pre, A_pp, A_pq, B_pp, B_pq, B_pf)
    for j = 1:ny
        for i = 1:nx
            f_k[i, j] = p_n[i, j]

            f_tmp[i, j] = f_n[i, j] + half * dt * f_k[i, j]
            p_tmp[i, j] = p_n[i, j] + half * dt * p_k[i, j]

            f_np1[i, j] = f_n[i, j] + sixth * dt * f_k[i, j]
            p_np1[i, j] = p_n[i, j] + sixth * dt * p_k[i, j]
        end
    end
    ## step 2
    Radial.set_d1!(f_rd1, f_tmp, dr)
    Radial.set_d1!(p_rd1, p_tmp, dr)
    Radial.set_d2!(f_rd2, f_tmp, dr)

    Sphere.angular_matrix_mult!(f_sph_lap, f_tmp, laplM)

    set_kp(p_k, f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp, pre, A_pp, A_pq, B_pp, B_pq, B_pf)
    for j = 1:ny
        for i = 1:nx
            f_k[i, j] = p_tmp[i, j]

            f_tmp[i, j] = f_n[i, j] + half * dt * f_k[i, j]
            p_tmp[i, j] = p_n[i, j] + half * dt * p_k[i, j]

            f_np1[i, j] += third * dt * f_k[i, j]
            p_np1[i, j] += third * dt * p_k[i, j]
        end
    end
    ## step 3
    Radial.set_d1!(f_rd1, f_tmp, dr)
    Radial.set_d1!(p_rd1, p_tmp, dr)
    Radial.set_d2!(f_rd2, f_tmp, dr)

    Sphere.angular_matrix_mult!(f_sph_lap, f_tmp, laplM)

    set_kp(p_k, f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp, pre, A_pp, A_pq, B_pp, B_pq, B_pf)
    for j = 1:ny
        for i = 1:nx
            f_k[i, j] = p_tmp[i, j]

            f_tmp[i, j] = f_n[i, j] + dt * f_k[i, j]
            p_tmp[i, j] = p_n[i, j] + dt * p_k[i, j]

            f_np1[i, j] += third * dt * f_k[i, j]
            p_np1[i, j] += third * dt * p_k[i, j]
        end
    end
    ## step 4
    Radial.set_d1!(f_rd1, f_tmp, dr)
    Radial.set_d1!(p_rd1, p_tmp, dr)
    Radial.set_d2!(f_rd2, f_tmp, dr)

    Sphere.angular_matrix_mult!(f_sph_lap, f_tmp, laplM)

    set_kp(p_k, f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp, pre, A_pp, A_pq, B_pp, B_pq, B_pf)
    for j = 1:ny
        for i = 1:nx
            f_k[i, j] = p_tmp[i, j]

            f_np1[i, j] += sixth * dt * f_k[i, j]
            p_np1[i, j] += sixth * dt * p_k[i, j]
        end
    end
    Radial.filter!(f_np1, f_tmp, 0.5)
    Radial.filter!(p_np1, p_tmp, 0.5)
    for j = 1:ny
        for i = 1:nx
            f_tmp[i, j] = f_np1[i, j]
            p_tmp[i, j] = p_np1[i, j]
        end
    end
    Sphere.angular_matrix_mult!(f_np1, f_tmp, fltrM)
    Sphere.angular_matrix_mult!(p_np1, p_tmp, fltrM)

    return nothing
end

end
