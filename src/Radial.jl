module Radial

export R_vals, set_d1!, set_d2!, filter!

"""
    R_vals(
       nx::Integer, 
       dr::Real 
       )::Vector{<:Real}

Computes array of compactified radial points 
"""
function R_vals(nx::Integer, dr::Real)::Vector{<:Real}
    return [dr * i for i = 1:nx]
end

"""
    set_d1!(
       dv::AbstractArray{T}, 
       v::AbstractArray{T},
       dr::Real,
       ) where T<:Number

Compute the first derivative in radial direction with a 4th
order finite difference stencil.
"""
function set_d1!(dv::AbstractArray{T}, v::AbstractArray{T}, dr::Real) where {T<:Number}
    inv_dr = 1 / dr

    nx, ny = size(dv)

    for j = 1:ny
        for i = 3:nx-2
            dv[i, j] =
                (
                    -(1 // 12) * v[i+2, j] + (2 // 3) * v[i+1, j] -
                    (2 // 3) * v[i-1, j] + (1 // 12) * v[i-2, j]
                ) * inv_dr
        end
        dv[1, j] =
            (
                -3 * v[5, j] + 16 * v[4, j] - 36 * v[3, j] + 48 * v[2, j] -
                25 * v[1, j]
            ) *
            (1 // 12) *
            inv_dr
        dv[2, j] =
            (
                +1 * v[5, j] - 6 * v[4, j] + 18 * v[3, j] - 10 * v[2, j] -
                3 * v[1, j]
            ) *
            (1 // 12) *
            inv_dr

        dv[nx, j] =
            -(
                -3 * v[nx-4, j] + 16 * v[nx-3, j] - 36 * v[nx-2, j] +
                48 * v[nx-1, j] - 25 * v[nx, j]
            ) *
            (1 // 12) *
            inv_dr
        dv[nx-1, j] =
            -(
                +1 * v[nx-4, j] - 6 * v[nx-3, j] + 18 * v[nx-2, j] -
                10 * v[nx-1, j] - 3 * v[nx, j]
            ) *
            (1 // 12) *
            inv_dr
    end
    return nothing
end
"""
    set_d2!(
            dv::AbstractArray{T}, 
            v::AbstractArray{T},
            dr::Real
           ) where T<:Number

Compute the second derivative in radial direction
with a 4th order finite difference stencil.
"""
function set_d2!(dv::AbstractArray{T}, v::AbstractArray{T}, dr::Real) where {T<:Number}
    inv_dr2 = 1 / dr^2

    nx, ny = size(dv)

    for j = 1:ny
        for i = 3:nx-2
            dv[i, j] =
                (
                    -1 * v[i+2, j] + 16 * v[i+1, j] - 30 * v[i, j] +
                    16 * v[i-1, j] - 1 * v[i-2, j]
                ) *
                (1 // 12) *
                inv_dr2
        end
        dv[1, j] =
            (
                -10 * v[6, j] + 61 * v[5, j] - 156 * v[4, j] + 214 * v[3, j] -
                154 * v[2, j] + 45 * v[1, j]
            ) *
            (1 // 12) *
            inv_dr2
        dv[2, j] =
            (
                +1 * v[6, j] - 6 * v[5, j] + 14 * v[4, j] - 4 * v[3, j] -
                15 * v[2, j] + 10 * v[1, j]
            ) *
            (1 // 12) *
            inv_dr2
        dv[nx, j] =
            (
                -10 * v[nx-5, j] + 61 * v[nx-4, j] - 156 * v[nx-3, j] +
                214 * v[nx-2, j] - 154 * v[nx-1, j] + 45 * v[nx, j]
            ) *
            (1 // 12) *
            inv_dr2
        dv[nx-1, j] =
            (
                +1 * v[nx-5, j] - 6 * v[nx-4, j] + 14 * v[nx-3, j] -
                4 * v[nx-2, j] - 15 * v[nx-1, j] + 10 * v[nx, j]
            ) *
            (1 // 12) *
            inv_dr2
    end
    return nothing
end

"""
    filter!(
            v::AbstractArray{T},
            tmp::AbstractArray{T},
            eps_KO::Real
           ) where T<:Number

6th order Kreiss-Oliger dissipation.
"""
function filter!(v::AbstractArray{T}, tmp::AbstractArray{T}, eps_KO::Real) where {T<:Number}
    nx, ny = size(v)

    for j = 1:ny
        for i = 1:nx
            tmp[i, j] = v[i, j]
        end
    end

    for j = 1:ny
        for i = 4:nx-3
            v[i, j] = (
                (eps_KO / 64) * tmp[i-3, j] +
                (-6 * eps_KO / 64) * tmp[i-2, j] +
                (15 * eps_KO / 64) * tmp[i-1, j] +
                (1 - (20 * eps_KO / 64)) * tmp[i, j] +
                (15 * eps_KO / 64) * tmp[i+1, j] +
                (-6 * eps_KO / 64) * tmp[i+2, j] +
                (eps_KO / 64) * tmp[i+3, j]
            )
        end
    end

    return nothing
end

end
