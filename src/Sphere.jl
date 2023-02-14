"""
Spin-weighted spherical harmonics and associated functions and operators.
"""
module Sphere

import FastGaussQuadrature as FGQ

export Y_vals,
    cos_vals,
    sin_vals,
    swal,
    swal_laplacian_matrix,
    swal_raising_matrix,
    swal_lowering_matrix,
    swal_filter_matrix,
    swal_killtop_matrix,
    angular_matrix_mult!


"""
    jacobi(x, n::Integer, a, b)

    The Jacobi polynomial.
"""
function jacobi(x, n::Integer, a, b)
    ox = one(x)
    zx = zero(x)
    if n == 0
        return ox
    elseif n == 1
        return (ox / 2) * (a - b + (a + b + 2) * x)
    end

    p0 = ox
    p1 = (ox / 2) * (a - b + (a + b + 2) * x)
    p2 = zx

    for i = 1:(n-1)
        a1 = 2 * (i + 1) * (i + a + b + 1) * (2 * i + a + b)
        a2 = (2 * i + a + b + 1) * (a * a - b * b)
        a3 = (2 * i + a + b) * (2 * i + a + b + 1) * (2 * i + a + b + 2)
        a4 = 2 * (i + a) * (i + b) * (2 * i + a + b + 2)
        p2 = (ox / a1) * ((a2 + a3 * x) * p1 - a4 * p0)

        p0 = p1
        p1 = p2
    end

    return p2
end


"""
    num_l(
       ny::Integer
       )::Integer

Compute the number of l angular values given ny angular collocation points.

num_l = ny - 2*(max_s+max_m), where max_s and max_m
are hard coded in Sphere.jl
"""
function num_l(ny::Integer,max_s::Integer=2,max_m::Integer=6)::Integer
    @assert ny > (2 * max_s + 2 * max_m) + 4
    return ny - 2 * max_s - 2 * max_m
end

"""
    Y_vals(
       ny::Integer
       )::Vector{<:Real}

Compute the Gauss-Legendre points (y=-cosθ) over the interval [-1,1].
"""
function Y_vals(ny::Integer)::Vector{<:Real}
    roots, weights = FGQ.gausslegendre(ny)
    return roots
end

"""
    inner_product(
       v1::Vector{T},
       v2::Vector{T},
       )::T where T<:Number

Compute the inner product over the interval [-1,1] using
Gauss quadrature. Returns the discretization of
∫ dy v1 conj(v2)
"""
function inner_product(v1::Array{T,1}, v2::Array{T,1})::T where {T<:Number}
    ny = size(v1)[1]
    roots, weights = FGQ.gausslegendre(ny)
    s = 0.0
    for j = 1:ny
        s += weights[j] * v1[j] * conj(v2[j])
    end

    return s
end

"""
    cos_vals(ny::Integer)::Vector{<:Real}

Compute cos(y) at the Gauss-Legendre points over the interval [-1,1].
"""
function cos_vals(ny::Integer)::Vector{<:Real}
    roots, weights = FGQ.gausslegendre(ny)
    return [-pt for pt in roots]
end

"""
    sin_vals(ny::Integer)::Vector{<:Real}

Compute sin(y) at the Gauss-Legendre points over interval [-1,1].
"""
function sin_vals(ny::Integer)::Vector{<:Real}
    roots, weights = FGQ.gausslegendre(ny)
    return [sqrt(1.0 - pt) * sqrt(1.0 + pt) for pt in roots]
end

"""
    swal(
          spin::Integer,
          m_ang::Integer,
          l_ang::Integer,
          y::Real
       )::Real

Compute the spin-weighted associated Legendre function Y^s_{lm}(y).
"""
function swal(spin::TI, m_ang::TI, l_ang::TI, y::TR) where {TI<:Integer,TR<:Number}
    @assert l_ang >= abs(m_ang)

    al = TI(abs(m_ang - spin))
    be = TI(abs(m_ang + spin))
    @assert((al + be) % 2 == 0)
    n = TI(l_ang - (al + be) // 2)

    if n < 0
        return convert(TR, 0)
    end

    norm = sqrt(
        (2 * n + al + be + 1) * (2^(-al - be - 1.0)) * factorial(big(n + al + be)) /
        factorial(big(n + al)) * factorial(big(n)) / factorial(big(n + be)),
    )
    norm *= (-1)^(max(m_ang, -spin))

    return TR(norm) * ((1 - y)^(al / 2.0)) * ((1 + y)^(be / 2.0)) * jacobi(y, n, al, be)
end

"""
    swal_vals(
          ny::Integer,
          spin::Integer,
          m_ang::Integer,
       )::Matrix{<:Real}

Compute the matrix swal^s_{lm}(y) at the Gauss-Legendre points,
over a grid of l angular values (the angular number m is fixed).
"""
function swal_vals(ny::Integer, spin::Integer, m_ang::Integer)::Matrix{<:Real}

    roots, weights = FGQ.gausslegendre(ny)
    nl = num_l(ny)
    vals = zeros(ny, nl)
    lmin = max(abs(spin), abs(m_ang))

    for k = 1:nl
        l_ang = k - 1 + lmin
        for j = 1:length(roots)
            vals[j, k] = swal(spin, m_ang, l_ang, roots[j])
        end
    end
    return vals
end

"""
    swal_laplacian_matrix(
          ny::Integer,
          spin::Integer,
          m_ang::Integer
       )::Matrix{<:Real}

Compute the matrix to compute the 
spin-weighted spherical harmonic laplacian operator.
"""
function swal_laplacian_matrix(ny::Integer, spin::Integer, m_ang::Integer)::Matrix{<:Real}

    yv, wv = FGQ.gausslegendre(ny)
    nl = num_l(ny)
    lmin = max(abs(spin), abs(m_ang))
    lap = zeros(ny, ny)

    for j = 1:ny
        for i = 1:ny
            for k = 1:nl
                l = k - 1 + lmin
                lap[j, i] -=
                    (l - spin) *
                    (l + spin + 1.0) *
                    (swal(spin, m_ang, l, yv[i]) * swal(spin, m_ang, l, yv[j]))
            end
            lap[j, i] *= wv[j]
        end
    end
    return lap
end

"""
    swal_raising_matrix(
          ny::Integer,
          spin::Integer,
          m_ang::Integer
       )::Matrix{<:Real}

Compute the matrix to raise spin-weighted spherical harmonics.
"""
function swal_raising_matrix(ny::Integer, spin::Integer, m_ang::Integer)::Matrix{<:Real}
    yv, wv = FGQ.gausslegendre(ny)
    nl = num_l(ny)
    lmin = max(abs(spin), abs(spin + 1), abs(m_ang))
    raise = zeros(ny, ny)

    for (i, yi) in enumerate(yv)
        for (j, yj) in enumerate(yv)
            for k = 1:nl
                l = k - 1 + lmin
                raise[j, i] += (
                    sqrt((l - spin) * (l + spin + 1)) *
                    wv[j] *
                    swal(spin, m_ang, l, yj) *
                    swal(spin + 1, m_ang, l, yi)
                )
            end
        end
    end
    return raise
end

"""
    swal_lowering_matrix(
            ny::Integer,
            spin::Integer,
            m_ang::Integer
        )::Matrix{<:Real}


Compute the matrix to lower spin-weighted spherical harmonics.
"""
function swal_lowering_matrix(ny::Integer, spin::Integer, m_ang::Integer)::Matrix{<:Real}

    yv, wv = FGQ.gausslegendre(ny)

    nl = num_l(ny)
    lmin = max(abs(spin), abs(spin - 1), abs(m_ang))

    lower = zeros(Float64, ny, ny)

    for (i, yi) in enumerate(yv)
        for (j, yj) in enumerate(yv)
            for k = 1:nl
                l = k - 1 + lmin
                lower[j, i] += (
                    -sqrt((l + spin) * (l - spin + 1.0)) *
                    wv[j] *
                    swal(spin, m_ang, l, yj) *
                    swal(spin - 1, m_ang, l, yi)
                )
            end
        end
    end

    return lower
end

"""
    swal_filter_matrix(
          ny::Integer,
          spin::Integer,
          m_ang::Integer
       )::Matrix{<:Real}

Compute the matrix to compute low pass filter.
Multiply the matrix on the left: v[i]*M[i,j] -> v[j]
"""
function swal_filter_matrix(ny::Integer, spin::Integer, m_ang::Integer)::Matrix{<:Real}

    yv, wv = FGQ.gausslegendre(ny)
    nl = num_l(ny)
    filter = zeros(Float64, ny, ny)
    lmin = max(abs(spin), abs(m_ang))

    for j = 1:ny
        for i = 1:ny
            for l = lmin:(nl-1+lmin)
                filter[j, i] +=
                    exp(-30.0 * (l / (nl - 1.0))^10) *
                    (swal(spin, m_ang, l, yv[i]) * swal(spin, m_ang, l, yv[j]))
            end
            filter[j, i] *= wv[j]
        end
    end
    return filter
end

"""
    swal_killtop_matrix(
          ny::Integer,
          spin::Integer,
          m_ang::Integer
       )::Matrix{<:Real}

Compute the matrix to compute killtopN low pass filter.
Multiply the matrix on the left: v[i]*M[i,j] -> v[j]
"""
function swal_killtop_matrix(ny::Integer, spin::Integer, m_ang::Integer, N::Integer)::Matrix{<:Real}

    yv, wv = FGQ.gausslegendre(ny)
    nl = num_l(ny)
    filter = zeros(Float64, ny, ny)
    lmin = max(abs(spin), abs(m_ang))
    @assert nl > N+1

    for j = 1:ny
        for i = 1:ny
            for l = lmin:(nl-1+lmin-N)
                filter[j, i] +=
                    (swal(spin, m_ang, l, yv[i]) * swal(spin, m_ang, l, yv[j]))
            end
            filter[j, i] *= wv[j]
        end
    end
    return filter
end

"""
    angular_matrix_mult!(
            f_m::AbstractArray{<:Number},
            f  ::AbstractArray{<:Number},
            mat::Matrix{<:Number}
           )::Nothing

Compute matrix multiplication in the angular direction.
"""
function angular_matrix_mult!(
    f_m::AbstractArray{<:Number},
    f::AbstractArray{<:Number},
    mat::Matrix{<:Number},
)::Nothing
    nx, ny = size(f)
    for j = 1:ny
        for i = 1:nx
            f_m[i, j] = 0.0
            for k = 1:ny
                f_m[i, j] += f[i, k] * mat[k, j]
            end
        end
    end
    return nothing
end

end
