"""
Background Newman-Penrose scalars
"""
module BackgroundNP

struct NP_0{T<:Real}
    mu_0  ::Array{Complex{T},2}
    tau_0 ::Array{Complex{T},2}
    pi_0  ::Array{Complex{T},2}
    rho_0 ::Array{Complex{T},2}
    eps_0 ::Array{Complex{T},2}
    psi2_0::Array{Complex{T},2}

    function NP_0{T}(;
        Rvals::Vector{T},
        Yvals::Vector{T},
        Cvals::Vector{T},
        Svals::Vector{T},
        bhm::Real,
        bhs::Real,
        cl::Real) where T<:Real 

        nx = length(Rvals)
        ny = length(Yvals)

        mu_0   = zeros(Complex{T},nx,ny) 
        tau_0  = zeros(Complex{T},nx,ny)  
        pi_0   = zeros(Complex{T},nx,ny)
        rho_0  = zeros(Complex{T},nx,ny)
        eps_0  = zeros(Complex{T},nx,ny)
        psi2_0 = zeros(Complex{T},nx,ny)

        for j=1:ny
            sy = Svals[j]
            cy = Cvals[j]

            for i=1:nx
                R = Rvals[i]

                mu_0[i,j]   = 1 / (-(cl^2) + im*bhs*R*cy)

                tau_0[i,j]  = (im*bhs*sy/sqrt(2)) / ((cl^2 - im*bhs*R*cy)^2)

                pi_0[i,j]   = - (im*bhs*sy/sqrt(2)) / (cl^4 + (bhs*cy*R)^2)

                rho_0[i,j]  = (
                    -  
                    0.5*(
                        cl^4 - 2*(cl^2)*bhm*R + (bhs*R)^2
                        )/(
                        ((cl^2 - im*bhs*R*cy)^2)*(cl^2 + im*bhs*R*cy)
                    )
                )

                eps_0[i,j]  = ( 
                    0.5*( 
                        (cl^2)*bhm - (bhs^2)*R - im*bhs*(cl^2-bhm*R)*cy 
                        )/( 
                        ((cl^2 - im*bhs*R*cy)^2)*(cl^2 + im*bhs*R*cy) 
                    )
                )

                psi2_0[i,j] = - bhm / ((cl^2 - im*bhs*R*cy)^3)
            end
        end
        return new{T}(mu_0,tau_0,pi_0,rho_0,eps_0,psi2_0)
    end
end

end
