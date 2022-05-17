"""
Background Newman-Penrose scalars
"""
module BackgroundNP

struct NP_0
   mu_0  ::Array{ComplexF64,2}
   tau_0 ::Array{ComplexF64,2}
   pi_0  ::Array{ComplexF64,2}
   rho_0 ::Array{ComplexF64,2}
   eps_0 ::Array{ComplexF64,2}
   psi2_0::Array{ComplexF64,2}
   
   function NP_0(;
         Rvals::Vector{Float64},
         Yvals::Vector{Float64},
         Cvals::Vector{Float64},
         Svals::Vector{Float64},
         bhm::Float64,
         bhs::Float64,
         cl::Float64) 
    
      nx = length(Rvals)
      ny = length(Yvals)
   
      mu_0   = zeros(ComplexF64,nx,ny) 
      tau_0  = zeros(ComplexF64,nx,ny)  
      pi_0   = zeros(ComplexF64,nx,ny)
      rho_0  = zeros(ComplexF64,nx,ny)
      eps_0  = zeros(ComplexF64,nx,ny)
      psi2_0 = zeros(ComplexF64,nx,ny)
      
      for j=1:ny
         sy = Svals[j]
         cy = Cvals[j]
         
         for i=1:nx
            R = Rvals[i]

            mu_0[i,j]   = 1.0 / (-(cl^2) + im*bhs*R*cy)
            
            tau_0[i,j]  = (im*bhs*sy/sqrt(2.0)) / ((cl^2 - im*bhs*R*cy)^2)
            
            pi_0[i,j]   = - (im*bhs*sy/sqrt(2.0)) / (cl^4 + (bhs*cy*R)^2)
            
            rho_0[i,j]  = (
               -  
               0.5*(
                    cl^4 - 2.0*(cl^2)*bhm*R + (bhs*R)^2
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
      return new(mu_0,tau_0,pi_0,rho_0,eps_0,psi2_0)
   end
end

end
