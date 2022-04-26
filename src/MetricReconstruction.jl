"""
Metric reconstruction

For more details see      
   Loutrel et. al. Phys.Rev.D 103 (2021) 10, 104017, arXiv:2008.11770

   Ripley et. al. Phys.Rev.D 103 (2021) 1040180, arXiv:2010.00162
"""
module MetricReconstruction

include("./BackgroundNP.jl")
using .BackgroundNP: NP_0
  
function k_psi3!(
      k,
      psi3,
      psi4,
      psi4_edth,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    4.0*R[i]*NP.mu_0[i,j]*psi3[i,j]
                    -  
                    R[i]*NP.tau_0[i,j]*psi4[i,j]
                    +  
                    psi4_edth[i,j]
                   )
      end
   end
   return nothing
end

function k_lam!(
      k,
      lam,
      psi4,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    R[i]*(NP.mu_0[i,j] + conj(NP.mu_0[i,j]))*lam[i,j] 
                    - 
                    psi4[i,j]
                   )
      end
   end
   return nothing
end

function k_psi2!(
      k,
      psi2,
      psi3,
      psi3_edth,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    3.0*R[i]*NP.mu_0[i,j] *psi2[i,j] 
                    -  
                    2.0*R[i]*NP.tau_0[i,j]*psi3[i,j] 
                    +  
                    psi3_edth[i,j]
                   )
      end
   end
   return nothing
end

function k_hmbmb!(
      k,
      hmbmb,
      lam,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = ( 
                    R[i]*(NP.mu_0[i,j] - conj(NP.mu_0[i,j]))*hmbmb[i,j] 
                    -  
                    2.0*lam[i,j]
                   ) 
      end
   end
   return nothing
end

function k_pi!(
      k,
      lam,
      hmbmb,
      psi3,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    R[i]*(conj(NP.pi_0[i,j]) + NP.ta_0[i,j])*lam[i,j] 
                    +  
                    (R[i]^2)*0.5*NP.mu_0[i,j]*( 
                                               conj(NP.pi_0[i,j]) 
                                               +  
                                               NP.ta_0[i,j] 
                                              )*hmbmb[i,j]
                    -  
                    psi3[i,j]
                   )
      end
   end
   return nothing
end

function k_hlmb!(
      k,
      lam,
      hmbmb,
      psi3,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    R[i]*conj(NP.mu_0[i,j])*hlmb[i,j] 
                    -  
                    2.0*pi_l[i,j] 
                    -  
                    R[i]*NP.tau_0[i,j]*hmbmb[i,j]
                   )
      end
   end
   return nothing
end

function k_muhll!(
      k,
      lam,
      hmbmb,
      psi3,
      R::Vector{Float64},
      NP::NP_0
   )
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                    R[i]*conj(NP.mu_0[i,j])*muhll[i,j] 
                    -  
                    R[i]*NP.mu_0[i,j]*hlmb_edth[i,j] 
                    -  
                    (R[i]^2)*NP.mu_0[i,j]*( 
                                           conj(NP.pi_0[i,j]) 
                                           +  
                                           2.0*NP.tau_0[i,j] 
                                          )*hlmb[i,j]
         
                    -  
                    2.0*pi_edth[i,j] 
                    -  
                    2.0*R[i]*conj(NP.pi_0[i,j])*pi_l[i,j] 
                    -  
                    2.0*psi2[i,j] 
                    -  
                    2.0*R[i]*NP.pi_0[i,j]*conj(pi_nm[i,j]) 
                    -  
                    R[i]*NP.pi_0[i,j]*conj(hmbmb_edth_nm[i,j]) 
                    +  
                    (R[i,j]^2)*(NP.pi_0[i,j]^2)*conj(hmbmb_nm[i,j]) 
                    +  
                    R[i]*NP.mu_0[i,j]*conj(hlmb_edth_nm[i,j])  
                    +  
                    (R[i]^2)*(-  
                              3.0*NP.mu_0[i,j]*NP.pi_0[i,j] 
                              +  
                              2.0*conj(NP.mu_0[i,j])*NP.pi_0[i,j] 
                              -  
                              2.0*NP.mu_0[i,j]*conj(NP.ta_0[i,j]) 
                             )*conj(hlmb_nm[i,j])
                   )
      end
   end
   return nothing
end

function set_k(
      kp,
      level,
      DR
      )

   for j=1:ny
      for i=1:nx
         k[i,j] += (-  
                     ((R[i]/cl)^2)*DR[i,j] 
                     -  
                     (falloff*R[i]/(cl^2))*level[i,j]
                    )/( 
                       2.0+(4.0*bhm*R[i]/(cl^2)) 
                      )
      end
   end
   return nothing
end

end
