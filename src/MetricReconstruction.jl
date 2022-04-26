"""
Metric reconstruction

For more details see      
   Loutrel et. al. Phys.Rev.D 103 (2021) 10, 104017, arXiv:2008.11770

   Ripley et. al. Phys.Rev.D 103 (2021) 1040180, arXiv:2010.00162
"""
module MetricReconstruction

include("./GHP.jl")
include("./BackgroundNP.jl")

using .Radial: set_d1! 
using .GHP: GHP_ops 
using .BackgroundNP: NP_0
  
function set_psi3_k!(
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

function set_lam_k!(
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

function set_psi2_k!(
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

function set_hmbmb_k!(
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

function set_pi_k!(
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

function set_hlmb_k!(
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

function set_muhll_k!(
      k,
      muhll,
      hlmb_edth,
      hlmb,
      pi_edth,
      pi_l,
      psi2,
      pi_nm,
      hmbmb_edth_nm,
      hmbmb_nm,
      hlmb_edth_nm,
      hlmb_nm
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

function set_k!(
      k,
      level,
      DR,
      dr::Float64
     )
   set_d1!(DR,level,dr)

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

function set_metric_recon_k!(;
      psi4_f ::Field,
      psi3_f ::Field,
      psi2_f ::Field,
      lam_f  ::Field,
      pi_f   ::Field,
      hmbmb_f::Field,
      hlmb_f ::Field,
      muhll_f::Field,
      m_ang::Int64,
      dr::Float64,
      R::Vector{Float64},
      Op::GHP_ops,
      NP::NP_0
     )
   psi4        = @view psi4_f.tmp[   :,:,m_ang] 
   psi4_k      = @view psi4_f.k[     :,:,m_ang] 
   psi4_raised = @view psi4_f.raised[:,:,m_ang] 
   psi4_edth   = @view psi4_f.edth[  :,:,m_ang] 

   psi3        = @view psi3_f.tmp[   :,:,m_ang] 
   psi3_k      = @view psi3_f.k[     :,:,m_ang] 
   psi3_raised = @view psi3_f.raised[:,:,m_ang] 
   psi3_edth   = @view psi3_f.edth[  :,:,m_ang] 
   psi3_DR     = @view psi3_f.rad_d1[:,:,m_ang] 

   psi2        = @view psi2_f.tmp[   :,:,m_ang] 
   psi2_k      = @view psi2_f.k[     :,:,m_ang] 
   psi2_dr     = @view psi2_f.rad_d1[:,:,m_ang] 

   lam        = @view lam_f.tmp[   :,:,m_ang] 
   lam_k      = @view lam_f.k[     :,:,m_ang] 
   lam_DR     = @view lam_f.rad_d1[:,:,m_ang] 

   pi_nm     = @view pi_f.tmp[   :,:,-m_ang]  
   pi_l      = @view pi_f.tmp[   :,:, m_ang] 
   pi_k      = @view pi_f.k[     :,:, m_ang] 
   pi_DR     = @view pi_f.rad_d1[:,:, m_ang] 
   pi_raised = @view pi_f.raised[:,:, m_ang] 
   pi_edth   = @view pi_f.edth[  :,:, m_ang] 

   hmbmb        = @view hmbmb_f.tmp[   :,:,m_ang] 
   hmbmb_k      = @view hmbmb_f.k[     :,:,m_ang] 
   hmbmb_DR     = @view hmbmb_f.rad_d1[:,:,m_ang] 

   hmbmb_nm        = @view hmbmb_f.tmp[:,:,-m_ang] 
   hmbmb_k_nm      = @view hmbmb_f.k[  :,:,-m_ang] 
   hmbmb_raised_nm = @view hmbmb_f.tmp[:,:,-m_ang] 
   hmbmb_edth_nm   = @view hmbmb_f.tmp[:,:,-m_ang] 

   hlmb        = @view hlmb_f.tmp[   :,:,m_ang] 
   hlmb_k      = @view hlmb_f.k[     :,:,m_ang] 
   hlmb_DR     = @view hlmb_f.rad_d1[:,:,m_ang] 
   hlmb_raised = @view hlmb_f.raised[:,:,m_ang] 
   hlmb_edth   = @view hlmb_f.edth[  :,:,m_ang] 

   hlmb_nm        = @view hlmb_f.tmp[:,:,-m_ang] 
   hlmb_k_nm      = @view hlmb_f.k[  :,:,-m_ang] 
   hlmb_raised_nm = @view hlmb_f.tmp[:,:,-m_ang] 
   hlmb_edth_nm   = @view hlmb_f.tmp[:,:,-m_ang] 

   muhll        = @view muhll_f.tmp[   :,:,m_ang] 
   muhll_k      = @view muhll_f.k[     :,:,m_ang] 
   muhll_DR     = @view muhll_f.rad_d1[:,:,m_ang] 

   ##===============
   GHP.set_edth!(edth=psi4_edth,spin=psi4_f.spin,boost=psi4_f.boost,level=psi4,DT=psi4_k,raised=psi4_raised,Op=Op) 

   set_psi3_k!(psi3_k, psi3, psi4, psi4_edth, R, NP)
   set_k!(psi3_k, psi3, psi3_DR, dr)
   
   ##===============
   set_lam_k!(lam_k, lam, psi4, R, NP)
   set_k!(lam_k, lam, lam_DR, dr)
   
   ##===============
   GHP.set_edth!(edth=psi3_edth,spin=psi3_f.spin,boost=psi3_f.boost,level=psi3,DT=psi3_k,raised=psi3_raised,Op=Op) 
   
   set_psi2_k!(psi2_k, psi2, psi3, psi3_edth, R, NP)
   set_k!(psi2_k, psi2, psi2_DR, dr)
   
   ##===============
   set_hmbmb_k!(hmbmb_k, hmbmb, lam, R, NP)
   set_k!(hmbmb_k, hmbmb, hmbmb_DR, dr)
   
   ##===============
   set_pi_k!(pi_k, lam, hmbmb, psi3, R, NP)
   set_k!(pi_k, pi_l, hmbmb_DR, dr)
   
   ##===============
   set_hlmb_k!(hlmb_k, lam, hmbmb, psi3, R, NP)
   set_k!(hlmb_k, hlmb, hmbmb_DR, dr)
   
   ##===============
   GHP.set_edth!(edth=hlmb_edth,        spin=hlmb_f.spin, boost=hlmb_f.boost, level=hlmb,        DT=hlmb_k,        raised=hlmb_raised,        Op=Op) 
   GHP.set_edth!(edth=pi_edth,          spin=pi_f.spin,   boost=pi_f.boost,   level=pi_l,        DT=pi_k,          raised=pi_raised,          Op=Op) 
   GHP.set_edth!(edth=hmbmb_edth_nm,    spin=hmbmb_f.spin,boost=hmbmb_f.boost,level=hmbmb_nm,    DT=hmbmb_k_nm,    raised=hmbmb_raised_nm,    Op=Op) 
   GHP.set_edth!(edth=hlmb_edth_edth_nm,spin=hlmb_f.spin, boost=hlmb.boost,   level=hlmb_edth_nm,DT=hlmb_edth_k_nm,raised=hlmb_edth_raised_nm,Op=Op) 
   
   set_muhll_k!(muhll_k, muhll, hlmb_edth, hlmb, pi_edth, pi_l, psi2,
                pi_nm, hmbmb_edth_nm, hmbmb_nm, hlmb_edth_nm, hlmb_nm
                R, NP
               )
   set_k!(muhll_k, muhll, muhll_DR, dr)
   
   return nothing
end

end
