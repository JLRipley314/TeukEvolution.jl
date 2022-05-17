"""
Linear evolution of the NP equations: this includes the
Teukolsky equation and the Metric reconstruction equations.

For more details see      
   Loutrel et. al. Phys.Rev.D 103 (2021) 10, 104017, arXiv:2008.11770

   Ripley et. al. Phys.Rev.D 103 (2021) 1040180, arXiv:2010.00162
"""
module LinearEvolution 

include("Radial.jl")
include("GHP.jl")
include("BackgroundNP.jl")
include("Fields.jl")

using .Radial: set_d1! 
using .GHP: GHP_ops 
using .Fields: Field 
using .BackgroundNP: NP_0

export set_metric_recon_k!

function set_p_psi4_k!(
      k, 
      f_rd1, 
      f_rd2, 
      sph_lap, 
      p_rd1, 
      f, 
      p,
      A_pp,
      A_pq,
      B_pp,
      B_pq,
      B_pf
   )
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (   
               A_pp[i,j] * p_rd1[i,j]
             + A_pq[i,j] * f_rd2[i,j]
             + B_pp[i,j] * p[i,j]
             + B_pq[i,j] * f_rd1[i,j]
             + B_pf[i,j] * f[i,j]

             + sph_lap[i,j]
      )
      end
   end
   return nothing
end
   
function set_psi4_k!(k, p)
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = p[i,j]
      end
   end
   return nothing
end

function set_psi3_k!(
      k,
      psi3,
      psi4,
      psi4_edth,
      R::Vector{Float64},
      NP::NP_0
   )
   nx, ny = size(k)
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
   nx, ny = size(k)
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
   nx, ny = size(k)
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
   nx, ny = size(k)
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
   nx, ny = size(k)
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
   nx, ny = size(k)
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
      hlmb_nm,
      R::Vector{Float64},
      NP::NP_0
   )
   nx, ny = size(k)
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
      f,
      DR,
      dr::Float64
     )
   nx, ny = size(k)
   set_d1!(DR,f,dr)
   for j=1:ny
      for i=1:nx
         k[i,j] += (-  
                    ((R[i]/cl)^2)*DR[i,j] 
                    -  
                    (falloff*R[i]/(cl^2))*f[i,j]
                   )/( 
                      2.0+(4.0*bhm*R[i]/(cl^2)) 
                     )
      end
   end
   return nothing
end

function set_linear_k!(;
      psi4_f ::Field,
      psi4_p ::Field,
      psi3_f ::Field,
      psi2_f ::Field,
      lam_f  ::Field,
      pi_f   ::Field,
      hmbmb_f::Field,
      hlmb_f ::Field,
      muhll_f::Field,
      Evo::Evo_psi4,
      Op::GHP_ops,
      NP::NP_0,
      R::Vector{Float64}
      m_ang::Int64,
      dr::Float64
     )
   pmi = Op.mmap[m_ang]
   nmi = Op.mmap[-m_ang]
  
   psi4_tmp     = @view psi4_f.tmp[    :,:,pmi]
   psi4_k       = @view psi4_f.k[      :,:,pmi]
   psi4_rd1     = @view psi4_f.rad_d1[ :,:,pmi]
   psi4_rd2     = @view psi4_f.rad_d2[ :,:,pmi]
   psi4_sph_lap = @view psi4_f.sph_lap[:,:,pmi]
   psi4_raised  = @view psi4_f.raised[ :,:,pmi]
   psi4_edth    = @view psi4_f.edth[   :,:,pmi]

   p_psi4     = @view psi4_p.tmp[    :,:,pmi]
   p_psi4_k   = @view psi4_p.k[      :,:,pmi]
   p_psi4_rd1 = @view psi4_p.rad_d1[ :,:,pmi]

   psi3        = @view psi3_f.tmp[   :,:,pmi] 
   psi3_k      = @view psi3_f.k[     :,:,pmi] 
   psi3_raised = @view psi3_f.raised[:,:,pmi] 
   psi3_edth   = @view psi3_f.edth[  :,:,pmi] 
   psi3_DR     = @view psi3_f.rad_d1[:,:,pmi] 

   psi2        = @view psi2_f.tmp[   :,:,pmi] 
   psi2_k      = @view psi2_f.k[     :,:,pmi] 
   psi2_dr     = @view psi2_f.rad_d1[:,:,pmi] 

   lam        = @view lam_f.tmp[   :,:,pmi] 
   lam_k      = @view lam_f.k[     :,:,pmi] 
   lam_DR     = @view lam_f.rad_d1[:,:,pmi] 

   pi_nm     = @view pi_f.tmp[   :,:,nmi]  
   pi_l      = @view pi_f.tmp[   :,:,pmi] 
   pi_k      = @view pi_f.k[     :,:,pmi] 
   pi_DR     = @view pi_f.rad_d1[:,:,pmi] 
   pi_raised = @view pi_f.raised[:,:,pmi] 
   pi_edth   = @view pi_f.edth[  :,:,pmi] 

   hmbmb        = @view hmbmb_f.tmp[   :,:,pmi] 
   hmbmb_k      = @view hmbmb_f.k[     :,:,pmi] 
   hmbmb_DR     = @view hmbmb_f.rad_d1[:,:,pmi] 

   hmbmb_nm        = @view hmbmb_f.tmp[:,:,nmi] 
   hmbmb_k_nm      = @view hmbmb_f.k[  :,:,nmi] 
   hmbmb_raised_nm = @view hmbmb_f.tmp[:,:,nmi] 
   hmbmb_edth_nm   = @view hmbmb_f.tmp[:,:,nmi] 

   hlmb        = @view hlmb_f.tmp[   :,:,pmi] 
   hlmb_k      = @view hlmb_f.k[     :,:,pmi] 
   hlmb_DR     = @view hlmb_f.rad_d1[:,:,pmi] 
   hlmb_raised = @view hlmb_f.raised[:,:,pmi] 
   hlmb_edth   = @view hlmb_f.edth[  :,:,pmi] 

   hlmb_nm        = @view hlmb_f.tmp[:,:,nmi] 
   hlmb_k_nm      = @view hlmb_f.k[  :,:,nmi] 
   hlmb_raised_nm = @view hlmb_f.tmp[:,:,nmi] 
   hlmb_edth_nm   = @view hlmb_f.tmp[:,:,nmi] 

   muhll        = @view muhll_f.tmp[   :,:,pmi] 
   muhll_k      = @view muhll_f.k[     :,:,pmi] 
   muhll_DR     = @view muhll_f.rad_d1[:,:,pmi] 
   
   A_pp = @view Evo.A_pp[:,:,pmi]
   A_pq = @view Evo.A_pq[:,:,pmi]
   B_pp = @view Evo.B_pp[:,:,pmi]
   B_pq = @view Evo.B_pq[:,:,pmi]
   B_pf = @view Evo.B_pf[:,:,pmi]

   laplM = @view Evo.S_lapl[:,:,mi]
   
   set_d1!(  psi4_rd1,   psi4, dr)
   set_d1!(p_psi4_rd1, p_psi4, dr)
   set_d2!(  psi4_rd2,   psi4, dr)

   Sphere.angular_matrix_mult!(psi4_f_sph_lap, psi4, laplM)

   set_p_psi4_k!(p_psi4_k, f_rd1, psi4_rd2, psi4_sph_lap, p_psi4_rd1, psi4, p_psi4, A_pp, A_pq, B_pp, B_pq, B_pf)
   set_psi4_k!(    psi4_k, p_psi4)
   ##===============
   GHP.set_edth!(edth=psi4_edth,spin=psi4_f.spin,boost=psi4_f.boost,f=psi4,DT=psi4_k,raised=psi4_raised,Op=Op) 

   set_psi3_k!(psi3_k, psi3, psi4, psi4_edth, R, NP)
   set_k!(psi3_k, psi3, psi3_DR, dr)
   ##===============
   set_lam_k!(lam_k, lam, psi4, R, NP)
   set_k!(lam_k, lam, lam_DR, dr)
   ##===============
   GHP.set_edth!(edth=psi3_edth,spin=psi3_f.spin,boost=psi3_f.boost,f=psi3,DT=psi3_k,raised=psi3_raised,Op=Op) 
   
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
   GHP.set_edth!(edth=hlmb_edth,        spin=hlmb_f.spin, boost=hlmb_f.boost, f=hlmb,        DT=hlmb_k,        raised=hlmb_raised,        Op=Op) 
   GHP.set_edth!(edth=pi_edth,          spin=pi_f.spin,   boost=pi_f.boost,   f=pi_l,        DT=pi_k,          raised=pi_raised,          Op=Op) 
   GHP.set_edth!(edth=hmbmb_edth_nm,    spin=hmbmb_f.spin,boost=hmbmb_f.boost,f=hmbmb_nm,    DT=hmbmb_k_nm,    raised=hmbmb_raised_nm,    Op=Op) 
   GHP.set_edth!(edth=hlmb_edth_edth_nm,spin=hlmb_f.spin, boost=hlmb.boost,   f=hlmb_edth_nm,DT=hlmb_edth_k_nm,raised=hlmb_edth_raised_nm,Op=Op) 
   
   set_muhll_k!(muhll_k, muhll, hlmb_edth, hlmb, pi_edth, pi_l, psi2,
                pi_nm, hmbmb_edth_nm, hmbmb_nm, hlmb_edth_nm, hlmb_nm,
                R, NP
               )
   set_k!(muhll_k, muhll, muhll_DR, dr)
   
   return nothing
end

"""
Fourth order Runge-Kutta evolution of psi4
"""
function linear_evolution(;
      psi4_f ::Field,
      psi4_p ::Field,
      psi3_f ::Field,
      psi2_f ::Field,
      lam_f  ::Field,
      pi_f   ::Field,
      hmbmb_f::Field,
      hlmb_f ::Field,
      muhll_f::Field,
      Evo::Evo_psi4,
      Op::GHP_ops,
      NP::NP_0,
      R::Vector{Float64}
      m_ang::Int64,
      dr::Float64,
      dt::Float64
   )
   pmi = Op.mmap[m_ang]
   nmi = Op.mmap[-m_ang]
   
   f_psi4_n   = @view psi4_f.n[  :,:,pmi]
   f_psi4_tmp = @view psi4_f.tmp[:,:,pmi]
   f_psi4_np1 = @view psi4_f.np1[:,:,pmi]
   f_psi4_k   = @view psi4_f.k[  :,:,pmi]
   
   p_psi4_n   = @view psi4_p.n[  :,:,pmi]
   p_psi4_tmp = @view psi4_p.tmp[:,:,pmi]
   p_psi4_np1 = @view psi4_p.np1[:,:,pmi]
   p_psi4_k   = @view psi4_p.k[  :,:,pmi]
   
   psi3_n   = @view psi3_f.n[  :,:,pmi]
   psi3_tmp = @view psi3_f.tmp[:,:,pmi]
   psi3_np1 = @view psi3_f.np1[:,:,pmi]
   psi3_k   = @view psi3_f.k[  :,:,pmi]

   psi2_n   = @view psi2_f.n[  :,:,pmi]
   psi2_tmp = @view psi2_f.tmp[:,:,pmi]
   psi2_np1 = @view psi2_f.np1[:,:,pmi]
   psi2_k   = @view psi2_f.k[  :,:,pmi]

   lam_n   = @view lam_f.n[  :,:,pmi]
   lam_tmp = @view lam_f.tmp[:,:,pmi]
   lam_np1 = @view lam_f.np1[:,:,pmi]
   lam_k   = @view lam_f.k[  :,:,pmi]

   pi_n   = @view pi_f.n[  :,:,pmi]
   pi_tmp = @view pi_f.tmp[:,:,pmi]
   pi_np1 = @view pi_f.np1[:,:,pmi]
   pi_k   = @view pi_f.k[  :,:,pmi]

   hmbmb_n   = @view hmbmb_f.n[  :,:,pmi] 
   hmbmb_tmp = @view hmbmb_f.tmp[:,:,pmi] 
   hmbmb_np1 = @view hmbmb_f.np1[:,:,pmi] 
   hmbmb_k   = @view hmbmb_f.k[  :,:,pmi] 

   hmbmb_nm_n   = @view hmbmb_f.n[  :,:,nmi] 
   hmbmb_nm_tmp = @view hmbmb_f.tmp[:,:,nmi] 
   hmbmb_nm_np1 = @view hmbmb_f.np1[:,:,nmi] 
   hmbmb_nm_k   = @view hmbmb_f.k[  :,:,nmi] 

   hlmb_n   = @view hlmb_f.n[  :,:,pmi] 
   hlmb_tmp = @view hlmb_f.tmp[:,:,pmi] 
   hlmb_np1 = @view hlmb_f.np1[:,:,pmi] 
   hlmb_k   = @view hlmb_f.k[  :,:,pmi] 

   hlmb_nm_n   = @view hlmb_f.n[  :,:,nmi] 
   hlmb_nm_tmp = @view hlmb_f.tmp[:,:,nmi] 
   hlmb_nm_np1 = @view hlmb_f.np1[:,:,nmi] 
   hlmb_nm_k   = @view hlmb_f.k[  :,:,nmi] 

   muhll_n   = @view muhll_f.n[  :,:,pmi] 
   muhll_tmp = @view muhll_f.tmp[:,:,pmi] 
   muhll_np1 = @view muhll_f.np1[:,:,pmi] 
   muhll_k   = @view muhll_f.k[  :,:,pmi] 
   
   nx, ny = size(f_psi4_n)

   ## set tmp = n for first step

   for j=1:ny
      for i=1:nx
         f_psi4_tmp[i,j] = f_psi4_n[i,j]
         p_psi4_tmp[i,j] = p_psi4_n[i,j]
         psi3_tmp[i,j]   = psi3_n[i,j] 
         psi2_tmp[i,j]   = psi2_n[i,j] 
         lam_tmp[i,j]    = lam_n[i,j] 
         pi_tmp[i,j]     = pi_n[i,j] 
         hmbmb_tmp[i,j]  = hmbmb_n[i,j] 
         hlmb_tmp[i,j]   = hlmb_n[i,j] 
         muhll_tmp[i,j]  = muhll_n[i,j]
      end
   end
  
   for j=1:ny
      for i=1:nx
         f_psi4_tmp[i,j] = f_psi4_n[i,j] + half*dt*f_psi4_k[i,j]
         p_psi4_tmp[i,j] = p_psi4_n[i,j] + half*dt*p_psi4_k[i,j]
         psi3_tmp[i,j]   = psi3_n[i,j]   + half*dt*psi3_k[i,j] 
         psi2_tmp[i,j]   = psi2_n[i,j]   + half*dt*psi2_k[i,j] 
         lam_tmp[i,j]    = lam_n[i,j]    + half*dt*lam_k[i,j] 
         pi_tmp[i,j]     = pi_n[i,j]     + half*dt*pi_k[i,j] 
         hmbmb_tmp[i,j]  = hmbmb_n[i,j]  + half*dt*hmbmb_k[i,j] 
         hlmb_tmp[i,j]   = hlmb_n[i,j]   + half*dt*hlmb_k[i,j] 
         muhll_tmp[i,j]  = muhll_n[i,j]  + half*dt*muhll_k[i,j] 
         
         f_psi4_np1[i,j] = f_psi4_n[i,j] + sixth*dt*f_psi4_k[i,j]
         p_psi4_np1[i,j] = p_psi4_n[i,j] + sixth*dt*p_psi4_k[i,j]
         psi3_np1[i,j]   = psi3_n[i,j]   + sixth*dt*psi3_k[i,j] 
         psi2_np1[i,j]   = psi2_n[i,j]   + sixth*dt*psi2_k[i,j] 
         lam_np1[i,j]    = lam_n[i,j]    + sixth*dt*lam_k[i,j] 
         pi_np1[i,j]     = pi_n[i,j]     + sixth*dt*pi_k[i,j] 
         hmbmb_np1[i,j]  = hmbmb_n[i,j]  + sixth*dt*hmbmb_k[i,j] 
         hlmb_np1[i,j]   = hlmb_n[i,j]   + sixth*dt*hlmb_k[i,j] 
         muhll_np1[i,j]  = muhll_n[i,j]  + sixth*dt*muhll_k[i,j] 
      end
   end
  
   ## tmp

   for j=1:ny
      for i=1:nx
         f_psi4_tmp[i,j] = f_psi4_n[i,j] + half*dt*f_k[i,j]
         p_psi4_tmp[i,j] = p_psi4_n[i,j] + half*dt*p_k[i,j]
         psi3_tmp[i,j]   = psi3_n[i,j]   + half*dt*psi3_k[i,j] 
         psi2_tmp[i,j]   = psi2_n[i,j]   + half*dt*psi2_k[i,j] 
         lam_tmp[i,j]    = lam_n[i,j]    + half*dt*lam_k[i,j] 
         pi_tmp[i,j]     = pi_n[i,j]     + half*dt*pi_k[i,j] 
         hmbmb_tmp[i,j]  = hmbmb_n[i,j]  + half*dt*hmbmb_k[i,j] 
         hlmb_tmp[i,j]   = hlmb_n[i,j]   + half*dt*hlmb_k[i,j] 
         muhll_tmp[i,j]  = muhll_n[i,j]  + half*dt*muhll_k[i,j] 

         f_psi4_np1[i,j] += third*dt*f_k[i,j]
         p_psi4_np1[i,j] += third*dt*p_k[i,j]
         psi3_np1[i,j]   += third*dt*psi3_k[i,j] 
         psi2_np1[i,j]   += third*dt*psi2_k[i,j] 
         lam_np1[i,j]    += third*dt*lam_k[i,j] 
         pi_np1[i,j]     += third*dt*pi_k[i,j] 
         hmbmb_np1[i,j]  += third*dt*hmbmb_k[i,j] 
         hlmb_np1[i,j]   += third*dt*hlmb_k[i,j] 
         muhll_np1[i,j]  += third*dt*muhll_k[i,j] 
      end
   end
   
   ## tmp

   for j=1:ny
      for i=1:nx
         f_psi4_tmp[i,j] = f_psi4_n[i,j] + dt*f_psi4_k[i,j]
         p_psi4_tmp[i,j] = p_psi4_n[i,j] + dt*p_psi4_k[i,j]
         psi3_tmp[i,j]   = psi3_n[i,j]   + dt*psi3_k[i,j] 
         psi2_tmp[i,j]   = psi2_n[i,j]   + dt*psi2_k[i,j] 
         lam_tmp[i,j]    = lam_n[i,j]    + dt*lam_k[i,j] 
         pi_tmp[i,j]     = pi_n[i,j]     + dt*pi_k[i,j] 
         hmbmb_tmp[i,j]  = hmbmb_n[i,j]  + dt*hmbmb_k[i,j] 
         hlmb_tmp[i,j]   = hlmb_n[i,j]   + dt*hlmb_k[i,j] 
         muhll_tmp[i,j]  = muhll_n[i,j]  + dt*muhll_k[i,j] 

         f_psi4_np1[i,j] += third*dt*f_psi4_k[i,j]
         p_psi4_np1[i,j] += third*dt*p_psi4_k[i,j]
         psi3_np1[i,j]   += third*dt*psi3_k[i,j] 
         psi2_np1[i,j]   += third*dt*psi2_k[i,j] 
         lam_np1[i,j]    += third*dt*lam_k[i,j] 
         pi_np1[i,j]     += third*dt*pi_k[i,j] 
         hmbmb_np1[i,j]  += third*dt*hmbmb_k[i,j] 
         hlmb_np1[i,j]   += third*dt*hlmb_k[i,j] 
         muhll_np1[i,j]  += third*dt*muhll_k[i,j] 
      end
   end
   
   ## tmp

   for j=1:ny
      for i=1:nx
         f_k[i,j] = p_tmp[i,j]
         
         f_psi4_np1[i,j] += sixth*dt*f_psi4_k[i,j]
         p_psi4_np1[i,j] += sixth*dt*p_psi4_k[i,j]   
         psi3_np1[i,j]   += third*dt*psi3_k[i,j] 
         psi2_np1[i,j]   += third*dt*psi2_k[i,j] 
         lam_np1[i,j]    += third*dt*lam_k[i,j] 
         pi_np1[i,j]     += third*dt*pi_k[i,j] 
         hmbmb_np1[i,j]  += third*dt*hmbmb_k[i,j] 
         hlmb_np1[i,j]   += third*dt*hlmb_k[i,j] 
         muhll_np1[i,j]  += third*dt*muhll_k[i,j] 
      end
   end
  
   ## Only filter psi4 for now

   Radial.filter!(f_psi4_np1,f_psi4_tmp,0.5)
   Radial.filter!(p_psi4_np1,p_psi4_tmp,0.5) 
   for j=1:ny
      for i=1:nx
         f_psi4_tmp[i,j] = f_psi4_np1[i,j] 
         p_psi4_tmp[i,j] = p_psi4_np1[i,j] 
      end
   end
   Sphere.angular_matrix_mult!(f_psi4_np1,f_psi4_tmp,fltrM)
   Sphere.angular_matrix_mult!(p_psi4_np1,p_psi4_tmp,fltrM)

   return nothing
end

end
