"""
Linear evolution of the NP equations: this includes the
Teukolsky equation and the Metric reconstruction equations.

For more details see      
   Loutrel et. al. Phys.Rev.D 103 (2021) 10, 104017, arXiv:2008.11770

   Ripley et. al. Phys.Rev.D 103 (2021) 1040180, arXiv:2010.00162
"""
module LinearEvolution 

include("Radial.jl")
include("Sphere.jl")
include("GHP.jl")
include("BackgroundNP.jl")
include("Fields.jl")
include("Evolution.jl")

using .Radial: set_d1!, set_d2!, filter! 
using .Sphere: angular_matrix_mult!
using .GHP: GHP_ops 
using .Fields: Field 
using .BackgroundNP: NP_0
using .Evolution: Evo_lin_f

export Linear_evolution!, Set_independent_residuals

const half  = 1.0/2.0
const third = 1.0/3.0
const sixth = 1.0/6.0

function set_psi4_p_k!(;
      kp::Array{ComplexF64,2}, 
      f_rd1::Array{ComplexF64,2}, 
      f_rd2::Array{ComplexF64,2}, 
      sph_lap::Array{ComplexF64,2}, 
      p_rd1::Array{ComplexF64,2}, 
      f::Array{ComplexF64,2}, 
      p::Array{ComplexF64,2},
      A_pp::Array{Float64,2},
      A_pq::Array{Float64,2},
      B_pp::Array{ComplexF64,2},
      B_pq::Array{ComplexF64,2},
      B_pf::Array{ComplexF64,2}
   )::Nothing
   nx, ny = size(kp)
   
   for j=1:ny
      for i=1:nx
         kp[i,j] = (   
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
   
function set_psi4_f_k!(;
      k::Array{ComplexF64,2}, 
      p::Array{ComplexF64,2}
     )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = p[i,j]
      end
   end
   return nothing
end

function set_psi3_k!(;
      k::Array{ComplexF64},
      psi3::Array{ComplexF64},
      psi4::Array{ComplexF64},
      psi4_edth::Array{ComplexF64},
      mu_0::Array{ComplexF64},
      tau_0::Array{ComplexF64},
      R::Vector{Float64},
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (- 
                   4.0*R[i]*mu_0[i,j]*psi3[i,j]
                   -  
                   R[i]*tau_0[i,j]*psi4[i,j]
                   +  
                   psi4_edth[i,j]
                  )
      end
   end
   return nothing
end

function set_lam_k!(;
      k::Array{ComplexF64,2},
      lam::Array{ComplexF64,2},
      psi4::Array{ComplexF64,2},
      mu_0::Array{ComplexF64},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                   R[i]*(mu_0[i,j] + conj(mu_0[i,j]))*lam[i,j] 
                   - 
                   psi4[i,j]
                  )
      end
   end
   return nothing
end

function set_psi2_k!(;
      k::Array{ComplexF64,2},
      psi2::Array{ComplexF64,2},
      psi3::Array{ComplexF64,2},
      psi3_edth::Array{ComplexF64,2},
      mu_0::Array{ComplexF64,2},
      tau_0::Array{ComplexF64,2},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                   3.0*R[i]*mu_0[i,j] *psi2[i,j] 
                   -  
                   2.0*R[i]*tau_0[i,j]*psi3[i,j] 
                   +  
                   psi3_edth[i,j]
                  )
      end
   end
   return nothing
end

function set_hmbmb_k!(;
      k::Array{ComplexF64,2},
      hmbmb::Array{ComplexF64,2},
      lam::Array{ComplexF64,2},
      mu_0::Array{ComplexF64,2},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = ( 
                   R[i]*(mu_0[i,j] - conj(mu_0[i,j]))*hmbmb[i,j] 
                   -  
                   2.0*lam[i,j]
                  ) 
      end
   end
   return nothing
end

function set_pi_k!(;
      k::Array{ComplexF64,2},
      lam::Array{ComplexF64,2},
      hmbmb::Array{ComplexF64,2},
      psi3::Array{ComplexF64,2},
      mu_0::Array{ComplexF64,2},
      tau_0::Array{ComplexF64,2},
      pi_0::Array{ComplexF64,2},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                   R[i]*(conj(pi_0[i,j]) + tau_0[i,j])*lam[i,j] 
                   +  
                   (R[i]^2)*0.5*mu_0[i,j]*( 
                                           conj(pi_0[i,j]) 
                                           +  
                                           tau_0[i,j] 
                                          )*hmbmb[i,j]
                   -  
                   psi3[i,j]
                  )
      end
   end
   return nothing
end

function set_hlmb_k!(;
      k::Array{ComplexF64,2},
      hlmb::Array{ComplexF64,2},
      hmbmb::Array{ComplexF64,2},
      pi_l::Array{ComplexF64,2},
      mu_0::Array{ComplexF64,2},
      tau_0::Array{ComplexF64,2},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                   R[i]*conj(mu_0[i,j])*hlmb[i,j] 
                   -  
                   2.0*pi_l[i,j] 
                   -  
                   R[i]*tau_0[i,j]*hmbmb[i,j]
                  )
      end
   end
   return nothing
end

function set_muhll_k!(;
      k::Array{ComplexF64,2},
      muhll::Array{ComplexF64,2},
      hlmb_edth::Array{ComplexF64,2},
      hlmb::Array{ComplexF64,2},
      pi_edth::Array{ComplexF64,2},
      pi_l::Array{ComplexF64,2},
      psi2::Array{ComplexF64,2},
      pi_nm::Array{ComplexF64,2},
      hmbmb_edth_nm::Array{ComplexF64,2},
      hmbmb_nm::Array{ComplexF64,2},
      hlmb_edth_nm::Array{ComplexF64,2},
      hlmb_nm::Array{ComplexF64,2},
      mu_0::Array{ComplexF64,2},
      tau_0::Array{ComplexF64,2},
      pi_0::Array{ComplexF64,2},
      R::Vector{Float64}
   )::Nothing
   nx, ny = size(k)
   for j=1:ny
      for i=1:nx
         k[i,j] = (-  
                   R[i]*conj(mu_0[i,j])*muhll[i,j] 
                   -  
                   R[i]*mu_0[i,j]*hlmb_edth[i,j] 
                   -  
                   (R[i]^2)*mu_0[i,j]*( 
                                       conj(pi_0[i,j]) 
                                       +  
                                       2.0*tau_0[i,j] 
                                      )*hlmb[i,j]
                   -  
                   2.0*pi_edth[i,j] 
                   -  
                   2.0*R[i]*conj(pi_0[i,j])*pi_l[i,j] 
                   -  
                   2.0*psi2[i,j] 
                   -  
                   2.0*R[i]*pi_0[i,j]*conj(pi_nm[i,j]) 
                   -  
                   R[i]*pi_0[i,j]*conj(hmbmb_edth_nm[i,j]) 
                   +  
                   (R[i]^2)*(pi_0[i,j]^2)*conj(hmbmb_nm[i,j]) 
                   +  
                   R[i]*mu_0[i,j]*conj(hlmb_edth_nm[i,j])  
                   +  
                   (R[i]^2)*(-  
                             3.0*mu_0[i,j]*pi_0[i,j] 
                             +  
                             2.0*conj(mu_0[i,j])*pi_0[i,j] 
                             -  
                             2.0*mu_0[i,j]*conj(tau_0[i,j]) 
                            )*conj(hlmb_nm[i,j])
                  )
      end
   end
   return nothing
end

function finish_linear_reconstruction_k!(;
      k::Array{ComplexF64,2},
      f::Array{ComplexF64,2},
      DR::Array{ComplexF64,2},
      R::Vector{Float64},
      falloff::Int64,
      bhm::Float64,
      cl::Float64,
      dr::Float64
     )::Nothing
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
      psi4_f,  psi4_f_l::Array{ComplexF64,2},
      psi4_p,  psi4_p_l::Array{ComplexF64,2},
      psi3_f,  psi3_l  ::Array{ComplexF64,2},
      psi2_f,  psi2_l  ::Array{ComplexF64,2},
      lam_f,   lam_l   ::Array{ComplexF64,2},
      pi_f,    pi_l    ::Array{ComplexF64,2}, pi_nm_l::Array{ComplexF64,2},
      hmbmb_f,    hmbmb_l   ::Array{ComplexF64,2},
      hmbmb_f_nm, hmbmb_nm_l::Array{ComplexF64,2},
      hlmb_f,     hlmb_l    ::Array{ComplexF64,2},
      hlmb_f_nm,  hlmb_nm_l ::Array{ComplexF64,2},
      muhll_f, muhll_l::Array{ComplexF64,2},
      Evo, Op, NP,
      R::Vector{Float64},
      m_ang::Int64,
      bhm::Float64,
      cl::Float64,
      dr::Float64
     )::Nothing
   psi4_f_k       = psi4_f.k
   psi4_f_rd1     = psi4_f.rad_d1
   psi4_f_rd2     = psi4_f.rad_d2
   psi4_f_sph_lap = psi4_f.sph_lap
   psi4_f_raised  = psi4_f.raised
   psi4_f_edth    = psi4_f.edth

   psi4_p_k   = psi4_p.k
   psi4_p_rd1 = psi4_p.rad_d1

   psi3_k      = psi3_f.k 
   psi3_raised = psi3_f.raised 
   psi3_edth   = psi3_f.edth 
   psi3_DR     = psi3_f.rad_d1 

   psi2_k      = psi2_f.k 
   psi2_DR     = psi2_f.rad_d1 

   lam_k      = lam_f.k 
   lam_DR     = lam_f.rad_d1 

   pi_k      = pi_f.k 
   pi_DR     = pi_f.rad_d1 
   pi_raised = pi_f.raised 
   pi_edth   = pi_f.edth 

   hmbmb_k      = hmbmb_f.k 
   hmbmb_DR     = hmbmb_f.rad_d1 

   hmbmb_k_nm      = hmbmb_f_nm.k 
   hmbmb_raised_nm = hmbmb_f_nm.raised
   hmbmb_edth_nm   = hmbmb_f_nm.edth

   hlmb_k      = hlmb_f.k 
   hlmb_DR     = hlmb_f.rad_d1 
   hlmb_raised = hlmb_f.raised 
   hlmb_edth   = hlmb_f.edth 

   hlmb_k_nm      = hlmb_f_nm.k 
   hlmb_raised_nm = hlmb_f_nm.raised 
   hlmb_edth_nm   = hlmb_f_nm.edth 

   muhll_k      = muhll_f.k 
   muhll_DR     = muhll_f.rad_d1 

   A_pp = Evo.A_pp
   A_pq = Evo.A_pq
   B_pp = Evo.B_pp
   B_pq = Evo.B_pq
   B_pf = Evo.B_pf
   
   laplM = Evo.S_lapl
   
   mu_0  = NP.mu_0
   tau_0 = NP.tau_0
   pi_0  = NP.pi_0
  
   set_d1!(psi4_f_rd1, psi4_f_l, dr)
   set_d1!(psi4_p_rd1, psi4_p_l, dr)
   set_d2!(psi4_f_rd2, psi4_f_l, dr)

   angular_matrix_mult!(psi4_f_sph_lap, psi4_f_l, laplM)

   set_psi4_p_k!(kp=psi4_p_k, 
      f_rd1=psi4_f_rd1, f_rd2=psi4_f_rd2, 
      sph_lap=psi4_f_sph_lap, p_rd1=psi4_p_rd1, 
      f=psi4_f_l, p=psi4_p_l, 
      A_pp=A_pp, A_pq=A_pq, B_pp=B_pp, B_pq=B_pq, B_pf=B_pf)
   
   set_psi4_f_k!(k=psi4_f_k, p=psi4_p_l)
   ##===============
   GHP.set_edth!(edth=psi4_f_edth,spin=psi4_f.spin,boost=psi4_f.boost,m_ang=m_ang,f=psi4_f_l,DT=psi4_f_k,raised=psi4_f_raised,Op=Op) 

   set_psi3_k!(k=psi3_k, psi3=psi3_l, psi4=psi4_f_l, psi4_edth=psi4_f_edth, mu_0=mu_0, tau_0=tau_0, R=R)
   finish_linear_reconstruction_k!(k=psi3_k, f=psi3_l, DR=psi3_DR, R=R, falloff=psi3_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   set_lam_k!(k=lam_k, lam=lam_l, psi4=psi4_f_l, mu_0=mu_0, R=R)
   finish_linear_reconstruction_k!(k=lam_k, f=lam_l, DR=lam_DR, R=R, falloff=lam_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   GHP.set_edth!(edth=psi3_edth,spin=psi3_f.spin,boost=psi3_f.boost,m_ang=m_ang,f=psi3_l,DT=psi3_k,raised=psi3_raised,Op=Op) 
   
   set_psi2_k!(k=psi2_k, psi2=psi2_l, psi3=psi3_l, psi3_edth=psi3_edth, mu_0=mu_0, tau_0=tau_0, R=R)
   finish_linear_reconstruction_k!(k=psi2_k, f=psi2_l, DR=psi2_DR, R=R, falloff=psi2_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   set_hmbmb_k!(k=hmbmb_k, hmbmb=hmbmb_l, lam=lam_l, mu_0=mu_0, R=R)
   finish_linear_reconstruction_k!(k=hmbmb_k, f=hmbmb_l, DR=hmbmb_DR, R=R, falloff=hmbmb_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   set_pi_k!(k=pi_k, lam=lam_l, hmbmb=hmbmb_l, psi3=psi3_l, mu_0=mu_0, tau_0=tau_0, pi_0=pi_0, R=R)
   finish_linear_reconstruction_k!(k=pi_k, f=pi_l, DR=pi_DR, R=R, falloff=pi_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   set_hlmb_k!(k=hlmb_k, hlmb=hlmb_l, hmbmb=hmbmb_l, pi_l=pi_l, mu_0=mu_0, tau_0=tau_0, R=R)
   finish_linear_reconstruction_k!(k=hlmb_k, f=hlmb_l, DR=hlmb_DR, R=R, falloff=hlmb_f.falloff, bhm=bhm, cl=cl, dr=dr)
   ##===============
   GHP.set_edth!(edth=hlmb_edth,    spin=hlmb_f.spin, boost=hlmb_f.boost, m_ang= m_ang,f=hlmb_l,    DT=hlmb_k,    raised=hlmb_raised,    Op=Op) 
   GHP.set_edth!(edth=pi_edth,      spin=pi_f.spin,   boost=pi_f.boost,   m_ang= m_ang,f=pi_l,      DT=pi_k,      raised=pi_raised,      Op=Op) 
   GHP.set_edth!(edth=hmbmb_edth_nm,spin=hmbmb_f.spin,boost=hmbmb_f.boost,m_ang=-m_ang,f=hmbmb_nm_l,DT=hmbmb_k_nm,raised=hmbmb_raised_nm,Op=Op) 
   GHP.set_edth!(edth=hlmb_edth_nm, spin=hlmb_f.spin, boost=hlmb_f.boost, m_ang=-m_ang,f=hlmb_nm_l, DT=hlmb_k_nm, raised=hlmb_raised_nm, Op=Op) 
   
   set_muhll_k!(k=muhll_k, muhll=muhll_l, 
                hlmb_edth=hlmb_edth, hlmb=hlmb_l, 
                pi_edth=pi_edth, pi_l=pi_l, 
                psi2=psi2_l,
                pi_nm=pi_nm_l, 
                hmbmb_edth_nm=hmbmb_edth_nm, hmbmb_nm=hmbmb_nm_l, 
                hlmb_edth_nm=hlmb_edth_nm,   hlmb_nm=hlmb_nm_l,
                mu_0=mu_0, tau_0=tau_0, pi_0=pi_0, 
                R=R
               )
   finish_linear_reconstruction_k!(k=muhll_k, f=muhll_l, DR=muhll_DR, R=R, falloff=muhll_f.falloff, bhm=bhm, cl=cl, dr=dr)
   
   return nothing
end

"""
One giant RK4 evolution for all of the linear variables
"""
function Linear_evolution!(;
      psi4_f_pm, psi4_f_nm,
      psi4_p_pm, psi4_p_nm,
      psi3_pm,   psi3_nm,
      psi2_pm,   psi2_nm,
      lam_pm,    lam_nm,
      pi_pm,     pi_nm,
      hmbmb_pm,  hmbmb_nm,
      hlmb_pm,   hlmb_nm,
      muhll_pm,  muhll_nm,
      Evo_pm,    Evo_nm,
      Op_pm,     Op_nm,
      NP,
      R::Vector{Float64},
      m_ang::Int64,
      bhm::Float64,
      cl::Float64,
      dr::Float64,
      dt::Float64
   )::Nothing
   psi4_f_pm_n   = psi4_f_pm.n  ; psi4_f_nm_n   = psi4_f_nm.n
   psi4_f_pm_tmp = psi4_f_pm.tmp; psi4_f_nm_tmp = psi4_f_nm.tmp
   psi4_f_pm_np1 = psi4_f_pm.np1; psi4_f_nm_np1 = psi4_f_nm.np1
   psi4_f_pm_k   = psi4_f_pm.k;   psi4_f_nm_k   = psi4_f_nm.k
   
   psi4_p_pm_n   = psi4_p_pm.n;   psi4_p_nm_n   = psi4_p_nm.n
   psi4_p_pm_tmp = psi4_p_pm.tmp; psi4_p_nm_tmp = psi4_p_nm.tmp
   psi4_p_pm_np1 = psi4_p_pm.np1; psi4_p_nm_np1 = psi4_p_nm.np1
   psi4_p_pm_k   = psi4_p_pm.k;   psi4_p_nm_k   = psi4_p_nm.k
   
   psi3_pm_n   = psi3_pm.n;   psi3_nm_n   = psi3_nm.n 
   psi3_pm_tmp = psi3_pm.tmp; psi3_nm_tmp = psi3_nm.tmp
   psi3_pm_np1 = psi3_pm.np1; psi3_nm_np1 = psi3_nm.np1
   psi3_pm_k   = psi3_pm.k;   psi3_nm_k   = psi3_nm.k

   psi2_pm_n   = psi2_pm.n;   psi2_nm_n   = psi2_nm.n
   psi2_pm_tmp = psi2_pm.tmp; psi2_nm_tmp = psi2_nm.tmp
   psi2_pm_np1 = psi2_pm.np1; psi2_nm_np1 = psi2_nm.np1
   psi2_pm_k   = psi2_pm.k;   psi2_nm_k   = psi2_nm.k

   lam_pm_n   = lam_pm.n;   lam_nm_n   = lam_nm.n
   lam_pm_tmp = lam_pm.tmp; lam_nm_tmp = lam_nm.tmp
   lam_pm_np1 = lam_pm.np1; lam_nm_np1 = lam_nm.np1
   lam_pm_k   = lam_pm.k;   lam_nm_k   = lam_nm.k

   pi_pm_n   = pi_pm.n;   pi_nm_n   = pi_nm.n
   pi_pm_tmp = pi_pm.tmp; pi_nm_tmp = pi_nm.tmp
   pi_pm_np1 = pi_pm.np1; pi_nm_np1 = pi_nm.np1
   pi_pm_k   = pi_pm.k;   pi_nm_k   = pi_nm.k

   hmbmb_pm_n   = hmbmb_pm.n;   hmbmb_nm_n   = hmbmb_nm.n 
   hmbmb_pm_tmp = hmbmb_pm.tmp; hmbmb_nm_tmp = hmbmb_nm.tmp 
   hmbmb_pm_np1 = hmbmb_pm.np1; hmbmb_nm_np1 = hmbmb_nm.np1 
   hmbmb_pm_k   = hmbmb_pm.k;   hmbmb_nm_k   = hmbmb_nm.k 

   hlmb_pm_n   = hlmb_pm.n;   hlmb_nm_n   = hlmb_nm.n 
   hlmb_pm_tmp = hlmb_pm.tmp; hlmb_nm_tmp = hlmb_nm.tmp 
   hlmb_pm_np1 = hlmb_pm.np1; hlmb_nm_np1 = hlmb_nm.np1 
   hlmb_pm_k   = hlmb_pm.k;   hlmb_nm_k   = hlmb_nm.k 

   muhll_pm_n   = muhll_pm.n;   muhll_nm_n   = muhll_nm.n 
   muhll_pm_tmp = muhll_pm.tmp; muhll_nm_tmp = muhll_nm.tmp 
   muhll_pm_np1 = muhll_pm.np1; muhll_nm_np1 = muhll_nm.np1 
   muhll_pm_k   = muhll_pm.k;   muhll_nm_k   = muhll_nm.k 

   nx, ny = size(psi4_f_pm_n)
   
   fltrM_pm = Evo_pm.S_fltr
   fltrM_nm = Evo_nm.S_fltr

   set_linear_k!(
      psi4_f=psi4_f_pm,    psi4_f_l=psi4_f_pm_n,
      psi4_p=psi4_p_pm,    psi4_p_l=psi4_p_pm_n,
      psi3_f=psi3_pm,      psi3_l=psi3_pm_n,
      psi2_f=psi2_pm,      psi2_l=psi2_pm_n,
      lam_f=lam_pm,        lam_l=lam_pm_n,
      pi_f=pi_pm,          pi_l=pi_pm_n,                            pi_nm_l=pi_nm_n,
      hmbmb_f=hmbmb_pm,    hmbmb_l=hmbmb_pm_n, hmbmb_f_nm=hmbmb_nm, hmbmb_nm_l=hmbmb_nm_n,
      hlmb_f=hlmb_pm,      hlmb_l=hlmb_pm_n,   hlmb_f_nm=hlmb_nm,   hlmb_nm_l=hlmb_nm_n,
      muhll_f=muhll_pm,    muhll_l=muhll_pm_n,
      Evo=Evo_pm, Op=Op_pm, NP=NP,
      R=R,
      m_ang=m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   set_linear_k!(
      psi4_f=psi4_f_nm,    psi4_f_l=psi4_f_nm_n,
      psi4_p=psi4_p_nm,    psi4_p_l=psi4_p_nm_n,
      psi3_f=psi3_nm,      psi3_l=psi3_nm_n,
      psi2_f=psi2_nm,      psi2_l=psi2_nm_n,
      lam_f=lam_nm,        lam_l=lam_nm_n,
      pi_f=pi_nm,          pi_l=pi_nm_n,                            pi_nm_l=pi_pm_n,
      hmbmb_f=hmbmb_nm,    hmbmb_l=hmbmb_nm_n, hmbmb_f_nm=hmbmb_pm, hmbmb_nm_l=hmbmb_pm_n,
      hlmb_f=hlmb_nm,      hlmb_l=hlmb_nm_n,   hlmb_f_nm=hlmb_pm,   hlmb_nm_l=hlmb_pm_n,
      muhll_f=muhll_nm,    muhll_l=muhll_nm_n,
      Evo=Evo_pm, Op=Op_nm, NP=NP,
      R=R,
      m_ang=-m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   for j=1:ny
      for i=1:nx
         psi4_f_pm_tmp[i,j] = psi4_f_pm_n[i,j] + half*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_tmp[i,j] = psi4_p_pm_n[i,j] + half*dt*psi4_p_pm_k[i,j]
         psi3_pm_tmp[i,j]   = psi3_pm_n[i,j]   + half*dt*psi3_pm_k[i,j] 
         psi2_pm_tmp[i,j]   = psi2_pm_n[i,j]   + half*dt*psi2_pm_k[i,j] 
         lam_pm_tmp[i,j]    = lam_pm_n[i,j]    + half*dt*lam_pm_k[i,j] 
         pi_pm_tmp[i,j]     = pi_pm_n[i,j]     + half*dt*pi_pm_k[i,j] 
         hmbmb_pm_tmp[i,j]  = hmbmb_pm_n[i,j]  + half*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_tmp[i,j]   = hlmb_pm_n[i,j]   + half*dt*hlmb_pm_k[i,j] 
         muhll_pm_tmp[i,j]  = muhll_pm_n[i,j]  + half*dt*muhll_pm_k[i,j] 
         
         psi4_f_pm_np1[i,j] = psi4_f_pm_n[i,j] + sixth*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_np1[i,j] = psi4_p_pm_n[i,j] + sixth*dt*psi4_p_pm_k[i,j]
         psi3_pm_np1[i,j]   = psi3_pm_n[i,j]   + sixth*dt*psi3_pm_k[i,j] 
         psi2_pm_np1[i,j]   = psi2_pm_n[i,j]   + sixth*dt*psi2_pm_k[i,j] 
         lam_pm_np1[i,j]    = lam_pm_n[i,j]    + sixth*dt*lam_pm_k[i,j] 
         pi_pm_np1[i,j]     = pi_pm_n[i,j]     + sixth*dt*pi_pm_k[i,j] 
         hmbmb_pm_np1[i,j]  = hmbmb_pm_n[i,j]  + sixth*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_np1[i,j]   = hlmb_pm_n[i,j]   + sixth*dt*hlmb_pm_k[i,j] 
         muhll_pm_np1[i,j]  = muhll_pm_n[i,j]  + sixth*dt*muhll_pm_k[i,j] 

         psi4_f_nm_tmp[i,j] = psi4_f_nm_n[i,j] + half*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_tmp[i,j] = psi4_p_nm_n[i,j] + half*dt*psi4_p_nm_k[i,j]
         psi3_nm_tmp[i,j]   = psi3_nm_n[i,j]   + half*dt*psi3_nm_k[i,j] 
         psi2_nm_tmp[i,j]   = psi2_nm_n[i,j]   + half*dt*psi2_nm_k[i,j] 
         lam_nm_tmp[i,j]    = lam_nm_n[i,j]    + half*dt*lam_nm_k[i,j] 
         pi_nm_tmp[i,j]     = pi_nm_n[i,j]     + half*dt*pi_nm_k[i,j] 
         hmbmb_nm_tmp[i,j]  = hmbmb_nm_n[i,j]  + half*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_tmp[i,j]   = hlmb_nm_n[i,j]   + half*dt*hlmb_nm_k[i,j] 
         muhll_nm_tmp[i,j]  = muhll_nm_n[i,j]  + half*dt*muhll_nm_k[i,j] 
         
         psi4_f_nm_np1[i,j] = psi4_f_nm_n[i,j] + sixth*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_np1[i,j] = psi4_p_nm_n[i,j] + sixth*dt*psi4_p_nm_k[i,j]
         psi3_nm_np1[i,j]   = psi3_nm_n[i,j]   + sixth*dt*psi3_nm_k[i,j] 
         psi2_nm_np1[i,j]   = psi2_nm_n[i,j]   + sixth*dt*psi2_nm_k[i,j] 
         lam_nm_np1[i,j]    = lam_nm_n[i,j]    + sixth*dt*lam_nm_k[i,j] 
         pi_nm_np1[i,j]     = pi_nm_n[i,j]     + sixth*dt*pi_nm_k[i,j] 
         hmbmb_nm_np1[i,j]  = hmbmb_nm_n[i,j]  + sixth*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_np1[i,j]   = hlmb_nm_n[i,j]   + sixth*dt*hlmb_nm_k[i,j] 
         muhll_nm_np1[i,j]  = muhll_nm_n[i,j]  + sixth*dt*muhll_nm_k[i,j] 
      end
   end
 
   set_linear_k!(
      psi4_f=psi4_f_pm,    psi4_f_l=psi4_f_pm_tmp,
      psi4_p=psi4_p_pm,    psi4_p_l=psi4_p_pm_tmp,
      psi3_f=psi3_pm,      psi3_l=psi3_pm_tmp,
      psi2_f=psi2_pm,      psi2_l=psi2_pm_tmp,
      lam_f=lam_pm,        lam_l=lam_pm_tmp,
      pi_f=pi_pm,          pi_l=pi_pm_tmp,                            pi_nm_l=pi_nm_tmp,
      hmbmb_f=hmbmb_pm,    hmbmb_l=hmbmb_pm_tmp, hmbmb_f_nm=hmbmb_nm, hmbmb_nm_l=hmbmb_nm_tmp,
      hlmb_f=hlmb_pm,      hlmb_l=hlmb_pm_tmp,   hlmb_f_nm=hlmb_nm,   hlmb_nm_l=hlmb_nm_tmp,
      muhll_f=muhll_pm,    muhll_l=muhll_pm_tmp,
      Evo=Evo_pm, Op=Op_pm, NP=NP,
      R=R,
      m_ang=m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   set_linear_k!(
      psi4_f=psi4_f_nm,    psi4_f_l=psi4_f_nm_tmp,
      psi4_p=psi4_p_nm,    psi4_p_l=psi4_p_nm_tmp,
      psi3_f=psi3_nm,      psi3_l=psi3_nm_tmp,
      psi2_f=psi2_nm,      psi2_l=psi2_nm_tmp,
      lam_f=lam_nm,        lam_l=lam_nm_tmp,
      pi_f=pi_nm,          pi_l=pi_nm_tmp,                            pi_nm_l=pi_pm_tmp,
      hmbmb_f=hmbmb_nm,    hmbmb_l=hmbmb_nm_tmp, hmbmb_f_nm=hmbmb_pm, hmbmb_nm_l=hmbmb_pm_tmp,
      hlmb_f=hlmb_nm,      hlmb_l=hlmb_nm_tmp,   hlmb_f_nm=hlmb_pm,   hlmb_nm_l=hlmb_pm_tmp,
      muhll_f=muhll_nm,    muhll_l=muhll_nm_tmp,
      Evo=Evo_pm, Op=Op_nm, NP=NP,
      R=R,
      m_ang=-m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )
   
   for j=1:ny
      for i=1:nx
         psi4_f_pm_tmp[i,j] = psi4_f_pm_n[i,j] + half*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_tmp[i,j] = psi4_p_pm_n[i,j] + half*dt*psi4_p_pm_k[i,j]
         psi3_pm_tmp[i,j]   = psi3_pm_n[i,j]   + half*dt*psi3_pm_k[i,j] 
         psi2_pm_tmp[i,j]   = psi2_pm_n[i,j]   + half*dt*psi2_pm_k[i,j] 
         lam_pm_tmp[i,j]    = lam_pm_n[i,j]    + half*dt*lam_pm_k[i,j] 
         pi_pm_tmp[i,j]     = pi_pm_n[i,j]     + half*dt*pi_pm_k[i,j] 
         hmbmb_pm_tmp[i,j]  = hmbmb_pm_n[i,j]  + half*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_tmp[i,j]   = hlmb_pm_n[i,j]   + half*dt*hlmb_pm_k[i,j] 
         muhll_pm_tmp[i,j]  = muhll_pm_n[i,j]  + half*dt*muhll_pm_k[i,j] 

         psi4_f_pm_np1[i,j] += third*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_np1[i,j] += third*dt*psi4_p_pm_k[i,j]
         psi3_pm_np1[i,j]   += third*dt*psi3_pm_k[i,j] 
         psi2_pm_np1[i,j]   += third*dt*psi2_pm_k[i,j] 
         lam_pm_np1[i,j]    += third*dt*lam_pm_k[i,j] 
         pi_pm_np1[i,j]     += third*dt*pi_pm_k[i,j] 
         hmbmb_pm_np1[i,j]  += third*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_np1[i,j]   += third*dt*hlmb_pm_k[i,j] 
         muhll_pm_np1[i,j]  += third*dt*muhll_pm_k[i,j] 
         
         psi4_f_nm_tmp[i,j] = psi4_f_nm_n[i,j] + half*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_tmp[i,j] = psi4_p_nm_n[i,j] + half*dt*psi4_p_nm_k[i,j]
         psi3_nm_tmp[i,j]   = psi3_nm_n[i,j]   + half*dt*psi3_nm_k[i,j] 
         psi2_nm_tmp[i,j]   = psi2_nm_n[i,j]   + half*dt*psi2_nm_k[i,j] 
         lam_nm_tmp[i,j]    = lam_nm_n[i,j]    + half*dt*lam_nm_k[i,j] 
         pi_nm_tmp[i,j]     = pi_nm_n[i,j]     + half*dt*pi_nm_k[i,j] 
         hmbmb_nm_tmp[i,j]  = hmbmb_nm_n[i,j]  + half*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_tmp[i,j]   = hlmb_nm_n[i,j]   + half*dt*hlmb_nm_k[i,j] 
         muhll_nm_tmp[i,j]  = muhll_nm_n[i,j]  + half*dt*muhll_nm_k[i,j] 

         psi4_f_nm_np1[i,j] += third*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_np1[i,j] += third*dt*psi4_p_nm_k[i,j]
         psi3_nm_np1[i,j]   += third*dt*psi3_nm_k[i,j] 
         psi2_nm_np1[i,j]   += third*dt*psi2_nm_k[i,j] 
         lam_nm_np1[i,j]    += third*dt*lam_nm_k[i,j] 
         pi_nm_np1[i,j]     += third*dt*pi_nm_k[i,j] 
         hmbmb_nm_np1[i,j]  += third*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_np1[i,j]   += third*dt*hlmb_nm_k[i,j] 
         muhll_nm_np1[i,j]  += third*dt*muhll_nm_k[i,j] 
      end
   end
  
   set_linear_k!(
      psi4_f=psi4_f_pm,    psi4_f_l=psi4_f_pm_tmp,
      psi4_p=psi4_p_pm,    psi4_p_l=psi4_p_pm_tmp,
      psi3_f=psi3_pm,      psi3_l=psi3_pm_tmp,
      psi2_f=psi2_pm,      psi2_l=psi2_pm_tmp,
      lam_f=lam_pm,        lam_l=lam_pm_tmp,
      pi_f=pi_pm,          pi_l=pi_pm_tmp,                            pi_nm_l=pi_nm_tmp,
      hmbmb_f=hmbmb_pm,    hmbmb_l=hmbmb_pm_tmp, hmbmb_f_nm=hmbmb_nm, hmbmb_nm_l=hmbmb_nm_tmp,
      hlmb_f=hlmb_pm,      hlmb_l=hlmb_pm_tmp,   hlmb_f_nm=hlmb_nm,   hlmb_nm_l=hlmb_nm_tmp,
      muhll_f=muhll_pm,    muhll_l=muhll_pm_tmp,
      Evo=Evo_pm, Op=Op_pm, NP=NP,
      R=R,
      m_ang=m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   set_linear_k!(
      psi4_f=psi4_f_nm,    psi4_f_l=psi4_f_nm_tmp,
      psi4_p=psi4_p_nm,    psi4_p_l=psi4_p_nm_tmp,
      psi3_f=psi3_nm,      psi3_l=psi3_nm_tmp,
      psi2_f=psi2_nm,      psi2_l=psi2_nm_tmp,
      lam_f=lam_nm,        lam_l=lam_nm_tmp,
      pi_f=pi_nm,          pi_l=pi_nm_tmp,                            pi_nm_l=pi_pm_tmp,
      hmbmb_f=hmbmb_nm,    hmbmb_l=hmbmb_nm_tmp, hmbmb_f_nm=hmbmb_pm, hmbmb_nm_l=hmbmb_pm_tmp,
      hlmb_f=hlmb_nm,      hlmb_l=hlmb_nm_tmp,   hlmb_f_nm=hlmb_pm,   hlmb_nm_l=hlmb_pm_tmp,
      muhll_f=muhll_nm,    muhll_l=muhll_nm_tmp,
      Evo=Evo_pm, Op=Op_nm, NP=NP,
      R=R,
      m_ang=-m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   for j=1:ny
      for i=1:nx
         psi4_f_pm_tmp[i,j] = psi4_f_pm_n[i,j] + dt*psi4_f_pm_k[i,j]
         psi4_p_pm_tmp[i,j] = psi4_p_pm_n[i,j] + dt*psi4_p_pm_k[i,j]
         psi3_pm_tmp[i,j]   = psi3_pm_n[i,j]   + dt*psi3_pm_k[i,j] 
         psi2_pm_tmp[i,j]   = psi2_pm_n[i,j]   + dt*psi2_pm_k[i,j] 
         lam_pm_tmp[i,j]    = lam_pm_n[i,j]    + dt*lam_pm_k[i,j] 
         pi_pm_tmp[i,j]     = pi_pm_n[i,j]     + dt*pi_pm_k[i,j] 
         hmbmb_pm_tmp[i,j]  = hmbmb_pm_n[i,j]  + dt*hmbmb_pm_k[i,j] 
         hlmb_pm_tmp[i,j]   = hlmb_pm_n[i,j]   + dt*hlmb_pm_k[i,j] 
         muhll_pm_tmp[i,j]  = muhll_pm_n[i,j]  + dt*muhll_pm_k[i,j] 

         psi4_f_pm_np1[i,j] += third*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_np1[i,j] += third*dt*psi4_p_pm_k[i,j]
         psi3_pm_np1[i,j]   += third*dt*psi3_pm_k[i,j] 
         psi2_pm_np1[i,j]   += third*dt*psi2_pm_k[i,j] 
         lam_pm_np1[i,j]    += third*dt*lam_pm_k[i,j] 
         pi_pm_np1[i,j]     += third*dt*pi_pm_k[i,j] 
         hmbmb_pm_np1[i,j]  += third*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_np1[i,j]   += third*dt*hlmb_pm_k[i,j] 
         muhll_nm_np1[i,j]  += third*dt*muhll_nm_k[i,j] 

         psi4_f_nm_tmp[i,j] = psi4_f_nm_n[i,j] + dt*psi4_f_nm_k[i,j]
         psi4_p_nm_tmp[i,j] = psi4_p_nm_n[i,j] + dt*psi4_p_nm_k[i,j]
         psi3_nm_tmp[i,j]   = psi3_nm_n[i,j]   + dt*psi3_nm_k[i,j] 
         psi2_nm_tmp[i,j]   = psi2_nm_n[i,j]   + dt*psi2_nm_k[i,j] 
         lam_nm_tmp[i,j]    = lam_nm_n[i,j]    + dt*lam_nm_k[i,j] 
         pi_nm_tmp[i,j]     = pi_nm_n[i,j]     + dt*pi_nm_k[i,j] 
         hmbmb_nm_tmp[i,j]  = hmbmb_nm_n[i,j]  + dt*hmbmb_nm_k[i,j] 
         hlmb_nm_tmp[i,j]   = hlmb_nm_n[i,j]   + dt*hlmb_nm_k[i,j] 
         muhll_nm_tmp[i,j]  = muhll_nm_n[i,j]  + dt*muhll_nm_k[i,j] 

         psi4_f_nm_np1[i,j] += third*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_np1[i,j] += third*dt*psi4_p_nm_k[i,j]
         psi3_nm_np1[i,j]   += third*dt*psi3_nm_k[i,j] 
         psi2_nm_np1[i,j]   += third*dt*psi2_nm_k[i,j] 
         lam_nm_np1[i,j]    += third*dt*lam_nm_k[i,j] 
         pi_nm_np1[i,j]     += third*dt*pi_nm_k[i,j] 
         hmbmb_nm_np1[i,j]  += third*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_np1[i,j]   += third*dt*hlmb_nm_k[i,j] 
         muhll_nm_np1[i,j]  += third*dt*muhll_nm_k[i,j] 
      end
   end
   
   set_linear_k!(
      psi4_f=psi4_f_pm,    psi4_f_l=psi4_f_pm_tmp,
      psi4_p=psi4_p_pm,    psi4_p_l=psi4_p_pm_tmp,
      psi3_f=psi3_pm,      psi3_l=psi3_pm_tmp,
      psi2_f=psi2_pm,      psi2_l=psi2_pm_tmp,
      lam_f=lam_pm,        lam_l=lam_pm_tmp,
      pi_f=pi_pm,          pi_l=pi_pm_tmp,                            pi_nm_l=pi_nm_tmp,
      hmbmb_f=hmbmb_pm,    hmbmb_l=hmbmb_pm_tmp, hmbmb_f_nm=hmbmb_nm, hmbmb_nm_l=hmbmb_nm_tmp,
      hlmb_f=hlmb_pm,      hlmb_l=hlmb_pm_tmp,   hlmb_f_nm=hlmb_nm,   hlmb_nm_l=hlmb_nm_tmp,
      muhll_f=muhll_pm,    muhll_l=muhll_pm_tmp,
      Evo=Evo_pm, Op=Op_pm, NP=NP,
      R=R,
      m_ang=m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   set_linear_k!(
      psi4_f=psi4_f_nm,    psi4_f_l=psi4_f_nm_tmp,
      psi4_p=psi4_p_nm,    psi4_p_l=psi4_p_nm_tmp,
      psi3_f=psi3_nm,      psi3_l=psi3_nm_tmp,
      psi2_f=psi2_nm,      psi2_l=psi2_nm_tmp,
      lam_f=lam_nm,        lam_l=lam_nm_tmp,
      pi_f=pi_nm,          pi_l=pi_nm_tmp,                            pi_nm_l=pi_pm_tmp,
      hmbmb_f=hmbmb_nm,    hmbmb_l=hmbmb_nm_tmp, hmbmb_f_nm=hmbmb_pm, hmbmb_nm_l=hmbmb_pm_tmp,
      hlmb_f=hlmb_nm,      hlmb_l=hlmb_nm_tmp,   hlmb_f_nm=hlmb_pm,   hlmb_nm_l=hlmb_pm_tmp,
      muhll_f=muhll_nm,    muhll_l=muhll_nm_tmp,
      Evo=Evo_pm, Op=Op_nm, NP=NP,
      R=R,
      m_ang=-m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   for j=1:ny
      for i=1:nx
         psi4_f_pm_np1[i,j] += sixth*dt*psi4_f_pm_k[i,j]
         psi4_p_pm_np1[i,j] += sixth*dt*psi4_p_pm_k[i,j]   
         psi3_pm_np1[i,j]   += third*dt*psi3_pm_k[i,j] 
         psi2_pm_np1[i,j]   += third*dt*psi2_pm_k[i,j] 
         lam_pm_np1[i,j]    += third*dt*lam_pm_k[i,j] 
         pi_pm_np1[i,j]     += third*dt*pi_pm_k[i,j] 
         hmbmb_pm_np1[i,j]  += third*dt*hmbmb_pm_k[i,j] 
         hlmb_pm_np1[i,j]   += third*dt*hlmb_pm_k[i,j] 
         muhll_pm_np1[i,j]  += third*dt*muhll_pm_k[i,j] 

         psi4_f_nm_np1[i,j] += sixth*dt*psi4_f_nm_k[i,j]
         psi4_p_nm_np1[i,j] += sixth*dt*psi4_p_nm_k[i,j]   
         psi3_nm_np1[i,j]   += third*dt*psi3_nm_k[i,j] 
         psi2_nm_np1[i,j]   += third*dt*psi2_nm_k[i,j] 
         lam_nm_np1[i,j]    += third*dt*lam_nm_k[i,j] 
         pi_nm_np1[i,j]     += third*dt*pi_nm_k[i,j] 
         hmbmb_nm_np1[i,j]  += third*dt*hmbmb_nm_k[i,j] 
         hlmb_nm_np1[i,j]   += third*dt*hlmb_nm_k[i,j] 
         muhll_nm_np1[i,j]  += third*dt*muhll_nm_k[i,j] 
      end
   end
  
   ## set k for level n+1 (for evaluating the independent residuals)

   set_linear_k!(
      psi4_f=psi4_f_pm,    psi4_f_l=psi4_f_pm_np1,
      psi4_p=psi4_p_pm,    psi4_p_l=psi4_p_pm_np1,
      psi3_f=psi3_pm,      psi3_l=psi3_pm_np1,
      psi2_f=psi2_pm,      psi2_l=psi2_pm_np1,
      lam_f=lam_pm,        lam_l=lam_pm_np1,
      pi_f=pi_pm,          pi_l=pi_pm_np1,                            pi_nm_l=pi_nm_np1,
      hmbmb_f=hmbmb_pm,    hmbmb_l=hmbmb_pm_np1, hmbmb_f_nm=hmbmb_nm, hmbmb_nm_l=hmbmb_nm_np1,
      hlmb_f=hlmb_pm,      hlmb_l=hlmb_pm_np1,   hlmb_f_nm=hlmb_nm,   hlmb_nm_l=hlmb_nm_np1,
      muhll_f=muhll_pm,    muhll_l=muhll_pm_np1,
      Evo=Evo_pm, Op=Op_pm, NP=NP,
      R=R,
      m_ang=m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )

   set_linear_k!(
      psi4_f=psi4_f_nm,    psi4_f_l=psi4_f_nm_np1,
      psi4_p=psi4_p_nm,    psi4_p_l=psi4_p_nm_np1,
      psi3_f=psi3_nm,      psi3_l=psi3_nm_np1,
      psi2_f=psi2_nm,      psi2_l=psi2_nm_np1,
      lam_f=lam_nm,        lam_l=lam_nm_np1,
      pi_f=pi_nm,          pi_l=pi_nm_np1,                            pi_nm_l=pi_pm_np1,
      hmbmb_f=hmbmb_nm,    hmbmb_l=hmbmb_nm_np1, hmbmb_f_nm=hmbmb_pm, hmbmb_nm_l=hmbmb_pm_np1,
      hlmb_f=hlmb_nm,      hlmb_l=hlmb_nm_np1,   hlmb_f_nm=hlmb_pm,   hlmb_nm_l=hlmb_pm_np1,
      muhll_f=muhll_nm,    muhll_l=muhll_nm_np1,
      Evo=Evo_pm, Op=Op_nm, NP=NP,
      R=R,
      m_ang=-m_ang, 
      bhm=bhm, cl=cl, dr=dr
     )
    
   ## First filter psi4 

   filter!(psi4_f_pm_np1,psi4_f_pm_tmp,0.5)
   filter!(psi4_p_pm_np1,psi4_p_pm_tmp,0.5) 
   
   filter!(psi4_f_nm_np1,psi4_f_nm_tmp,0.5)
   filter!(psi4_p_nm_np1,psi4_p_nm_tmp,0.5) 
   
   for j=1:ny
      for i=1:nx
         psi4_f_pm_tmp[i,j] = psi4_f_pm_np1[i,j] 
         psi4_p_pm_tmp[i,j] = psi4_p_pm_np1[i,j] 
         
         psi4_f_nm_tmp[i,j] = psi4_f_nm_np1[i,j] 
         psi4_p_nm_tmp[i,j] = psi4_p_nm_np1[i,j] 
      end
   end
   angular_matrix_mult!(psi4_f_pm_np1,psi4_f_pm_tmp,fltrM_pm)
   angular_matrix_mult!(psi4_p_pm_np1,psi4_p_pm_tmp,fltrM_pm)

   angular_matrix_mult!(psi4_f_nm_np1,psi4_f_nm_tmp,fltrM_nm)
   angular_matrix_mult!(psi4_p_nm_np1,psi4_p_nm_tmp,fltrM_nm)
   
   ## Filter metric reconstructed variables for stability 

   filter!(psi3_pm_np1,psi3_pm_tmp,0.5)
   filter!(psi3_nm_np1,psi3_nm_tmp,0.5)
   return nothing
end
   
function set_res_bianchi3!(;
      res_bianchi3::Array{ComplexF64,2}, 
      psi4::Array{ComplexF64,2},
      psi3::Array{ComplexF64,2},
      lam ::Array{ComplexF64,2},
      psi4_thorn::Array{ComplexF64,2},
      psi3_edth_prime::Array{ComplexF64,2},
      pi_0  ::Array{ComplexF64,2},
      rho_0 ::Array{ComplexF64,2},
      psi2_0::Array{ComplexF64,2}
     )::Nothing
   nx, ny = size(res_bianchi3)
   for j=1:ny
      for i=1:nx
         res_bianchi3[i,j] = (
                          psi3_edth_prime[i,j]
                          +
                          4.0*pi_0[i,j]*psi3[i,j]
                          -
                          psi4_thorn[i,j]
                          +
                          rho_0[i,j]*psi4[i,j]
                          -
                          3.0*psi2_0[i,j]*lam[i,j]
                         )
      end
   end
   return nothing
end

function set_res_bianchi2!(;
      res_bianchi2::Array{ComplexF64,2}, 
      psi3::Array{ComplexF64,2},
      psi2::Array{ComplexF64,2},
      pi_f::Array{ComplexF64,2},
      hlmb ::Array{ComplexF64,2},
      hmbmb::Array{ComplexF64,2},
      psi2_edth_prime::Array{ComplexF64,2},
      psi3_thorn::Array{ComplexF64,2},
      mu_0  ::Array{Float64,2},
      tau_0 ::Array{Float64,2},
      pi_0  ::Array{Float64,2},
      rho_0 ::Array{Float64,2},
      psi2_0::Array{Float64,2}
     )::Nothing
   nx, ny = size(res_bianchi2)
   for j=1:ny
      for i=1:nx
         res_bianchi2[i,j] = (-
                          3.0*psi2_0[i,j]*mu_0[i,j]*hlmb[i,j]
                          -
                          1.5*psi2_0[i,j]*tau_0[i,j]*hmbmb[i,j]
                          -
                          3.0*psi2_0[i,j]*pi_f[i,j]
                          -
                          psi2_edth_prime[i,j]
                          -
                          3.0*pi_0[i,j]*psi2[i,j]
                          +
                          psi3_thorn[i,j]
                          -
                          2.0*rho_0[i,j]*psi3[i,j]
                         )
      end
   end
   return nothing
end


"""
Computation of independent residuals;
See Sec. V.H of arXiv:2010.00162
"""
function Set_independent_residuals!(;
      res_bianchi3_f,
      res_bianchi2_f,
      res_hll_f,
      psi4_f,
      psi3_f,
      psi2_f,
      lam_f,
      pi_f,
      hmbmb_f,
      hlmb_f,
      muhll_f,
      Op,
      NP,
      R::Vector{Float64},
      m_ang::Int64
   )::Nothing
 
   res_bianchi3 = res_bianchi3_f.np1
   res_bianchi2 = res_bianchi2_f.np1
   res_hll      = res_hll_f.np1
   
   psi4       = psi4_f.np1
   psi4_k     = psi4_f.k
   psi4_thorn = psi4_f.thorn
   psi4_DR    = psi4_f.rad_d1
   
   psi3            = psi3_f.np1
   psi3_k          = psi3_f.k
   psi3_lowered    = psi3_f.lowered 
   psi3_edth_prime = psi3_f.edth_prime
   
   psi2   = psi2_f.np1
   psi2_k = psi2_f.k
   
   lam = lam_f.np1
   Pi  = pi_f.np1
   
   hmbmb = hmbmb_f.np1
   hlmb  = hlmb_f.np1
   muhll = muhll_f.np1
  
   mu_0   = NP.mu_0
   tau_0  = NP.tau_0
   pi_0   = NP.pi_0
   rho_0  = NP.rho_0
   eps_0  = NP.eps_0
   psi2_0 = NP.psi2_0
  
   ##
   ## Bianchi 2 residual
   ##
   
   GHP.set_edth_prime!(
      edth_prime=psi3_edth_prime,
      spin=psi3_f.spin, 
      boost=psi3_f.boost, 
      m_ang=m_ang, 
      f=psi3,
      DT=psi3_k, 
      lowered=psi3_lowered,
      Op=Op
     )
   
   GHP.set_thorn!(;
      thorn=psi4_thorn,
      spin=psi4_f.spin, 
      boost=psi4_f.boost, 
      m_ang=m_ang,
      falloff=psi4_f.falloff, 
      f=psi4, 
      DT=psi4_k, 
      DR=psi4_DR,
      eps_0=eps_0,
      R=R,
      Op=Op
     )
  
   nx, ny = size(res_bianchi3)

   set_res_bianchi3!(
         res_bianchi3=res_bianchi3, 
         psi4=psi4,
         psi3=psi3,
         lam=lam,
         psi4_thorn=psi4_thorn,
         psi3_edth_prime=psi3_edth_prime,
         pi_0=pi_0,
         rho_0=rho_0,
         psi2_0=psi2_0
        )

   return nothing
end

end