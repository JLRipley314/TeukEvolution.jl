module Evolution

include("Fields.jl")
include("Radial.jl")
include("Sphere.jl")

using  .Fields: Field
import .Radial
import .Sphere

const half  = 1.0/2.0
const third = 1.0/3.0
const sixth = 1.0/6.0

export Evo_psi4, evolve_psi4


"""
   Struct for the evolution matrices for psi4
"""
struct Evo_psi4 
   A_pp::Array{Float64,3}
   A_pq::Array{Float64,3}
   B_pp::Array{ComplexF64,3}
   B_pq::Array{ComplexF64,3}
   B_pf::Array{ComplexF64,3}

   S_lapl::Array{Float64,3}
   S_fltr::Array{Float64,3}

   function Evo_psi4(
         Rvals::Vector{Float64},
         Cvals::Vector{Float64},
         Svals::Vector{Float64},
         Mvals::Vector{Int64},
         bhm::Float64,
         bhs::Float64,
         cl::Float64,
         spin::Int64
      )
      nx = length(Rvals)
      ny = length(Cvals)
      nm = length(Mvals)

      A_pp = zeros(Float64,nx,ny,nm)
      A_pq = zeros(Float64,nx,ny,nm)

      B_pp = zeros(ComplexF64,nx,ny,nm)
      B_pq = zeros(ComplexF64,nx,ny,nm)
      B_pf = zeros(ComplexF64,nx,ny,nm)

      S_lapl = zeros(Float64,ny,ny,nm)
      S_fltr = zeros(Float64,ny,ny,nm)

      pre = zeros(Float64,nx,ny)

      for j=1:ny
         sy = Svals[j]
         for i=1:nx
            R = Rvals[i]

            pre[i,j] = 1.0 / (
               8*bhm*(2*(cl^2)*bhm - (bhs^2)*R)*((cl^2) + 2*bhm*R) / (cl^4)
            -  (bhs*sy)^2
            )
         end
      end

      for k=1:nm
         m_ang = Mvals[k]
         
         S_lapl_tmp = Sphere.swal_laplacian_matrix(ny,spin,m_ang)
         S_fltr_tmp = Sphere.swal_filter_matrix(   ny,spin,m_ang)

         for i=1:ny
            for j=1:ny
               S_lapl[i,j,k] = pre[i,j]*S_lapl_tmp[i,j]
               S_fltr[i,j,k] =          S_fltr_tmp[i,j]
            end
         end

         for j=1:ny
            cy = Cvals[j]
            for i=1:nx
               R = Rvals[i]

               A_pp[i,j,k] = (2 / (cl^4)) * ( 
                  (cl^6) 
               +  (cl^2)*(bhs^2 - 8*(bhm^2))*(R^2) 
               +  4*(bhs^2)*bhm*(R^3) 
               )

               A_pq[i,j,k] = ((R^2) / (cl^4)) * ( 
                  (cl^4) 
               -  2*(cl^2)*bhm*R 
               +  (bhs^2)*(R^2) 
               )
               
               B_pp[i,j,k] = - ( 
                  2*im*bhs*m_ang*((cl^2) + 4*bhm*R) / (cl^2) 
               -  2*(bhs^2)*(cl^2 + 6*bhm*R)*R / (cl^4)   
               -  4*bhm*spin 
               +  8*(bhm^2)*(2 + spin)*R / (cl^2) 
               +  2*im*bhs*spin*cy 
               )

               B_pq[i,j,k] = (2 * R / (cl^4)) * ( 
                  2*(bhs^2)*(R^2) 
               +  (1 + spin)*(cl^4)  
               -  (im*bhs*m_ang + (3 + spin)*bhm) * ((cl^2) * R) 
               )

               B_pf[i,j,k] = - (2 * R / (cl^4)) * ( 
                  im*bhs*(cl^2)*m_ang 
               -  (bhs^2)*R 
               +  (cl^2)*bhm*(1 + spin) 
               )

               A_pp[i,j,k] *= pre[i,j] 
               A_pq[i,j,k] *= pre[i,j] 
               
               B_pp[i,j,k] *= pre[i,j] 
               B_pq[i,j,k] *= pre[i,j] 
               B_pf[i,j,k] *= pre[i,j] 
            end
         end
      end

      return new(A_pp,A_pq,B_pp,B_pq,B_pf,S_lapl,S_fltr) 
   end
end
"""
set_kp
"""
function set_kp(
      kp, 
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
"""
Fourth order Runge-Kutta evolution of psi4
"""
function evolve_psi4(
      psi4_f::Field, 
      psi4_p::Field, 
      Evo::Evo_psi4,
      mi::Int64,
      dr::Float64,
      dt::Float64
   )
   f_n       = @view psi4_f.n[      :,:,mi]
   f_tmp     = @view psi4_f.tmp[    :,:,mi]
   f_np1     = @view psi4_f.np1[    :,:,mi]
   f_k       = @view psi4_f.k[      :,:,mi]
   f_rd1     = @view psi4_f.rad_d1[ :,:,mi]
   f_rd2     = @view psi4_f.rad_d2[ :,:,mi]
   f_sph_lap = @view psi4_f.sph_lap[:,:,mi]

   p_n       = @view psi4_p.n[      :,:,mi]
   p_tmp     = @view psi4_p.tmp[    :,:,mi]
   p_np1     = @view psi4_p.np1[    :,:,mi]
   p_k       = @view psi4_p.k[      :,:,mi]
   p_rd1     = @view psi4_p.rad_d1[ :,:,mi]

   A_pp = @view Evo.A_pp[:,:,mi]
   A_pq = @view Evo.A_pq[:,:,mi]
   B_pp = @view Evo.B_pp[:,:,mi]
   B_pq = @view Evo.B_pq[:,:,mi]
   B_pf = @view Evo.B_pf[:,:,mi]

   laplM = @view Evo.S_lapl[:,:,mi]
   fltrM = @view Evo.S_fltr[:,:,mi]

   nx, ny = size(f_n)

   ## step 1
   Radial.set_d1!(f_rd1, f_n, dr)
   Radial.set_d1!(p_rd1, p_n, dr)
   Radial.set_d2!(f_rd2, f_n, dr)

   Sphere.angular_matrix_mult!(f_sph_lap,f_n,laplM)

   set_kp(p_k, 
      f_rd1, f_rd2, f_sph_lap, p_rd1, f_n, p_n,
      A_pp, A_pq,
      B_pp, B_pq, B_pf
   )
   for j=1:ny
      for i=1:nx
         f_k[i,j] = p_n[i,j]

         f_tmp[i,j] = f_n[i,j] + half*dt*f_k[i,j]
         p_tmp[i,j] = p_n[i,j] + half*dt*p_k[i,j]

         f_np1[i,j] = f_n[i,j] + sixth*dt*f_k[i,j]
         p_np1[i,j] = p_n[i,j] + sixth*dt*p_k[i,j]
      end
   end
   ## step 2
   Radial.set_d1!(f_rd1, f_tmp, dr)
   Radial.set_d1!(p_rd1, p_tmp, dr)
   Radial.set_d2!(f_rd2, f_tmp, dr)

   Sphere.angular_matrix_mult!(f_sph_lap,f_tmp,laplM)

   set_kp(p_k, 
      f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp,
      A_pp, A_pq,
      B_pp, B_pq, B_pf
   )
   for j=1:ny
      for i=1:nx
         f_k[i,j] = p_tmp[i,j]

         f_tmp[i,j] = f_n[i,j] + half*dt*f_k[i,j]
         p_tmp[i,j] = p_n[i,j] + half*dt*p_k[i,j]

         f_np1[i,j] += third*dt*f_k[i,j]
         p_np1[i,j] += third*dt*p_k[i,j]
      end
   end
   ## step 3
   Radial.set_d1!(f_rd1, f_tmp, dr)
   Radial.set_d1!(p_rd1, p_tmp, dr)
   Radial.set_d2!(f_rd2, f_tmp, dr)

   Sphere.angular_matrix_mult!(f_sph_lap,f_tmp,laplM)

   set_kp(p_k, 
      f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp,
      A_pp, A_pq,
      B_pp, B_pq, B_pf
   )
   for j=1:ny
      for i=1:nx
         f_k[i,j] = p_tmp[i,j]
         
         f_tmp[i,j] = f_n[i,j] + dt*f_k[i,j]
         p_tmp[i,j] = p_n[i,j] + dt*p_k[i,j]

         f_np1[i,j] += third*dt*f_k[i,j]
         p_np1[i,j] += third*dt*p_k[i,j]
      end
   end
   ## step 4
   Radial.set_d1!(f_rd1, f_tmp, dr)
   Radial.set_d1!(p_rd1, p_tmp, dr)
   Radial.set_d2!(f_rd2, f_tmp, dr)

   Sphere.angular_matrix_mult!(f_sph_lap,f_tmp,laplM)

   set_kp(p_k, 
      f_rd1, f_rd2, f_sph_lap, p_rd1, f_tmp, p_tmp,
      A_pp, A_pq,
      B_pp, B_pq, B_pf
   )
   for j=1:ny
      for i=1:nx
         f_k[i,j] = p_tmp[i,j]
         
         f_np1[i,j] += sixth*dt*f_k[i,j]
         p_np1[i,j] += sixth*dt*p_k[i,j]   
      end
   end
   Radial.filter!(f_np1,f_tmp,0.5)
   Radial.filter!(p_np1,p_tmp,0.5) 
   for j=1:ny
      for i=1:nx
         f_tmp[i,j] = f_np1[i,j] 
         p_tmp[i,j] = p_np1[i,j] 
      end
   end
   Sphere.angular_matrix_mult!(f_np1,f_tmp,fltrM)
   Sphere.angular_matrix_mult!(p_np1,p_tmp,fltrM)

   return nothing
end

end
