module Id

include("Sphere.jl")

import .Sphere: swal 

export set_psi

"""
Initial data for psi.

function set_psi(
      f,
      p,
      spin::Int64,
      mi::Int64,
      mv::Int64,
      l_ang::Int64,
      ru::Float64, 
      rl::Float64, 
      width::Float64,
      amp::ComplexF64,
      cl::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )
"""
function set_psi(
      f,
      p,
      spin::Int64,
      mi::Int64,
      mv::Int64,
      l_ang::Int64,
      ru::Float64, 
      rl::Float64, 
      width::Float64,
      amp::ComplexF64,
      cl::Float64,
      Rv::Vector{Float64},
      Yv::Vector{Float64}
   )

   nx, ny = f.nx, f.ny

   max_val = 0.0

   for j=1:ny
      for i=1:nx
         r = (cl^2)/Rv[i] 

         bump = 0.0
         if ((r<ru) && (r>rl))
            bump = exp(-1.0*width/(r-rl))*exp(-2.0*width/(ru-r))
         end

         f.n[i,j,mi]  = (((r-rl)/width)^2) * (((ru-r)/width)^2) * bump
         f.n[i,j,mi] *= swal(spin,mv,l_ang,Yv[j])

         p.n[i,j,mi] = 0.0

         max_val = max(abs(f.n[i,j,mi]),max_val)
      end
   end

   ## rescale
  
   for j=1:ny
      for i=1:nx
         f.n[i,j,mi] *= amp / max_val 
         
         f.np1[i,j,mi] = f.n[i,j,mi] 
         p.np1[i,j,mi] = p.n[i,j,mi] 
      end
   end
   return nothing
end

end
