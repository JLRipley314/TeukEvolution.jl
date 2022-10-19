module TestRadial

include("../src/Radial.jl")
import .Radial

using LinearAlgebra: norm

export norms_diff, norms_filter

"""
Computes the norm of the difference between the numerical and
exact derivative.

norms_diff(
   nx::Integer
   ny::Integer
   )::(Real,Real)
"""
function norms_diff(nx::Integer,ny::Integer)::Tuple{Real,Real}
   dr = 1.0/(nx-1.0)
   dy = 1.0/(ny-1.0)

   Rv = Radial.R_vals(nx,dr)
   yv = [j*ny for j=1:ny] 

   v1     = zeros(nx,ny)
   v1_rd1 = zeros(nx,ny)
   v1_rd2 = zeros(nx,ny)
   v2_rd1 = zeros(nx,ny)
   v2_rd2 = zeros(nx,ny)

   for j=1:ny
      for i=1:nx
         v1[i,j]     = Rv[i]*cos(Rv[i])
         v1_rd1[i,j] = cos(Rv[i]) - Rv[i]*sin(Rv[i]) 
         v1_rd2[i,j] = -2.0*sin(Rv[i]) - Rv[i]*cos(Rv[i])
      end
      v1[:,j]     .*= sin(yv[j])
      v1_rd1[:,j] .*= sin(yv[j])
      v1_rd2[:,j] .*= sin(yv[j])
   end

   Radial.set_d1!(v2_rd1,v1,dr)
   Radial.set_d2!(v2_rd2,v1,dr)

   n1 = norm(v1_rd1 .- v2_rd1,1)/length(v1_rd1)
   n2 = norm(v1_rd2 .- v2_rd2,1)/length(v1_rd2)

   return (n1, n2) 
end

"""
Computes one norm of random field 
v(nx,ny) before and after applying Kreiss-Oliger filter.

filter_norm(
   nx::Int64,
   ny::Int64
   )::(Float64,Float64)
"""
function norms_filter(nx::Int64,ny::Int64)::Tuple{Float64,Float64}
   v = 0.1.*(rand(nx,ny) .- 0.5)

   tmp = deepcopy(v)

   n1 = norm(v,1)/length(v) 
   Radial.filter!(v,tmp,1.0)
   n2 = norm(v,1)/length(v)
   
   return (n1, n2) 
end

end
