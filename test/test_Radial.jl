push!(LOAD_PATH,".")
push!(LOAD_PATH,"../Source/")

import Test: @test
import Norms 
import Radial

println("Tests for radial derivatives")

"""
Computes the norm of the difference between the numerical and
exact derivative.

norms_diff(
   nx::Int64,
   ny::Int64
   )::(Float64,Float64)
"""
function norms_diff(nx::Int64,ny::Int64)::Tuple{Float64,Float64}
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

   n1 = Norms.one_norm(v1_rd1 .- v2_rd1)
   n2 = Norms.one_norm(v1_rd2 .- v2_rd2)

   return (n1, n2) 
end

n11, n12 = norms_diff(128,48)
n21, n22 = norms_diff(256,60)

println("Testing second order convergence of radial finite differences...")
@test n11/n21 >= 3.5 && n11/n21 <= 4.5
@test n12/n22 >= 3.5 && n12/n22 <= 4.5
println("Passed test.")

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

   n1 = Norms.one_norm(v)
   Radial.filter!(v,tmp,1.0)
   n2 = Norms.one_norm(v)

   return (n1, n2) 
end
n1, n2 = norms_filter(128,48)
n3, n4 = norms_filter(180,90)

println("Testing filter is TVD")
@test n2 < n1 
@test n4 < n3 
println("Passed test.")
