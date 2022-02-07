push!(LOAD_PATH,".")
push!(LOAD_PATH,"../Source/")

import Test: @test
import Norms 
import Sphere

println("Tests for spin-weighted spherical harmonics")

const tol = 5e-13
println("Error tolerance: $tol")
"""
Computes inner product between different Y^s_{lm} functions

test_swal_inner_product(
   ny::Int64,
   spin::Int64,
   m_ang::Int64
   )::Nothing
"""
function test_swal_inner_product(
      ny::Int64,
      spin::Int64,
      m_ang::Int64
   )::Nothing

   nl = Sphere.num_l(ny)

   vals = Sphere.swal_vals(ny,spin,m_ang)

   println("Test: Y^s_{lm} orthogonality:\tny=$ny\tspin=$spin\tm_ang=$m_ang\tl=[0..$nl]")
   for i=1:nl
      for j=i:nl
         val = Sphere.inner_product(vals[:,i],vals[:,j])
         if i==j && i>max(abs(spin),abs(m_ang))
            @test abs(val-1.0) < tol 
         else
            @test abs(val) < tol 
         end
      end
   end
   println("Passed test")
   return nothing
end

test_swal_inner_product(48,-2,-3)
test_swal_inner_product(48,-1,-3)
test_swal_inner_product(48, 0,-3)
test_swal_inner_product(48, 1,-3)
test_swal_inner_product(48, 2,-3)

test_swal_inner_product(48,-2,-2)
test_swal_inner_product(48,-1,-2)
test_swal_inner_product(48, 0,-2)
test_swal_inner_product(48, 1,-2)
test_swal_inner_product(48, 2,-2)

test_swal_inner_product(48,-2,-1)
test_swal_inner_product(48,-1,-1)
test_swal_inner_product(48, 0,-1)
test_swal_inner_product(48, 1,-1)
test_swal_inner_product(48, 2,-1)

test_swal_inner_product(48,-2,0)
test_swal_inner_product(48,-1,0)
test_swal_inner_product(48, 0,0)
test_swal_inner_product(48, 1,0)
test_swal_inner_product(48, 2,0)

test_swal_inner_product(48,-2,1)
test_swal_inner_product(48,-1,1)
test_swal_inner_product(48, 0,1)
test_swal_inner_product(48, 1,1)
test_swal_inner_product(48, 2,1)

test_swal_inner_product(48,-2,2)
test_swal_inner_product(48,-1,2)
test_swal_inner_product(48, 0,2)
test_swal_inner_product(48, 1,2)
test_swal_inner_product(48, 2,2)

test_swal_inner_product(48,-2,3)
test_swal_inner_product(48,-1,3)
test_swal_inner_product(48, 0,3)
test_swal_inner_product(48, 1,3)
test_swal_inner_product(48, 2,3)

"""
Computes the norm of the difference between the numerical and
exact spin-weighted spherical laplacian for a given
spin-weighted spherical harmonic.

test_norm_swal_lap(
   ny::Int64,
   spin::Int64,
   m_ang::Int64,
   l_ang::Int64
   )::Nothing
"""
function test_norm_swal_lap(
      ny::Int64,
      spin::Int64,
      m_ang::Int64,
      l_ang::Int64
   )::Nothing
   
   Yv = Sphere.Y_vals(ny)

   swal = [Sphere.swal(spin,m_ang,l_ang,y) for y in Yv]

   swal_lap_v1 = [-(l_ang-spin)*(l_ang+spin+1.0)*v for v in swal]
   swal_lap_v2 = zeros(Float64,ny) 
   
   lap = Sphere.swal_laplacian_matrix(ny,spin,m_ang)
   for j=1:ny
      for k=1:ny
         swal_lap_v2[j] += swal[k]*lap[k,j]
      end
   end

   ## compute integral over interval [-1,1] of difference
   n = Sphere.inner_product(ones(ny),swal_lap_v1 .- swal_lap_v2)

   println("Test: Δ_{S2}Y^s_{lm}:\tny=$ny\tspin=$spin\tm_ang=$m_ang\tl=$l_ang")
   @test abs(n) < tol 
   println("Passed test")

   return nothing
end

test_norm_swal_lap(32, 0,0,2)

test_norm_swal_lap(32,-1,1,2)
test_norm_swal_lap(32, 0,1,2)
test_norm_swal_lap(32, 1,1,2)

test_norm_swal_lap(32,-2,-2,2)
test_norm_swal_lap(32,-1,-2,2)
test_norm_swal_lap(32, 0,-2,2)
test_norm_swal_lap(32, 1,-2,2)
test_norm_swal_lap(32, 2,-2,2)

test_norm_swal_lap(32,-2,2,2)
test_norm_swal_lap(32,-1,2,2)
test_norm_swal_lap(32, 0,2,2)
test_norm_swal_lap(32, 1,2,2)
test_norm_swal_lap(32, 2,2,2)

test_norm_swal_lap(48,-2,5,23)
test_norm_swal_lap(48,-2,5,23)
test_norm_swal_lap(48,-1,5,23)
test_norm_swal_lap(48, 0,5,23)
test_norm_swal_lap(48, 1,5,23)
test_norm_swal_lap(48, 2,5,23)
"""
Test that the swal filter matrix acts as a low pass filter
for angular perturbations.
"""
function test_norm_filter_lap(
      ny::Int64,
      spin::Int64,
      m_ang::Int64,
      l_ang::Int64
   )::Nothing
   
   Yv = Sphere.Y_vals(ny)

   v1 = [0.1*(rand()-0.5)*Sphere.swal(spin,m_ang,l_ang,y) for y in Yv]
   v2 = zeros(Float64,ny) 

   filter = Sphere.swal_filter_matrix(ny,spin,m_ang)

   println(typeof(filter))
   
   for j=1:ny
      for k=1:ny
         v2[j] += v1[k]*filter[k,j]
      end
   end

   ## compute integral over interval [-1,1] of difference
   i1 = abs(Sphere.inner_product(ones(ny), v1))
   i2 = abs(Sphere.inner_product(ones(ny), v2))

   println("$i1\t$i2")
   println("Test: filter(rand) smoothens:\tny=$ny\tspin=$spin\tm_ang=$m_ang")
   @test i2 < i1 
   println("Passed test")

   return nothing
end

test_norm_filter_lap(32,-2,2,2)
test_norm_filter_lap(32,-1,2,2)
test_norm_filter_lap(32, 0,2,2)
test_norm_filter_lap(32, 1,2,2)
test_norm_filter_lap(32, 2,2,2)
