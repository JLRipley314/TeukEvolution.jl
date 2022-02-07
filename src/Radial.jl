module Radial

export R_vals, set_d1!, set_d2!, filter!

"""
Computes array of compactified radial points 

R_vals(
   nx::Int64, 
   dr::Float64 
   )::Vector{Float64}
"""
function R_vals(nx::Int64, dr::Float64)::Vector{Float64}
   return [dr*i for i=1:nx]
end
"""
Takes first derivative in radial direction.

set_d1!(
   dv, 
   v,
   dr::Float64,
   )
"""
function set_d1!(
      dv, 
      v,
      dr::Float64,
     )
   inv_h_dr = 0.5/dr
   inv_dr   = 1.0/dr

   nx, ny = size(dv)

   for j=1:ny
      for i=2:nx-1
         dv[i,j] = (v[i+1,j]-v[i-1,j])*inv_h_dr
      end
      dv[1, j]  = (-0.5*v[   3,j] + 2*v[   2,j] - 1.5*v[ 1,j])*inv_dr
      dv[nx,j]  = ( 0.5*v[nx-2,j] - 2*v[nx-1,j] + 1.5*v[nx,j])*inv_dr
   end
   return nothing
end
"""
Takes second derivative in radial direction 

set_d2!(
   dv, 
   v,
   dr::Float64
   )
"""
function set_d2!(
      dv, 
      v,
      dr::Float64
     )
   inv_dr2 = 1.0/dr^2

   nx, ny = size(dv)

   for j=1:ny
      for i=2:nx-1
         dv[i,j] = (v[i+1,j] - 2.0*v[i,j] + v[i-1,j])*inv_dr2
      end
      dv[1,j]  = (-1*v[4,j]    + 4.0*v[   3,j] - 5.0*v[   2,j] + 2.0*v[ 1,j])*inv_dr2
      dv[nx,j] = (-1*v[nx-3,j] + 4.0*v[nx-2,j] - 5.0*v[nx-1,j] + 2.0*v[nx,j])*inv_dr2
   end
   return nothing
end
"""
Low pass filter (ep controls how strongly the filter should ask,
and should be in [0,1]) 

filter!(
   v, 
   ep::Float64
   )
"""
function filter!(
      v, 
      tmp, 
      ep::Float64
     )
   nx, ny = size(v)

   for j=1:ny
      for i=3:nx-2
         v[i,j] = (
         +        (    -ep/16.) *tmp[i-2,j]
         +        ( 4.0*ep/16.) *tmp[i-1,j]
         + (1.0 + (-6.0*ep/16.))*tmp[i  ,j]
         +        ( 4.0*ep/16.) *tmp[i+1,j]
         +        (    -ep/16.) *tmp[i+2,j]
         )
      end
      v[1,j]    = tmp[1,j]
      v[2,j]    = tmp[2,j]
      v[nx-1,j] = tmp[nx-1,j]
      v[nx  ,j] = tmp[nx  ,j]
   end
   return nothing
end

end
