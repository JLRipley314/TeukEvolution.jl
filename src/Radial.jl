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
   inv_dr   = 1.0/dr

   nx, ny = size(dv)

   for j=1:ny
      for i=3:nx-2
         dv[i,j] = (-
                    (1.0/12.0)*v[i+2,j]
                    +
                    (2.0/3.0 )*v[i+1,j]
                    -
                    (2.0/3.0 )*v[i-1,j]
                    +
                    (1.0/12.0)*v[i-2,j]
                   )*inv_dr
      end
      dv[1,j] = (-
                 3.0 *v[5,j]
                 +
                 16.0*v[4,j]
                 -
                 36.0*v[3,j] 
                 +
                 48.0*v[2,j]
                 -
                 25.0*v[1,j]
                )*(1.0/12.0)*inv_dr
      dv[2,j] = (+
                 1.0 *v[5,j]
                 -
                 6.0 *v[4,j]
                 +
                 18.0*v[3,j]
                 -
                 10.0*v[2,j]
                 -
                 3.0 *v[1,j]
                )*(1.0/12.0)*inv_dr
      
      dv[nx,j] = -(-
                   3.0 *v[nx-4,j]
                   +
                   16.0*v[nx-3,j]
                   -
                   36.0*v[nx-2,j] 
                   +
                   48.0*v[nx-1,j]
                   -
                   25.0*v[nx,j]
                  )*(1.0/12.0)*inv_dr
      dv[nx-1,j] = -(+
                     1.0 *v[nx-4,j]
                     -
                     6.0 *v[nx-3,j]
                     +
                     18.0*v[nx-2,j]
                     -
                     10.0*v[nx-1,j]
                     -
                     3.0 *v[nx,j]
                    )*(1.0/12.0)*inv_dr
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
      for i=3:nx-2
         dv[i,j] = (-
                    1.0 *v[i+2,j]
                    +
                    16.0*v[i+1,j]
                    -
                    30.0*v[i  ,j]
                    +
                    16.0*v[i-1,j]
                    -
                    1.0 *v[i-2,j]
                   )*(1.0/12.0)*inv_dr2
      end
      dv[1,j] = (-
                 10.0 *v[6,j]
                 +
                 61.0 *v[5,j]
                 -
                 156.0*v[4,j]
                 +
                 214.0*v[3,j]
                 -
                 154.0*v[2,j]
                 +
                 45.0 *v[1,j]
                )*(1.0/12.0)*inv_dr2
      dv[2,j] = (+
                 1.0 *v[6,j]
                 -
                 6.0 *v[5,j]
                 +
                 14.0*v[4,j]
                 -
                 4.0 *v[3,j]
                 -
                 15.0*v[2,j]
                 +
                 10.0*v[1,j]
                )*(1.0/12.0)*inv_dr2
      dv[nx,j] = (-
                  10.0 *v[nx-5,j]
                  +
                  61.0 *v[nx-4,j]
                  -
                  156.0*v[nx-3,j]
                  +
                  214.0*v[nx-2,j]
                  -
                  154.0*v[nx-1,j]
                  +
                  45.0 *v[nx  ,j]
                 )*(1.0/12.0)*inv_dr2
      dv[nx-1,j] = (+
                    1.0 *v[nx-5,j]
                    -
                    6.0 *v[nx-4,j]
                    +
                    14.0*v[nx-3,j]
                    -
                    4.0 *v[nx-2,j]
                    -
                    15.0*v[nx-1,j]
                    +
                    10.0*v[nx  ,j]
                   )*(1.0/12.0)*inv_dr2
   end
   return nothing
end

"""
6th order Kreiss-Oliger dissipation
""" 
function filter!(
      v,
      tmp,
      eps_KO::Float64 
     )
   nx, ny = size(v)

   for j=1:ny
      for i=1:nx
         tmp[i,j] = v[i,j]
      end
   end

   for j=1:ny
      for i=4:nx-3
         v[i,j] = (
                         (eps_KO/64.) *tmp[i-3,j]
           +        (-6.0*eps_KO/64.) *tmp[i-2,j]
           +        (15.0*eps_KO/64.) *tmp[i-1,j]
           + (1.0 - (20.0*eps_KO/64.))*tmp[i  ,j]
           +        (15.0*eps_KO/64.) *tmp[i+1,j]
           +        (-6.0*eps_KO/64.) *tmp[i+2,j]
           +             (eps_KO/64.) *tmp[i+3,j]
          )
      end
   end
   
   return nothing
end

end
