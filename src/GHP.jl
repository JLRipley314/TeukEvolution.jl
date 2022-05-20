"""
Geroch-Held-Penrose (GHP) operators; see 

   Geroch, Held, and Penrose, 
   Journal of Mathematical Physics 14, 874 (1973)

For the formulas in the coordinates we use, see
(in particular Sec V.C)
   
   Ripley, Loutrel, Giorgi, and Pretorius, 
   Phys.Rev.D 103 (2021) 104018, arXiv:2010.00162,
   
"""
module GHP

include("Fields.jl")
include("Sphere.jl")
using .Fields: Field
using .Sphere: swal_raising_matrix, swal_lowering_matrix, angular_matrix_mult!

export GHP_ops, Initialize_GHP_ops, set_edth!, set_edth_prime!, set_thorn!, set_thorn_prime!

struct GHP_ops

   pre_edth_DT::Array{ComplexF64,2}
   pre_edth_raised::Array{ComplexF64,2}
   pre_edth::Array{ComplexF64,2}
   pre_edth_prime_DT::Array{ComplexF64,2}
   pre_edth_prime_lowered::Array{ComplexF64,2}
   pre_edth_prime::Array{ComplexF64,2}
   pre_thorn_DT::Array{ComplexF64,2}
   pre_thorn_DR::Array{ComplexF64,2}
   pre_thorn::Array{ComplexF64,2}
   pre_thorn_prime_DT::Array{ComplexF64,2}
   pre_thorn_prime_DR::Array{ComplexF64,2} 

   raise::Array{ComplexF64,3}
   lower::Array{ComplexF64,3}

   smap::Dict{Int64,Int64}

   function GHP_ops(;
         Rvals::Vector{Float64},
         Cvals::Vector{Float64},
         Svals::Vector{Float64},
         mval::Int64,
         bhm::Float64,
         bhs::Float64,
         cl::Float64) 
     
      nx = length(Rvals)
      ny = length(Cvals)
  
      spins = [-2,-1,0,1,2]

      pre_edth_DT            = zeros(ComplexF64,nx,ny)
      pre_edth_raised        = zeros(ComplexF64,nx,ny) 
      pre_edth               = zeros(ComplexF64,nx,ny) 
      pre_edth_prime_DT      = zeros(ComplexF64,nx,ny) 
      pre_edth_prime_lowered = zeros(ComplexF64,nx,ny) 
      pre_edth_prime         = zeros(ComplexF64,nx,ny) 
      pre_thorn_DT           = zeros(ComplexF64,nx,ny) 
      pre_thorn_DR           = zeros(ComplexF64,nx,ny) 
      pre_thorn              = zeros(ComplexF64,nx,ny) 
      pre_thorn_prime_DT     = zeros(ComplexF64,nx,ny) 
      pre_thorn_prime_DR     = zeros(ComplexF64,nx,ny) 
  
      raise = zeros(ComplexF64,ny,ny,length(spins))
      lower = zeros(ComplexF64,ny,ny,length(spins))

      smap = Dict{Int64,Int64}()

      for j=1:ny
         cy = Cvals[j]
         sy = Svals[j]
         for i=1:nx
            R = Rvals[i]
            ##=============
            ## divided by R
            ##=============
            pre_edth_DT[i,j] = (1.0/sqrt(2.0))*(
                                    1.0/((cl^2) 
                                    - 
                                    im*bhs*R*cy)
                                )*(-im*bhs*sy)

            pre_edth_raised[i,j] = (1.0/sqrt(2.0))*(
                                    1.0/((cl^2) 
                                    - 
                                    im*bhs*R*cy)
                                   )

            pre_edth[i,j] = (
                  (im*bhs*R*sy/sqrt(2.0))
                  /  
                  ((im*(cl^2) + bhs*R*cy)^2)
                 )
            
            ##=============
            ## divided by R
            ##=============
            pre_edth_prime_DT[i,j] = (1.0/sqrt(2.0))*(
                                       1.0/((cl^2) 
                                       + 
                                       im*bhs*R*cy)
                                      )*im*bhs*sy

            pre_edth_prime_lowered[i,j] = (1.0/sqrt(2.0))*(
                                             1.0/((cl^2) 
                                             + 
                                             im*bhs*R*cy)
                                            )

            pre_edth_prime[i,j] = (
                  (im*bhs*R*sy/sqrt(2.0))
                  /  
                  ((cl^2 + im*bhs*R*cy)^2)
                 )
            
            ##=============
            ## divided by R
            ##=============
            pre_thorn_DT[i,j] = (
               (1.0/((cl^4)+((bhs*R*cy)^2)))
               *
               R*2.0*bhm*(2.0*bhm-((bhs/cl)^2)*R)
              )

            ##=============
            ## divided by R
            ##=============
            pre_thorn_DR[i,j] = (
               (1.0/((cl^4)+((bhs*R*cy)^2)))*(
               -  0.5*((cl^2)-(2.0*bhm*R) + ((bhs*R/cl)^2))
               )
              )

            ##=================
            ## NOT divided by R
            ##=================
            pre_thorn[i,j] = (
               (1.0/((cl^4)+((bhs*R*cy)^2)))*R*(im*bhs)
              )
            
            pre_thorn_prime_DT[i,j] = (
               (2.0 + (4.0*bhm*R/(cl^2)))
              )

            ##=============
            ## divided by R
            ##=============
            pre_thorn_prime_DR[i,j] = ((1.0/cl)^2)*R
         end
      end
    
      for (i,s) in enumerate(spins)
         tmp_raise = swal_raising_matrix( ny,s,mval)
         tmp_lower = swal_lowering_matrix(ny,s,mval)

         raise[:,:,i] .= tmp_raise
         lower[:,:,i] .= tmp_lower
      end

      for (i,s) in enumerate(spins)
         smap[s]=i
      end

      return new(
         pre_edth_DT,
         pre_edth_raised, 
         pre_edth,
         pre_edth_prime_DT, 
         pre_edth_prime_lowered, 
         pre_edth_prime, 
         pre_thorn_DT, 
         pre_thorn_DR, 
         pre_thorn, 
         pre_thorn_prime_DT, 
         pre_thorn_prime_DR,
         raise,
         lower,
         smap
      )
   end
end   

function Initialize_GHP_ops(;
      Rvals::Vector{Float64},
      Cvals::Vector{Float64},
      Svals::Vector{Float64},
      Mvals::Vector{Int64},
      bhm::Float64,
      bhs::Float64,
      cl::Float64) 
   return Dict([
      (mv,GHP_ops(Rvals=Rvals,Cvals=Cvals,Svals=Svals,mval=mv,bhm=bhm,bhs=bhs,cl=cl)) 
      for mv in Mvals
     ])
end

function set_edth!(;
      edth ::Array{ComplexF64,2},
      spin ::Int64, 
      boost::Int64, 
      m_ang::Int64, 
      f     ::Array{ComplexF64,2}, 
      DT    ::Array{ComplexF64,2}, 
      raised::Array{ComplexF64,2},
      Op
     )
   nx, ny = size(edth)
   p = (spin+boost)
   
   angular_matrix_mult!(raised,f,view(Op.raise,:,:,Op.smap[spin]))
   
   for j=1:ny
      for i=1:nx
         edth[i,j] = (
            Op.pre_edth_DT[i,j]     *DT[i,j]
            +  
            Op.pre_edth_raised[i,j] *raised[i,j]
            +  
            p*Op.pre_edth[i,j]*f[i,j]
        )
      end
   end
   return nothing
end 

function set_edth_prime!(;
      edth_prime::Array{ComplexF64,2},
      spin ::Int64, 
      boost::Int64, 
      m_ang::Int64, 
      f      ::Array{ComplexF64,2},
      DT     ::Array{ComplexF64,2}, 
      lowered::Array{ComplexF64,2},
      Op
     )
   nx, ny = size(edth)
   q = (-spin+boost)

   angular_matrix_mult!(lowered,f,view(Op.lower,:,:,Op.smap[spin])) 

   for j=1:ny
      for i=1:nx
         edth_prime[i,j] = (
            Op.pre_edth_prime_DT[i,j]     *DT[i,j]
            +  
            Op.pre_edth_prime_lowered[i,j]*lowered[i,j]
            +  
            q*Op.pre_edth_prime[i,j]*f[i,j]
           )
      end
   end
   return nothing
end 

function set_thorn!(;
      thorn::Array{ComplexF64,2},
      spin ::Int64, 
      boost::Int64, 
      m_ang::Int64,
      falloff::Int64, 
      f    ::Array{ComplexF64,2}, 
      DT   ::Array{ComplexF64,2}, 
      DR   ::Array{ComplexF64,2},
      ep_0 ::Array{ComplexF64,2},
      R    ::Vector{Float64},
      Op
     )
   nx, ny = size(thorn)
   p = ( spin+boost)
   q = (-spin+boost)

   for j=1:ny
      for i=1:nx
         thorn[i,j] = (
            Op.pre_thorn_DT[i,j]*DT[i,j]
            +  
            Op.pre_thorn_DR[i,j]*(R[i]*DR[i,j] + falloff*f[i,j])
            +  
            m_ang*Op.pre_thorn[i,j]*f[i,j]
            -  
            R[i]*(p*ep_0[i,j] + q*conj(ep_0[i,j]))*f[i,j]
           )
      end
   end
   return nothing
end

##===========================================================================
## no rescaling in R for thorn prime
##===========================================================================
function set_thorn_prime!(;
      thorn_prime::Array{ComplexF64,2},
      falloff::Int64, 
      f ::Array{ComplexF64,2}, 
      DT::Array{ComplexF64,2}, 
      DR::Array{ComplexF64,2}, 
      R ::Vector{Float64},
      Op
     )
   for j=1:ny
      for i=1:nx
         thorn_prime_arr[i,j] = (
            Op.pre_thorn_prime_DT[i,j]*DT[i,j]
            +  
            Op.pre_thorn_prime_DR[i,j]*(R[i]*DR[i,j] + falloff*f[i,j])
           )
      end
   end
   return nothing
end 

end
