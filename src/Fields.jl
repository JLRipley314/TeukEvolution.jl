module Fields

export Field, Initialize_Field

struct Field
   name::String

   spin::Int64
   boost::Int64
   falloff::Int64

   mv::Int64

   nx::Int64
   ny::Int64

   n  ::Array{ComplexF64,2}
   tmp::Array{ComplexF64,2}
   np1::Array{ComplexF64,2}

   k::Array{ComplexF64,2} 

   rad_d1::Array{ComplexF64,2} 
   rad_d2::Array{ComplexF64,2} 
   
   raised ::Array{ComplexF64,2} 
   lowered::Array{ComplexF64,2}  
   sph_lap::Array{ComplexF64,2} 

   edth       ::Array{ComplexF64,2}
   edth_prime ::Array{ComplexF64,2}
   thorn      ::Array{ComplexF64,2}
   thorn_prime::Array{ComplexF64,2}

   Field(;
         name, 
         spin, boost, falloff, mv, 
         nx, ny
        ) = new(name, 
                spin, boost, falloff, mv,
                nx, ny, 
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny),
                zeros(ComplexF64,nx,ny)
               )
end

function Initialize_Field(;
      name::String,
      spin::Int64,
      boost::Int64,
      falloff::Int64,
      Mvals::Vector{Int64},
      nx::Int64,
      ny::Int64)
   return Dict([
      (mv,Field(name=name,spin=spin,boost=boost,falloff=falloff,mv=mv,nx=nx,ny=ny)) 
      for mv in Mvals
     ]) 
end

end
