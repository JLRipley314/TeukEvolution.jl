module Fields

export Field

struct Field
   name::String

   spin::Int64
   boost::Int64
   falloff::Int64

   nx::Int64
   ny::Int64
   nz::Int64

   n  ::Array{ComplexF64}
   tmp::Array{ComplexF64}
   np1::Array{ComplexF64}

   k::Array{ComplexF64} 

   rad_d1::Array{ComplexF64} 
   rad_d2::Array{ComplexF64} 
   
   raised ::Array{ComplexF64} 
   lowered::Array{ComplexF64}  
   sph_lap::Array{ComplexF64} 

   Field(name, 
         spin, boost, falloff, 
         nx, ny, nz,
        ) = new(name, 
                spin, boost, falloff,
                nx, ny, nz, 
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz),
                zeros(nx,ny,nz)
               )
end

end
