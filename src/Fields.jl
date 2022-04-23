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

   n  ::Array{ComplexF64,3}
   tmp::Array{ComplexF64,3}
   np1::Array{ComplexF64,3}

   k::Array{ComplexF64,3} 

   rad_d1::Array{ComplexF64,3} 
   rad_d2::Array{ComplexF64,3} 
   
   raised ::Array{ComplexF64,3} 
   lowered::Array{ComplexF64,3}  
   sph_lap::Array{ComplexF64,3} 

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
