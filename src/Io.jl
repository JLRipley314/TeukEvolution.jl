module Io 

include("Fields.jl")

using .Fields: Field

export save_csv

"""
Save the real and imaginary parts of level np1

save_csv(
   tc::Int64,
   mi::Int64,
   mv::Int64,
   f::Field
   )
"""
function save_csv(
      tc::Int64,
      mi::Int64,
      mv::Int64,
      rv::Vector{Float64},
      yv::Vector{Float64},
      outdir::String,
      f::Field)
      
   nx, ny = f.nx, f.ny 

   open("$(outdir)/$(f.name)_$(mv)_re_$tc.csv","a") do out
      write(out,"R,Y,val\n")
      for i=1:nx
         for j=1:ny
            if abs(f.np1[i,j,mi].re)>1e-16
               write(out,"$(rv[i]),$(yv[j]),$(f.np1[i,j,mi].re)\n")
            end
         end
      end
   end

   open("$(outdir)/$(f.name)_$(mv)_im_$tc.csv","a") do out
      write(out,"R,Y,val\n")
      for i=1:nx
         for j=1:ny
            if abs(f.np1[i,j,mi].im)>1e-16
               write(out,"$(rv[i]),$(yv[j]),$(f.np1[i,j,mi].im)\n")
            end
         end
      end
   end

   return nothing
end

end
