module Io 

export save_csv

"""
Save the real and imaginary parts of level np1
"""
function save_csv(;
      t::Float64,
      mv::Int64,
      outdir::String,
      f)
      
   nx, ny = f.nx, f.ny 

   open("$(outdir)/$(f.name)_re_$mv.csv","a") do out
      write(out,"$t,$nx,$ny,")
      for i=1:nx
         for j=1:ny
            write(out,"$(f.np1[i,j].re),")
            saved_once=true
         end
      end
      write(out,"\n")
   end

   open("$(outdir)/$(f.name)_im_$mv.csv","a") do out
      write(out,"$t,$nx,$ny,")
      for i=1:nx
         for j=1:ny
            write(out,"$(f.np1[i,j].im),")
            saved_once=true
         end
      end
      write(out,"\n")
   end

   return nothing
end

end
