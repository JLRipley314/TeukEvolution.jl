module TeukEvolution

include("Fields.jl")
include("Io.jl")
include("Radial.jl")
include("Sphere.jl")
include("Id.jl")
include("Evolution.jl")

using .Fields: Field
import .Io
import .Radial
import .Sphere
import .Id
import .Evolution as Evo

import TOML

#import MPI
#MPI.Init()
#const comm = MPI.COMM_WORLD
#const nm = MPI.Comm_size(comm) ## number of m angular numbers

#const mi = 1 + MPI.Comm_rank(comm) ## m index
#const m_val = params["m_vals"][mi] ## value of m angular number

function launch(paramfile::String)
   println("Launching run, params=$paramfile")
   params = TOML.parsefile(paramfile)

   nx = convert(Int64,params["nx"])
   ny = convert(Int64,params["ny"])
   nt = convert(Int64,params["nt"])
   ts = convert(Int64,params["ts"])

   psi_spin    = convert(Int64,params["psi_spin"])
   psi_falloff = convert(Int64,params["psi_falloff"])

   cl  = convert(Float64,params["cl"])
   cfl = convert(Float64,params["cfl"])
   bhs = convert(Float64,params["bhs"])
   bhm = convert(Float64,params["bhm"])

   outdir = params["outdir"]
   ##
   ## Derived parameters
   ##
   nm   = length(params["m_vals"])
   minr = bhm*(
      1.0 + sqrt(1.0+(bhs/bhm))*sqrt(1.0-(bhs/bhm))
     ) # horizon (uncompactified)
   maxR = (cl^2)/minr
   dr   = maxR/(nx-1.0)
   dt   = min(cfl*dr,6.0/ny^2)
   
   println("Number of threads: $(Threads.nthreads())")

   println("Setting up output directory")
   if !isdir(outdir)
      mkdir(outdir)
   else
      rm(outdir,recursive=true)
      mkdir(outdir)
   end
   println("Initializing constant fields")
   Rv = Radial.R_vals(nx, dr)
   Yv = Sphere.Y_vals(ny)
   Cv = Sphere.cos_vals(ny)
   Sv = Sphere.sin_vals(ny)
   Mv = params["m_vals"]
   time = 0.0

   println("Initializing dynamical fields")
   psi4_f = Field("psi4_f",psi_spin,psi_spin,psi_falloff,nx,ny,nm)
   psi4_p = Field("psi4_p",psi_spin,psi_spin,psi_falloff,nx,ny,nm)
   
   println("Initializing psi4 evolution fields")
   evo_psi4 = Evo.Evo_psi4(Rv,Cv,Sv,Mv,bhm,bhs,cl,psi_spin)
   ##=============
   ## Initial data
   ##=============
   println("Initial data")
   
   for mi=1:nm
      Id.set_psi(psi4_f, psi4_p, 
         psi_spin,
         mi,
         Mv[mi],
         params["id_l_ang"][mi],
         params["id_ru"][mi], 
         params["id_rl"][mi], 
         params["id_width"][mi],
         params["id_amp"][mi][1] + params["id_amp"][mi][2]*im,
         cl, Rv, Yv
      )
      Io.save_csv(0,mi,Mv[mi],Rv,Yv,outdir,psi4_f)
      #Io.save_csv(0,mi,Mv[mi],Rv,Yv,outdir,psi4_p)
   end
   println("Beginning evolution")
   for tc=1:nt
      Threads.@threads for mi=1:nm
         Evo.evolve_psi4(psi4_f,psi4_p,evo_psi4,mi,dr,dt) 
         for j=1:ny
            for i=1:nx
               psi4_f.n[i,j,mi] = psi4_f.np1[i,j,mi] 
               psi4_p.n[i,j,mi] = psi4_p.np1[i,j,mi] 
            end
         end
      end
      
      if tc%ts==0
         Threads.@threads for mi=1:nm
            println("time/bhm ", tc*dt/bhm)
            Io.save_csv(tc,mi,Mv[mi],Rv,Yv,outdir,psi4_f)
            #Io.save_csv(tc,mi,Mv[mi],Rv,Yv,outdir,psi4_p)
         end 
      end
   end
   println("Finished evolution")
   return nothing
end

end
