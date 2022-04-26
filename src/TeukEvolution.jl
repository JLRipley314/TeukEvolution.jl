module TeukEvolution

include("Fields.jl")
include("Io.jl")
include("Radial.jl")
include("Sphere.jl")
include("Id.jl")
include("Evolution.jl")
include("GHP.jl")

using .Fields: Field
import .Io
import .Radial
import .Sphere
import .Id
import .Evolution as Evo
import .GHP

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
   ##===================
   ## Derived parameters
   ##===================
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

   ##=================
   ## Dynamical fields 
   ##=================
   println("Initializing linear psi4")
   psi4_lin_f = Field(name="psi4_lin_f",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,nx=nx,ny=ny,nz=nm)
   psi4_lin_p = Field(name="psi4_lin_p",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,nx=nx,ny=ny,nz=nm)
   
   println("Initializing metric reconstruction fields")
   psi3_f = Field(name="psi3",spin=-1,boost=-1,falloff=2,nx=nx,ny=ny,nz=nm)
   psi2_f = Field(name="psi2",spin= 0,boost= 0,falloff=3,nx=nx,ny=ny,nz=nm)

   la_f   = Field(name="la",spin=-2,boost=-1,falloff=1,nx=nx,ny=ny,nz=nm)
   pi_f   = Field(name="pi",spin=-1,boost= 0,falloff=2,nx=nx,ny=ny,nz=nm)

   muhll_f = Field(name="muhll",spin= 0,boost=1,falloff=3,nx=nx,ny=ny,nz=nm)
   hlmb_f  = Field(name="hlmb" ,spin=-1,boost=1,falloff=2,nx=nx,ny=ny,nz=nm)
   hmbmb_f = Field(name="hmbmb",spin=-2,boost=0,falloff=1,nx=nx,ny=ny,nz=nm)
  
   println("Initializing independent residuals")
   res_bianchi3_f = Field(name="res_bianchi3",spin=-2,boost=-1,falloff=2,nx=nx,ny=ny,nz=nm)
   res_bianchi2_f = Field(name="res_bianchi2",spin=-1,boost= 0,falloff=2,nx=nx,ny=ny,nz=nm)
   res_hll_f      = Field(name="res_hll",     spin= 0,boost= 2,falloff=2,nx=nx,ny=ny,nz=nm)
  
   println("Initializing 2nd order psi4")
   psi4_scd_f = Field(name="psi4_scd_f",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,nx=nx,ny=ny,nz=nm)
   psi4_scd_p = Field(name="psi4_scd_p",spin=psi_spin,boost=psi_spin,falloff=psi_falloff,nx=nx,ny=ny,nz=nm)
   
   ##=======================================
   ## Fixed fields (for evolution equations) 
   ##=======================================
   println("Initializing psi4 evolution operators")
   evo_psi4 = Evo.Evo_psi4(Rvals=Rv,Cvals=Cv,Svals=Sv,Mvals=Mv,bhm=bhm,bhs=bhs,cl=cl,spin=psi_spin)
   
   println("Initializing GHP operators")
   ghp = GHP.GHP_ops(Rvals=Rv,Cvals=Cv,Svals=Sv,Mvals=Mv,bhm=bhm,bhs=bhs,cl=cl)
   ##=============
   ## Initial data
   ##=============
   println("Initial data")
 
   if params["id_kind"]=="gaussian"
      for mi=1:nm
         Id.set_gaussian!(psi4_lin_f, psi4_lin_p, 
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
         Io.save_csv(0,mi,Mv[mi],Rv,Yv,outdir,psi4_lin_f)
         #Io.save_csv(0,mi,Mv[mi],Rv,Yv,outdir,psi4_lin_p)
      end
   elseif params["id_kind"]=="qnm"
      Id.set_qnm!()
   else
      throw(DomainError(params["id_kind"],"Unsupported `id_kind`")) 
   end
   
   ##===================
   ## Time evolution 
   ##===================
   println("Beginning evolution")
 
   for tc=1:nt
      Threads.@threads for mi=1:nm
         Evo.evolve_psi4(psi4_lin_f,psi4_lin_p,evo_psi4,mi,dr,dt) 
         
         for j=1:ny
            for i=1:nx
               psi4_lin_f.n[i,j,mi] = psi4_lin_f.np1[i,j,mi] 
               psi4_lin_p.n[i,j,mi] = psi4_lin_p.np1[i,j,mi] 
            end
         end
      end
      
      if tc%ts==0
         println("time/bhm ", tc*dt/bhm)
         Threads.@threads for mi=1:nm
            Io.save_csv(tc,mi,Mv[mi],Rv,Yv,outdir,psi4_lin_f)
            #Io.save_csv(tc,mi,Mv[mi],Rv,Yv,outdir,psi4_lin_p)
         end 
      end
   end
   println("Finished evolution")
   return nothing
end

end
