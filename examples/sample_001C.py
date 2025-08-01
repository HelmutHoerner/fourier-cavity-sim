import math
from fo_cavity_sim import clsCavity1path, clsPropagation, \
                         clsGrating, clsPlaneWaveMixField, Dir 
                         
if __name__ == '__main__':
    length_fov = 0.001 # field-of-view sidelength
    
    # --- BUILD CAVITY BREADBOARD ---------------------------------------------
    myCavity = clsCavity1path("my cavity")   
    myCavity.Lambda_nm = 633
    
    # --- APPROXIMATE TALBOT DISTANCE -----------------------------------------
    d_0 = 0.0001 # assumed grating period
    zt_0 = 2 * d_0**2 / myCavity.Lambda # approximate Talbot distance
    
    # --- CREATE OPTIMAL GRID FOR zt/4 PROPAGATION ----------------------------
    myCavity.grid.set_opt_res_based_on_sidelength(length_fov, 1.5, zt_0/4, True)
    
    # --- GENERATE COMPONENTS -------------------------------------------------
    # 0: cosine phase grating   
    grating = clsGrating("phase grating", myCavity)
    grating.phase_grating = True
    dx, x_max, dy, y_max = grating.set_cos_grating(d_0, length_fov, math.inf,0, 
                                                   True, 0, math.pi/2)
    # 1: free space propagation over 1/4 talbot-distance
    prop = clsPropagation("zt/4 propagation", myCavity)
    # calculate exact talbot distance based on actual phase grating
    zt = myCavity.Lambda / (1 - math.sqrt(1-myCavity.Lambda**2/dx**2)) 
    prop.set_params(zt/4, 1)
    prop.transfer_function = 0
    
    # --- ASSEMBLE OPTICAL SYSTEM ---------------------------------------------
    myCavity.add_component(grating)   # 0
    myCavity.add_component(prop)      # 1
    
    # --- INCIDENT PLANE WAVE -------------------------------------------------
    input_field = clsPlaneWaveMixField(myCavity.grid)
    input_field.fov_only = False
    input_field.add_fourier_basis_func(0, 0, 1)
    
    # --- PLOT INTENSITY IMMEDIATELY AFTER GRATING ----------------------------
    field_0 = myCavity.prop(input_field, 0, 0, Dir.LTR)
    field_0.name = "Constant Intensity after Phase Grating"
    field_0.plot_field(5, False)
    # --- PLOT PHASE IMMEDIATELY AFTER GRATING --------------------------------
    field_0.name = "Varying Phase after Phase Grating"
    field_0.plot_field(4, False) 
    
    # --- PROPAGATE zt/4 AND PLOT INTENSITY -----------------------------------
    output_field = myCavity.prop(input_field, 0, 1, Dir.LTR)
    output_field.name = "Varying Intensity after zt/4 Propagation"
    output_field.plot_field(5, False) 
    
    
  