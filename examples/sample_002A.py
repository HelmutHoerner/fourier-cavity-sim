import math
from fo_cavity_sim import clsCavity1path, clsMirror, clsPropagation, \
    clsThinLens, clsSpeckleField, get_sample_vec

if __name__ == '__main__':  
    # --- SIMULATION PARAMETERS -----------------------------------------------
    R_left = 0.7                   # reflectivity left mirrror
    R_right = 0.999                # reflectivity right mirror
    d_absorb = 0.0006              # thickness of absorber
    T_abs = math.sqrt(R_left/R_right) # optimal transmittivity of absorber
    pos_absorb = 0.005             # distance of absorber after left mirror
    nr = 1.5                       # real part of absorber's refractive index
    f1 = 0.075                     # focal length first lens  
    f2 = f1 - d_absorb/2*(nr-1/nr) # second lens
    length_fov = 0.0021            # field-of-view sidelength    
    epsilon = 1e-6                 # accuracy of steady-state approximation
    TSR = R_left * R_right * T_abs**2     # single roundtrip transmittivity
    no_of_RT = int(round(math.log(epsilon)/math.log(TSR))) # roundtrips
    
    # --- CREATE CAVITY -------------------------------------------------------
    myCavity = clsCavity1path("4f cavity")
    myCavity.use_swap_files_in_bmatrix_class = False
    myCavity.folder = "cavity_folder"
    myCavity.Lambda_nm = 633    
    
    # --- DETERMINE RESONANCE WAVELENGTH, CREATE OPTIMAL GRID -----------------
    lambda_c, k_c, n_c, lambda_period = \
        myCavity.resonance_data_simple_cavity(R_left, R_right, 4*f1)
    myCavity.Lambda = lambda_c 
    myCavity.grid.set_opt_res_tot_based_on_res_fov(length_fov, 100, 2*f1) 
    print(f"Field-of-view:  {myCavity.grid.res_fov}x{myCavity.grid.res_fov}")
    print(f"Total grid res: {myCavity.grid.res_tot}x{myCavity.grid.res_tot}")
    print("Number of roundtrips:", no_of_RT)
    
    # --- GENERATE COMPONENTS -------------------------------------------------
    # left input coupling mirror
    mirror1 = clsMirror("left mirror", myCavity)    
    mirror1.R = R_left
    
    # propagation over distance f1
    prop_f1 = clsPropagation("f propagation", myCavity)    
    prop_f1.set_params(f1, 1)
    
    # lens with focal length f
    lens_f1 = clsThinLens("lens 1", myCavity)        
    lens_f1.f = f1      
    
    # propagation over distance f2
    prop_f2 = clsPropagation("f2 propagation", myCavity)    
    prop_f2.set_params(f2, 1)    
    
    # lens with focal length f2
    lens_f2 = clsThinLens("lens 2", myCavity)    
    lens_f2.f = f2                    
    
    # propagation between second lens and absorber 
    prop_lens_absorb = clsPropagation("propagation lens<>absorber", myCavity)    
    prop_lens_absorb.set_params(f2-pos_absorb-d_absorb/nr, 1)                    
    
    # absorber with thickness d_absorb
    absorber = clsPropagation("absorber", myCavity)    
    absorber.set_params(d_absorb, nr)    
    absorber.set_ni_based_on_T(T_abs)            
    
    # propagation between absorber and right mirror
    prop_abs_mirr = clsPropagation("propagation absorber<>mirror2", myCavity)    
    prop_abs_mirr.set_params(pos_absorb, 1)     
    
    # right mirror
    mirror2 = clsMirror("right mirror", myCavity)    
    mirror2.R = R_right
    
    # --- ASSEMBLE 4F CAVITY --------------------------------------------------
    myCavity.add_component(mirror1)             # 0
    myCavity.add_component(prop_f1)             # 1
    myCavity.add_component(lens_f1)             # 2
    myCavity.add_component(prop_f1)             # 3
    myCavity.add_component(prop_f2)             # 4
    myCavity.add_component(lens_f2)             # 5
    myCavity.add_component(prop_lens_absorb)    # 6
    myCavity.add_component(absorber)            # 7
    myCavity.add_component(prop_abs_mirr)       # 8
    myCavity.add_component(mirror2)             # 9
    
    # --- INPUT: RANDOM SPECKLE FIELD -----------------------------------------
    inp = clsSpeckleField(myCavity.grid)    
    inp.create_field_eq_distr(100, 20, 0.6*myCavity.grid.length_fov, 0)
    inp.name = "Input Field"    
    inp.plot_field(5)
    myCavity.incident_field_left = inp
    
    # --- CALCULATE CAVITY'S LEFT OUTPUT FOR 3 DIFFERENT WAVELENGHTS ----------    
    lambda_vec = get_sample_vec(3, lambda_period/120, 0, False, 1)
    for index, dLambda in enumerate(lambda_vec):
        # SET CURRENT WAVELENGTH
        dL_pm = dLambda * 10**12 # delta lambda in pm
        print("")
        print(f"***   delta lambda = {dL_pm:.3f} pm   ***")
        myCavity.Lambda = lambda_c + dLambda     

        # CALCULATE OUTPUT FIELD WITH "MULTIPLE ROUNDTRIPS" METHOD"           
        myCavity.prop_mult_round_trips_LTR_RTL(no_of_RT ,0, 9)
        out = myCavity.output_field_left
        out.name = f"delta lambda = {dL_pm:.3f} pm"
        out.plot_field(5, vmax_limit=0.03, c_map = "custom")
        
        # CALCULATE REFLECTION COEFFICIENT OVER FOV AND SAVE RESULT TO FILE
        r = out.intensity_integral_fov() / inp.intensity_integral_fov()
        print(f"Reflectivity: {r}")
        data = [index, dL_pm, r]
        myCavity.write_to_file("result.csv",data)
    
    