import math
from fo_cavity_sim import clsCavity1path, clsMirror, clsPropagation, \
    clsThinLens, clsSpeckleField, get_sample_vec

if __name__ == '__main__':    
    # --- SIMULATION PARAMETERS -----------------------------------------------
    R_left = 0.7                         # right mirrror
    R_right = 0.999                      # left mirror
    R_center =  4*R_left/((1+R_left)**2) # center mirror
    d_absorb = 0.0006                    # thickness of absorber
    T_abs = math.sqrt(R_left/R_right)    # optimal absorption
    pos_absorb = 0.005                   # distance absorber after left mirror
    nr = 1.5                             # absorber's refractive index
    f1 = 0.015                           # focal length first lens  
    f2 = f1 - d_absorb/2*(nr-1/nr)       # focal length second lens  
    length_fov =  0.00081                # field-of-view sidelength
    
    # --- CREATE CAVITY -------------------------------------------------------
    myCavity = clsCavity1path("8f cavity")
    myCavity.use_swap_files_in_bmatrix_class = True
    myCavity.folder = "cavity_folder"
    myCavity.tmp_folder = "tmp_folder"
    myCavity.Lambda_nm = 633    
    
    # --- DETERMINE RESONANCE WAVELENGTH, CREATE GRID -------------------------
    lambda_c, k_c, n_c, lambda_period, l2_corr = \
        myCavity.resonance_data_8f_cavity(R_left, R_center, R_right, f1)
    l2_corr *= 1.212
    myCavity.Lambda = lambda_c        
    myCavity.grid.set_opt_res_based_on_sidelength(length_fov, 2, 2*f1, True)
    print(f"Field-of-view:  {myCavity.grid.res_fov}x{myCavity.grid.res_fov}")
    print(f"Total grid res: {myCavity.grid.res_tot}x{myCavity.grid.res_tot}")
    
    # --- GENERATE COMPONENTS -------------------------------------------------
    # input coupling miror
    mirror1 = clsMirror("left mirror", myCavity)    
    mirror1.R = R_left
    
    # f1 propagation
    prop_f1 = clsPropagation("f1 propagation", myCavity)    
    prop_f1.set_params(f1, 1)
    
    # lens f1
    lens_f1 = clsThinLens("lens 1", myCavity)        
    lens_f1.f = f1      
    
    # center mirror
    center_mirror = clsMirror("center mirror", myCavity)    
    center_mirror.R = R_center
    
    # f2 propagation
    prop_f2 = clsPropagation("f2 propagation", myCavity)    
    prop_f2.set_params(f2, 1)    
    
    # lens f2
    lens_f2 = clsThinLens("lens 2", myCavity)    
    lens_f2.f = f2                    
    
    # propagation between second lens and absorber 
    prop_lens_absorb = clsPropagation("propagation lens<>absorber", myCavity)    
    prop_lens_absorb.set_params(f2-pos_absorb-d_absorb/nr, 1)                   
    
    # absorber
    absorber = clsPropagation("absorber", myCavity)    
    absorber.set_params(d_absorb, nr)    
    absorber.set_ni_based_on_T(T_abs)            
    
    # propagation between absorber and right mirror
    prop_absorb_mirr = clsPropagation("propagation absorber<>mirror2", myCavity)    
    prop_absorb_mirr.set_params(pos_absorb + l2_corr, 1)     
    
    # right mirror
    mirror2 = clsMirror("right mirror", myCavity)    
    mirror2.R = R_right
    
    # --- ASSEMBLE CAVITY -----------------------------------------------------
    myCavity.add_component(mirror1)             # 0 ******
    myCavity.add_component(prop_f1)             # 1
    myCavity.add_component(lens_f1)             # 2
    myCavity.add_component(prop_f1)             # 3
    myCavity.add_component(prop_f1)             # 4
    myCavity.add_component(lens_f1)             # 5
    myCavity.add_component(prop_f1)             # 6
    myCavity.add_component(center_mirror)       # 7 ******
    myCavity.add_component(prop_f1)             # 8
    myCavity.add_component(lens_f1)             # 9
    myCavity.add_component(prop_f1)             # 10
    myCavity.add_component(prop_f2)             # 11
    myCavity.add_component(lens_f2)             # 12
    myCavity.add_component(prop_lens_absorb)    # 13
    myCavity.add_component(absorber)            # 14
    myCavity.add_component(prop_absorb_mirr)    # 15
    myCavity.add_component(mirror2)             # 16 *****
    
    # --- INPUT: RANDOM SPECKLE FIELD -----------------------------------------
    inp = clsSpeckleField(myCavity.grid)    
    inp.create_field_eq_distr(100, 20, 0.6*myCavity.grid.length_fov, 0)
    inp.name = "Input Field"    
    inp.plot_field(5)
    myCavity.incident_field_left = inp
    
    # --- CALCULATE LEFT OUTPUT AND BULK FIELD FOR 3 DIFFERENT WAVELENGHTS ----     
    lambda_vec = get_sample_vec(3, lambda_period/120, 0, False, 1)
    for index, dLambda in enumerate(lambda_vec):
        # SET CURRENT WAVELENGTH
       dL_pm = dLambda * 10**12 # delta lambda in pm
       print("")
       print(f"***   delta lambda = {dL_pm:.3f} pm   ***")
       myCavity.Lambda = lambda_c + dLambda                   
       
       # OUTPUT FIELD LEFT
       out = myCavity.output_field_left
       out.name = f"LEFT OUTPUT dL = {dL_pm:.3f} pm"
       out.plot_field(5, vmax_limit=0.001, c_map = "custom")
       
       # BULK FIELD LEFT OF CENTRAL MIRROR
       myCavity.calc_bulk_field_from_left(6)
       bulk = myCavity.bulk_field
       bulk.name = f"BULK, DL = {dL_pm:.3f} pm"
       bulk.plot_field(5)
       
       # CALCULATE REFLECTION COEFFICIENT OVER FOV AND SAVE RESULT TO FILE
       R = out.intensity_integral_fov() / inp.intensity_integral_fov()
       print(f"Reflectivity: {R}")
       data = [index, dL_pm, R]
       myCavity.write_to_file("result.csv",data)
    