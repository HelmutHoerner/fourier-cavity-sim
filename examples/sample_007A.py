import math
from fo_cavity_sim import clsCavity1path, clsMirror, clsPropagation, \
    clsThinLens, clsSpeckleField, get_sample_vec, clsTaskManager

if __name__ == '__main__':    
    # --- SIMULATION PARAMETERS -----------------------------------------------
    R_left = 0.7                         # right mirrror
    R_right = 0.999                      # left mirror
    R_center =  4*R_left/((1+R_left)**2) # center mirror
    d_absorb = 0.0006                    # thickness of absorber
    T_abs = math.sqrt(R_left/R_right)    # optimal absorption
    pos_absorb = 0.005                   # distance absorber after left mirror
    nr = 1.5                             # absorber's refractive index
    f1 = 0.025                           # focal length first lens  
    f2 = f1 - d_absorb/2*(nr-1/nr)       # focal length second lens  
    length_fov =  0.00081                # field-of-view sidelength
    folder = "cavity_folder"
    tmp_folder = "tmp_folder"
    
    # --- CAVITY SUB-CLASS CONTAINS FINAL SIMULATION STEP ---------------------
    class clsMyCavity(clsCavity1path):
        def __init__(self, name):
            super().__init__(name)
        
        def additional_step(self, step: int):
            """ 
            Final steps after calulating M_bmat 
            """      
            # --- INPUT: RANDOM SPECKLE FIELD ---------------------------------
            self.progress.print("Generating incident field")
            inp = clsSpeckleField(myCavity.grid)    
            inp.create_field_eq_distr(100, 20, 0.6*myCavity.grid.length_fov, 0)
            inp.name = "Input Field"    
            self.incident_field_left = inp
            
            # --- SAVE INCIDENT FIELD (ONLY ONCE) -----------------------------
            if self.UID==0:
                file_name = self._full_file_name("in", file_extension = "png")
                inp.plot_field(5, True, file_name)
            
            # --- GET AND SAVE OUTPUT FIELD -----------------------------------    
            self.progress.print("")
            self.progress.print("applying reflection matrix to incident field")
            out = self.output_field_left
            file_name = self._full_file_name("out", file_extension = "png")
            out.plot_field(5, True, file_name, c_map="custom") 
            
            # --- CALCULATE AND SAVE REFLECTIVITY -----------------------------
            R = out.intensity_integral_fov() / inp.intensity_integral_fov()
            self.progress.print(f"Reflectivity: {R}")
            data = [self.UID, self.Lambda_nm, 
                    1000*(self.Lambda_nm-self.Lambda_ref_nm), R]
            self.write_to_file("result.csv",data)
            
            # --- DELETE TMP FILES --------------------------------------------
            self.M_bmat_tot.keep_tmp_files = False
            self.M_bmat_tot.clear()
            self.M_bmat_tot = None
            self.progress.pop()
            
    # --- CREATE CAVITY -------------------------------------------------------
    myCavity = clsMyCavity("8f cavity")
    myCavity.additional_steps=1
    myCavity.allow_temp_file_caching = True
    myCavity.use_swap_files_in_bmatrix_class = True
    myCavity.folder = folder
    myCavity.tmp_folder = tmp_folder
    myCavity.Lambda_nm = 633    
    
    # --- DETERMINE RESONANCE WAVELENGTH, CREATE GRID -------------------------
    lambda_c, k_c, n_c, lambda_period, l2_corr = \
        myCavity.resonance_data_8f_cavity(R_left, R_center, R_right, f1)
    l2_corr *= 1.212
    myCavity.Lambda = lambda_c        
    myCavity.grid.set_opt_res_based_on_sidelength(length_fov, 2, f1, True)
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

    #########################################################################
    # Main Program
    #########################################################################
    myCavity.Lambda_ref = lambda_c
    points_to_simulate = 17
    lambda_vec = get_sample_vec(points_to_simulate, lambda_period/25, 0, True, 1)
    steps = myCavity.total_steps
    task_manager = clsTaskManager(points_to_simulate, steps, tmp_folder)
    task_manager.sleep_time = 600
    
    sim, step = task_manager.get_next_task()
    if not sim == -1: 
        dLambda = lambda_vec[sim]
        dL_pm = dLambda * 1e12
        print("")
        print("****************************************************")
        print(f"UID = {sim}, step = {step}, delta lambda = {dL_pm:.4f} pm")
        print("****************************************************")
        myCavity.Lambda = lambda_c + dLambda
        myCavity.UID = sim
        myCavity.single_step(step)
        task_manager.end_task(sim, step)   

    
    