from fo_cavity_sim import clsPropagation, clsThinLens, \
                         clsTestImage, clsCavity1path, Dir                         

if __name__ == '__main__':
    length_fov = 0.0021 # field-of-view sidelength
    f = 0.075            # focal length of lenses
    
    # --- BUILD CAVITY -------------------------------------------------
    myCavity = clsCavity1path("my_cavity")
    myCavity.Lambda_nm = 633
    myCavity.grid.set_opt_res_tot_based_on_res_fov(length_fov, 100, 2*f)
    print(f"Field-of-view:  {myCavity.grid.res_fov}x{myCavity.grid.res_fov}")
    print(f"Total grid res: {myCavity.grid.res_tot}x{myCavity.grid.res_tot}")
    
    # --- DEFINE COMPONENTS -------------------------------------------
    prop = clsPropagation("f propagation", myCavity)   # distance f
    prop.set_params(f, 1)
    
    lens = clsThinLens("lens", myCavity)               # thin lens
    lens.lens_type_spherical = True
    lens.f = f    
    
    # --- ASSEMBLE 4f TELESCOPE ---------------------------------------
    myCavity.add_component(prop)  # 0 : free space f
    myCavity.add_component(lens)  # 1 : lens f
    myCavity.add_component(prop)  # 2 : free space f
    myCavity.add_component(prop)  # 3 : free space f
    myCavity.add_component(lens)  # 4 : lens f
    myCavity.add_component(prop)  # 5 : free space f
    
    # --- INPUT FIELD --------------------------------------------------
    input_field = clsTestImage(myCavity.grid)
    input_field.create_test_image(3)   # “T’’ pattern
    input_field.name = "Input Field"
    input_field.plot_field(5)
    
    # --- PROPAGATION --------------------------------------------------
    output_field = myCavity.prop(input_field, 0, 5, Dir.LTR)
    output_field.name = "Output Field"
    output_field.plot_field(5)