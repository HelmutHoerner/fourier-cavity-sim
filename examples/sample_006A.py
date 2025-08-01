from fo_cavity_sim import (
    clsCavity2path, clsBeamSplitterMirror,
    clsPropagation, clsThinLens, clsTransmissionMixer,
    clsSpeckleField, clsTestImage, Side, Path
)

if __name__ == '__main__':
    # --- SIMULATION PARAMETERS -----------------------------------------------
    R_left = 0.7          # left mirror
    R_right = 0.999       # right mirror
    f = 0.05              # focal length of lenses
    length_fov = 0.002    # field-of-view width

    # --- GENERATE CAVITY -----------------------------------------------------
    myCavity = clsCavity2path("ring cavity")
    myCavity.folder = "cavity_folder"
    myCavity.tmp_folder = "G:\\tmp"
    myCavity.Lambda_nm = 633
    myCavity.use_swap_files_in_bmatrix_class = True

    # --- RESONANCE WAVELENGTH ------------------------------------------------
    lambda_c, k_c, n_c, lambda_period = \
        myCavity.resonance_data_simple_cavity(R_left, R_right, 4 * f)
    myCavity.Lambda = lambda_c

    # --- GENERATE GRID -------------------------------------------------------
    myCavity.grid.set_opt_res_based_on_sidelength(length_fov, 1.5, 2 * f, True)
    print(f"Field-of-view:  {myCavity.grid.res_fov}x{myCavity.grid.res_fov}")
    print(f"Total grid res: {myCavity.grid.res_tot}x{myCavity.grid.res_tot}")

    # --- GENERATE COMPONENTS -------------------------------------------------
    # left input coupling mirror
    mirror1 = clsBeamSplitterMirror("left mirror", myCavity)
    mirror1.R = R_left

    # f propagation
    prop_f = clsPropagation("f propagation", myCavity)
    prop_f.set_params(f, 1)

    # lens
    lens_f = clsThinLens("lens", myCavity)
    lens_f.f = f

    # top-right and bottom-left mirrors (no path mixing; pure "corner" mirrors)
    corner_mirrors = clsTransmissionMixer("corner mirrors", myCavity)
    corner_mirrors.T_same = 1
    corner_mirrors.refl_behavior_for_path_mixing = False

    # right mirror
    mirror2 = clsBeamSplitterMirror("right mirror", myCavity)
    mirror2.R = R_right

    # --- BUILD RING CAVITY ---------------------------------------------------
    myCavity.add_4port_component(mirror1)               # 0 left mirror
    myCavity.add_2port_component(prop_f, prop_f)        # 1
    myCavity.add_2port_component(lens_f, lens_f)        # 2
    myCavity.add_2port_component(prop_f, prop_f)        # 3
    myCavity.add_4port_component(corner_mirrors)        # 4 corner mirrors
    myCavity.add_2port_component(prop_f, prop_f)        # 5
    myCavity.add_2port_component(lens_f, lens_f)        # 6
    myCavity.add_2port_component(prop_f, prop_f)        # 7
    myCavity.add_4port_component(mirror2)               # 8 right mirror

    # --- INCIDENT FIELD PATH A (HORIZONTAL) ----------------------------------
    in_hor = clsTestImage(myCavity.grid)
    in_hor.create_test_image(3)
    in_hor.name = "Input (Horizontal)"
    in_hor.plot_field(5)
    myCavity.set_incident_field(Side.LEFT, Path.A, in_hor)

    # --- INCIDENT FIELD PATH B (VERTICAL) ------------------------------------
    in_ver = clsSpeckleField(myCavity.grid)
    in_ver.create_field(500, 0.6 * length_fov, 0)
    in_ver.name = "Input (Vertical)"
    in_ver.plot_field(5)
    myCavity.set_incident_field(Side.LEFT, Path.B, in_ver)

    # --- CALCULATE OUTPUT FIELDS LEFT SIDE -----------------------------------
    out_L_B = myCavity.get_output_field(Side.LEFT, Path.B)
    out_L_B.plot_field(5)
    out_L_A = myCavity.get_output_field(Side.LEFT, Path.A)
    out_L_A.plot_field(5)

    # --- FIELD INSIDE THE CAVITY RIGHT OF FIRST LENS -------------------------
    myCavity.calc_bulk_field_from_left(2)

    bulk_field_A_LTR = myCavity.get_bulk_field_LTR(Path.A)
    bulk_field_A_LTR.name = "Left-to-Right Bulk Field (Path A)"
    bulk_field_A_LTR.plot_field(5)

    bulk_field_A_RTL = myCavity.get_bulk_field_RTL(Path.A)
    bulk_field_A_RTL.name = "Right-to-Left Bulk Field (Path A)"
    bulk_field_A_RTL.plot_field(5)

    bulk_field_A = myCavity.get_bulk_field(Path.A)
    bulk_field_A.name = "Total Bulk Field (Path A)"
    bulk_field_A.plot_field(5)
