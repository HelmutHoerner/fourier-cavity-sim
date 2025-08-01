# fo_cavity_sim

**Fourier Optics Cavity Simulation Library**

This Python package provides a software library for simulating optical cavities (linear cavities and ring cavities) using Fourier optics methods. 

---

## Documentation

📄 [User Manual (PDF)](docs/fo_cavity_sim_user_manual.pdf)

The user manual contains:
- Installation Instructions
- Overview of cavity types and configuration
- Description of key methods and parameters
- Example workflows
- Complete Reference: Classes and Methods 

## Installation

### Requirements

This package was tested in a Conda environment under MS Windows 10. You can manually create the environment like this:

```bash
conda create --name fourier-cavity-sim python=3.8.5
conda activate fourier-cavity-sim
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install numpy blas=*=openblas
conda install spyder matplotlib scipy portalocker joblib pandas
```
Alternatively, `pip install -e . --config-settings editable_mode=compat` will automatically install the required packages. However, if you're using Anaconda, I recommend creating the environment with the above Conda commands to ensure compatibility and performance. 

Note: The command `conda install numpy blas=*=openblas` is particularly recommended if you're running the package on an AMD CPU, as the default NumPy builds are optimized for Intel and may run significantly slower on AMD systems.

###  Recommended 

Clone the repository and install with pip (requires Python ≥3.8):

```bash
git clone https://github.com/HelmutHoerner/fourier-cavity-sim.git
cd fourier-cavity-sim
pip install -e . --config-settings editable_mode=compat
```

### Please cite
If you use **fo_cavity_sim** in academic work, please cite the user manual:

> H. Hoerner, *fo_cavity_sim User Manual*, 2025.  
> https://github.com/HelmutHoerner/fourier-cavity-sim/blob/main/docs/fo_cavity_sim_user_manual.pdf

**BibTeX**
```bibtex
@manual{hoerner2025focavitysim,
  title   = {fourier-cavity-sim: A Python Library for Fourier-Optics Cavity Simulations (User Manual)},
  author  = {Hoerner, Helmut},
  year    = {2025},
  url     = {https://github.com/HelmutHoerner/fourier-cavity-sim/blob/main/docs/fo_cavity_sim_user_manual.pdf},
  note    = {Version 1.0}
