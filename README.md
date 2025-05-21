# fo_cavity_sim

**Fourier Optics Cavity Simulation Library**

This Python package provides a software library for simulating optical cavities (linear cavities and ring cavities) using Fourier optics methods. 

---

## Installation

###  Recommended (editable install for development)

Clone the repository and install with pip (requires Python ≥3.8):

```bash
git clone https://github.com/HelmutHoerner/fourier-cavity-sim.git
cd fourier-cavity-sim
pip install -e . --config-settings editable_mode=compat
```

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
