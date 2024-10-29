# Data-driven site characterization - Focus on small-strain stiffness

This repository contains the code used in the conference paper presented at the [7th International Conference on Geotechnical and Geophysical Site Characterization](https://isc7.cimne.com/). The paper details the training of machine learning models to obtain shear wave velocity estimates based on in-situ tests.

The conference paper can be found here: [Data-driven site characterization - Focus on small-strain stiffness - Felić et al. (2024)](https://www.scipedia.com/public/Felic*_et_al_2024a) - DOI: [10.23967/isc.2024.148](https://www.scipedia.com/public/Felic*_et_al_2024a)

## Folder structure

```
DataDriven
├── data                                  - data
├── graphics                              - saved graphics from running scripts in src
├── src                                   - folder that contains the python script files
│   ├── main.py                           - main script for now
├── environment.txt                       - dependency file to use with python
├── LICENSE                               - Github license file to specify the license of the repository 
├── README.md                             - repository description
```

## Requirements

The environment is set up using `python`.

To achieve this, create an environment named `venv` and label it as DataDriven (or any other desired name).
```bash
C:\Users\haris\Documents\GitHub\ISC7_DataDrivenSiteCharacterization>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv DataDriven
```

Activate the new environment with:
```bash
DataDriven\Scripts\activate
```

Then, install all packages using 'environment.txt'. If you encounter pip errors, install the libraries manually, for example:
```bash
(venv) C:\Users\haris\Documents\GitHub\ISC7_DataDrivenSiteCharacterization>py -m pip install -r environment.txt
```

## Database for Machine Learning
The database is accessible on the website of the [Computational Geotechnics Group (Graz University of Technology)](https://www.tugraz.at/fileadmin/user_upload/Institute/IBG/Datenbank/Database_CPT_PremstallerGeotechnik.zip). A description of the database itself can be found in the paper by Oberhollenzer et al. (2021) - DOI: https://doi.org/10.1016/j.dib.2020.106618
