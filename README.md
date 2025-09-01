# A DATA-DRIVEN APPROACH FOR SOIL PARAMETER DETERMINATION USING SUPERVISED MACHINE LEARNING

This repository contains the code used in the conference paper presented at the [9th International Symposium for Geotechnical Safety and Risk (ISGSR)](https://www.isgsr2025.com). The paper details the training of machine learning models to obtain soil property estimates based on in-situ tests.

The conference paper can be found here: [A DATA-DRIVEN APPROACH FOR SOIL PARAMETER DETERMINATION USING SUPERVISED MACHINE LEARNING]([xxxx](https://www.researchgate.net/publication/395129478_A_data-driven_approach_for_soil_parameter_determination_using_supervised_machine_learning)) - DOI: [10.3850/GRF-25280825_isgsr-156-P210-cd](10.3850/GRF-25280825_isgsr-156-P210-cd)

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
C:\Users\haris\Documents\GitHub\ISGSR25_DataDrivenSiteCharacterization>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv DataDriven
```
```bash
C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv DataDriven
```

Activate the new environment with:
```bash
DataDriven\Scripts\activate
```

Then, install all packages using 'environment.txt'. If you encounter pip errors, install the libraries manually, for example:
```bash
(venv) C:\Users\haris\Documents\GitHub\ISGSR25_DataDrivenSiteCharacterization>py -m pip install -r environment.txt
```
```bash
py -m pip install -r environment.txt
```
