# EEG Biometric Authentication

This project is for the summer 2024 block course Decoding Neuronal Activity at the University of Osnabrück. 

## Installation
Create a conda environment and install the dependencies using the included `requirements.txt` file.

```
conda create --name <env_name> --file requirements.txt
```

## Data

Download the EEG dataset from this page:

https://physionet.org/content/auditory-eeg/1.0.0/

The [zip file](https://physionet.org/static/published-projects/auditory-eeg/auditory-evoked-potential-eeg-biometric-dataset-1.0.0.zip) appears to have all of the data, whereas when I tried downloading data using the `wget` command listed on the page, I was not able to get the filtered data.

Unzip the data so that the code is at the same level as the folder containing the entire dataset. So your directory should look something like:

```
./data.py
./network.py
./auditory-evoked-potential-eeg-biometric-dataset-1.0.0
...
```

## Usage

This project was meant to be run as an IPython notebook. Run all of the cells in `project.ipynb`. Note that the last cell which runs training over twenty models may take a while. It took 117 minutes on my Macbook Air M3.

## Contact

If you have any questions or comments please send them to tho@uni-osnabrueck.de.