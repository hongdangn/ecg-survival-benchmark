# ecg-survival-benchmark

This is a set of scripts for modeling patient mortality in support of https://arxiv.org/abs/2406.17002

In the paper, we predict all-cause mortality using deep learning from ECG using Code-15, MIMIC-IV, an internal dataset from Boston Children's Hospital, and models trained on all three at once. 

These scripts include data processing, model building, survival function fitting, and evaluating models to .csv files with metrics.

We use PyCox deep survival models (LH, MTLR, CoxPH (DeepSurv), and DeepHit) and also implement classifiers predicting mortality by a time horizon (e.g. --horizon 10.0), assuming censored patients survive, and then append those classifiers with Cox regressions.

Evaluations are based on Concordance, AUPRC, and AUROC. Time-censored measures are also generated.

## Quick Start

Build a singularity container from `Sing_Torch_05032024.def`

Minimal Code-15 data:

1. Go to       https://zenodo.org/records/4916206
1.  Download    `exams.csv`                                  to `./RAW_DATA/Code15/exams.csv`
1. Download    `exams_part0.hdf5`                           to `./RAW_DATA/Code15/TRAIN/exams_part0.hdf5`
1. Run         `./RAW_DATA/Code15_ECG_Process_8020.py`      with `os.getcwd()` as ./ 
1. Move        `./RAW_DATA/Code15_Processed`                to `./HDF5_DATA/Code15`
1. Open        `model_runner_caller.py`
1. Run         top section

You should see a model folder, populated with .pdfs, results, .csvs, and .hdf5s, appear in `./Trained Models/`

## Installation

We ran everything using a singularity container built from Sing_Torch_05032024.def

## Setting up Datasets

Code-15 (100GB RAM, ~2 hours)
1. Download data from       https://zenodo.org/records/4916206
1. Place all .hdf5 in       `./RAW_DATA/Code15/TRAIN/exams_part0…17.hdf5`
1. Place exams.csv in       `./RAW_DATA/Code15/exams.csv`
1. Run                      `./RAW_DATA/Code15_ECG_Process_8020.py`          with `os.getcwd()` as ./ 
1. Move                     `./RAW_DATA/Code15_Processed/…`                  to     `./HDF5_DATA/Code15`

MIMIC-IV (300GB RAM, ~8 hours)
1. From “hosp” in  "https://physionet.org/content/mimiciv/2.2/  
	1. Download `/patients.csv.gz`         to `./RAW_DATA/MIMIC IV/`
	1. Download `/admissions.csv.gz`       to `./RAW_DATA/MIMIC IV/`

1. From https://physionet.org/content/mimic-iv-ecg/1.0/ 
	1. Download `machine_measurements.csv`                   to `./RAW_DATA/MIMIC IV/`
	1. Download `machine_measurements_data_dictionary.csv`   to `./RAW_DATA/MIMIC IV/`
	1. Download `record_list.csv`                            to `./RAW_DATA/MIMIC IV/`
	1. Download `waveform_note-links.csv`                    to `./RAW_DATA/MIMIC IV/`
	1. Download “files”                                      to `./RAW_DATA/MIMIC IV/ECGs/p1000….`
1. Run 				                            `./RAW_DATA/MIMIC_IV_PreProcess.py`
1. Run                                                        `./RAW_DATA/MIMIC_IV_Process.py`          with `os.getcwd()` as `./RAW_DATA`
1. Move                                                        `./RAW_DATA/MIMICIV_Processed/…`          to `./HDF5_DATA/MIMICIV`



## Use Overview:

Classifier-based survival models are built and evaluated through `Model_Runner_SurvClass.py`.
PyCox survival models are built and evaluated through `Model_Runner_PyCox.py`.

To manually build/evaluate models, use `Model_Runner_Caller.py`, which is pre-set with minimal calls to train a model.

To do this with job files, structure the args to Model_Runner_SurvClass/PyCox similarly to Model_Runner_Caller does it (more on this later).

Once you have several trained models, you can summarize them to tables using `Analysis_Summarize_Trained_Models_Pandas_0325.py`
This summarizes model outputs into a summary table and analyzes the data.
Kaplan-Meier and histogram Survival Functions can be made by adapting Gen_KM_Curves.py to run on a particular training folder

We include Job_File_Maker_SurvClass and Job_File_Maker_PyCox - these generate job files in our institution's format, but can be adapted to yours.
Job_File_Maker_From_Summary.py can be adapted to add / remove / change model runner arguments (ex: switch the evaluation dataset)
In our paper, models were trained with random seeds 10-14. Data was split with random seed 12345.

---
Script descriptions:

Model_Runner_SurvClass/PyCox: These scripts train and evaluate models. Model_Runner_SurvClass is built for classifier-cox models, Model_Runner_PyCox is built for the deep survival models. All parameters are entered through a series of string-string argument-value pairs. Arguments must begin with "--" and be separated by spaces. These are turned into a dictionary mapping arguments to values, and are generally passed throughout the scripts.

GenericModelClassifierCox and GenericModelDeepSurvival are the two main model classes . Each extends GenericModel, which provides shared functions. Additional functions, which are likely to be needed in model training but which don't make sense as a model function, appear in Support_Functions.py 

The neural network architectures themselves appear as additional scripts (e.g. InceptionTimeClassifier.py). These have been adapted to return model features rather than outputs.

In general, the neural network has three "chunks". An ECG-interpreting chunk (InceptionTime or a ResNet developed by the Ribeiro group), a covariate-processing chunk (usually 3 linear-ReLu layers of dimension 32), and a fusion "chunk" (usually 3 128-dimensional linear-relu layers) heading into an output linear layer.

This format allows flexibility in adding new model types, since e.g. managing training is done by a more general class.
Models are evaluated after training, but a model can be evaluated without training by loading the highest-validation-metric model by skipping training with "--Epoch_End -1"

Since models are evaluated on-the-fly, a GPU is required for both training and evaluation.


## Example Job File:

See Job_File_Maker_PyCox.py 

## On Data formatting:

Labels are expected to have several column names: 
- Covariates are set by listing the column names in an array, such as '-- covariates [Age,Is_Male]' (with no spaces in the list)
- PID (patient ID, int64), Mort_Event(bool), Mort_TTE (float), SID (Study Id, int64) are required columns

We tried to make data processing flexible: inputs in the N-H (single-channel time series) or N-H-W (multi-channel time series) format are expanded to the image N-C-H-W format. Individual models then change the shape as needed (an image model stays in N-C-H-W, ECG models tend to prefer N-H-W).


## See Also

[ArXiv link here]

