![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# multitask_impute
This repository contains the supplementary code for the paper 'Deep Learning Imputation for Multi Task Learning' (authors Sophie Peacock, Etai Jacob, Nikolay Burlutskiy). Please get in touch if you would like to make any contributions to this work.

The workflow can be split into three stages. For each stage a .sh file can be found:
1. Preprocessing data and training imputation models
2. Using trained imputation models to predict missing values 
3. Using the imputed data in multitask models

## Preprocessing data and training imputation models
All code for this stage is contained in `multitask_impute/TDimpute` and scripts to run this code are in the root directory of the repository. `TDimpute_for_omiembed.py` preprocesses the three datasets (for each modality) and trains a deep imputation model to impute one modality from another. This step requires lists as an input which contain information regarding which patients have data for each combination of modalities.

## Using trained imputation models to predict missing values
This stage is done using `TDimpute_omiembed_predict.py` which loads trained models from the previous stage and predicts values for patients who do not have the target modality. Once missing data values have been predicted, `merge_imputed_data.py` should be run to merge the original data with the imputed values and obtain a complete dataset. This is the dataset which will be used in multitask experiments.

## Using the imputed data in multitask models
The code required for this stage of the pipeline can be found in `multitask_impute/OmiEmbed` and can be run using `multitask_impute/OmiEmbed/run_gpu.sh`. Data should be saved in a folder called 'data'. The data for the three modalities should be saved as 'A', 'B' and 'C', and target values labelled as 'labels' for a classification task, 'values' for a regression task, and 'survival' for a survival analysis task. 
