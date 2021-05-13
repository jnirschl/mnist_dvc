# MNIST DVC

![GitHub](https://img.shields.io/github/license/jnirschl/mnist_dvc)
==============================

## Project Goals
Classify digits from images using the MNIST dataset

## Introduction

This repository uses [Data Version Control (DVC)](https://dvc.org/) to create a machine learning pipeline and track
experiments for the Kaggle competition digit-recognizer. We will use a modified version of
the [Team Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
as our Data Science Life cycle template. This repository template is based on
the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science).

In order to start, clone this repository and install [DataVersionControl](https://dvc.org/). Follow the instructions
below to proceed through the data science life cycle using DVC to manage parameters, scripts, artifacts, and metrics.

### 1. Domain understanding/problem definition

The first step any data science life cycle is to define the question and to understand the problem domain and prior
knowledge. Given a well-formulated question, the team can specify the goal of the machine learning application (e.g.,
regression, classification, clustering, outlier detection) and how it will be measured, and which data sources will be
needed. The scope of the project, key personnel, key milestones, and general project architecture/overview is specified
in the Project Charter and iterated throughout the life of the project. A list of data sources which are available or
need to be collected is specified in the table of data sources. Finally, the existing data is summarized in a data
dictionary that describes the features, number of elements, non-null data, data type (e.g., nominal, ordinal,
continuous), data range, as well as a table with key descriptive summary statistics.

*Deliverables Step 1:*
1. Project charter
2. Table of data sources
3. Data dictionary
4. Summary table of raw dataset

#### Downloading the dataset

The script [download.py](src/data/download.py) will download the dataset from Kaggle. The key artifacts of this 
DVC stage are the [raw training and testing datasets](data/raw). 

In your terminal, use the command-line interface to build the first stage of the pipeline.

Download data from Kaggle
``` bash
dvc run -n download_data -p competition \
    -d src/data/download.py \
    -o data/raw/train.csv \
    -o data/raw/test.csv \
    --desc "Download data from Kaggle"\
    python3 src/data/download.py -tr train.csv -te test.csv -o "./data/raw"
```

Reshape flattened images in CSV to two-dimensional images, compute mean image, and create a file mapping
the image filepath with the ground truth label.  
``` bash
dvc run -n prepare_images \
    -d src/data/prepare_img.py -d src/img/transforms.py \
    -d data/raw/train.csv      -d data/raw/test.csv \
    -o data/processed/train/ \
    -o data/processed/train_mapfile.csv \
    -o data/processed/train_mean_image.png \
    -o data/processed/test/ \
    -o data/processed/test_mapfile.csv \
    -o data/processed/test_mean_image.png \
    --desc "Create images from numpy array"\
    python3 src/data/prepare_img.py -tr data/raw/train.csv -te data/raw/test.csv -o "./data/processed/"
```

### 2. Data acquisition and understanding

The second step involves acquiring and exploring the data to determine the quality of the data and prepare the data for
machine learning models. This step involves exploring and cleaning the data to account for missing or corrupted data as
well as validating that data meet specified validation rules to ensure there were no errors in data collection or data
entry.

*Deliverables Step 2:*
1. Data quality report
2. Proposed data pipeline/architecture
3. Checkpoint decision

TODO- Validation rules: Images must be (28,28,1) UINT8 images saved as 'png' or 'jpeg'. 

### 3. Modeling

##### Split data into the train, dev, and test sets

+ Data split report

```bash
dvc run -n split_train_dev -p random_seed,train_test_split \
    -d src/data/split_train_dev.py \
    -d data/processed/train_mapfile.csv \
    -o data/processed/split_train_dev.csv \
    --desc "Split training data into the train and dev sets using stratified K-fold cross validation." \
    python3 src/data/split_train_dev.py -tr data/processed/train_mapfile.csv  -o data/processed/
```

#### Model training

``` bash
dvc run -n train_model -p classifier,model_params,random_seed \
    -d src/models/train_model.py \
    -d data/processed/train_mapfile.csv \
    -d data/processed/split_train_dev.csv \
    --plots reports/figures/logs.csv \
    --desc "Train the specified classifier using the pre-allocated stratified K-fold cross validation splits and the current params.yaml settings." \
    python3 src/models/train_model.py -mf data/processed/train_mapfile.csv -cv data/processed/split_train_dev.csv
```

#### Predict output

``` bash
dvc run -n predict_output -p predict,train_test_split.target_class \
    -d src/models/predict.py \
    -d src/models/metrics.py \
    -d models/estimator.pkl \
    -d data/processed/test_processed.csv \
    -o results/test_predict_proba.csv \
    -o results/test_predict_binary.csv \
    --desc "Predict output on held-out test set for submission to Kaggle." \
    python3 src/models/predict.py -te data/processed/test_processed.csv -rd results/ -md models/
```

### 4. Deployment

#### Status dashboard

+ Display system health
+ Final modeling report
+ Final solution architecture

### 5. Project conclusion

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
