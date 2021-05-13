#   -*- coding: utf-8 -*-
#  Copyright (c)  2021.  Jeffrey Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE file in the project
#   root directory for  license information.
#
#   Time-stamp: <>
#   ======================================================================


import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from keras.models import Sequential
# Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator


from src.data import load_data, load_params
from src.models.cnn import simple_mnist
from src.models.metrics import gmpr_score

# Specify opencv optimization
cv2.setUseOptimized(True)


def main(mapfile_path, cv_idx_path,
         results_dir, model_dir,
         image_size=(28, 28, 1),
         batch_size=32,
         shuffle=False):
    """Train model and predict digits"""
    results_dir = Path(results_dir).resolve()
    model_dir = Path(model_dir).resolve()

    assert (os.path.isdir(results_dir)), NotADirectoryError
    assert (os.path.isdir(model_dir)), NotADirectoryError

    # read files
    mapfile_df, cv_idx = load_data([mapfile_path, cv_idx_path],
                                   sep=",", header=0,
                                   index_col=0, )
    # load params
    params = load_params()
    classifier = params["classifier"]
    # target_class = params["train_test_split"]["target_class"]
    model_params = params["model_params"]["cnn"]
    random_seed = params["random_seed"]

    # label column must be string
    mapfile_df["label"] = mapfile_df["label"].astype('str')

    # get train and dev indices
    train_idx = cv_idx[cv_idx["fold_01"] == "train"].index.tolist()
    dev_idx = cv_idx[cv_idx["fold_01"] == "test"].index.tolist()
    train_df = mapfile_df.iloc[train_idx]
    dev_df = mapfile_df.iloc[dev_idx]

    # create train/dev data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # preprocessing_function
    train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                        x_col='filenames', y_col='label',
                                                        weight_col=None, target_size=image_size[0:2],
                                                        color_mode='grayscale', classes=None,
                                                        class_mode='categorical', batch_size=batch_size,
                                                        shuffle=shuffle, seed=random_seed,
                                                        interpolation='nearest',
                                                        validate_filenames=True)

    dev_datagen = ImageDataGenerator(rescale=1. / 255)
    dev_generator = dev_datagen.flow_from_dataframe(dataframe=dev_df, rescale=1. / 255,
                                                    x_col='filenames', y_col='label',
                                                    weight_col=None, target_size=image_size[0:2],
                                                    color_mode='grayscale', classes=None,
                                                    class_mode='categorical', batch_size=batch_size,
                                                    shuffle=shuffle, seed=random_seed,
                                                    interpolation='nearest',
                                                    validate_filenames=True)

    # create model
    if classifier.lower() == "simple_mnist":
        # simple mnist parameters
        base_filter = 32
        fc_width = 512

        model = simple_mnist(base_filter, fc_width,
                             dropout_rate=model_params["dropout_rate"],
                             learn_rate=model_params["learn_rate"],
                             image_size=image_size)
    else:
        raise NotImplementedError

    # callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                     patience=5, min_lr=0.0001)

    print(model_params["epochs"])
    history = model.fit(train_generator,
                        epochs=model_params["epochs"],
                        verbose=1,
                        shuffle=True,
                        callbacks=[reduce_lr],
                        validation_data=dev_generator)

    # set model scoring metrics


    # # TODO - add custom metric for GMPR
    # scoring = {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy',
    #            'f1': 'f1',
    #            "gmpr": make_scorer(gmpr_score, greater_is_better=True),
    #            'jaccard': 'jaccard', 'precision': 'precision',
    #            'recall': 'recall', 'roc_auc': 'roc_auc'}

    # train using cross validation
    # cv_output = cross_validate(model, train_feats.to_numpy(),
    #                            train_labels.to_numpy(),
    #                            cv=split_generator,
    #                            fit_params=None,
    #                            scoring=scoring,
    #                            return_estimator=True)
    #
    # # get cv estimators
    # cv_estimators = cv_output.pop('estimator')
    # cv_metrics = pd.DataFrame(cv_output)
    #
    # # rename columns
    # col_mapper = dict(zip(cv_metrics.columns,
    #                       [elem.replace('test_', '') for elem in cv_metrics.columns]))
    # cv_metrics = cv_metrics.rename(columns=col_mapper)
    #
    # # save cv estimators as pickle file
    # with open(model_dir.joinpath("estimator.pkl"), "wb") as file:
    #     pickle.dump(cv_estimators, file)

    # save training history
    logs_df = pd.DataFrame(data=history.history,
                           index=range(1, model_params["epochs"]+1))
    logs_df.index.name = "epochs"
    logs_df.to_csv(Path("./reports/figures/logs.csv").resolve())

    # # save metrics
    # metrics = json.dumps(dict(cv_metrics.mean()))
    # with open(results_dir.joinpath("metrics.json"), "w") as writer:
    #     writer.writelines(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mf", "--mapfile", dest="mapfile",
                        required=True, help="CSV file with image filepath and label")
    parser.add_argument("-cv", "--cvindex", dest="cv_index",
                        default=Path("data/processed/split_train_dev.csv").resolve(),
                        required=False, help="CSV file with train/dev split")
    parser.add_argument("-rd", "--results-dir", dest="results_dir",
                        default=Path("./results").resolve(),
                        required=False, help="Metrics output directory")
    parser.add_argument("-md", "--model-dir", dest="model_dir",
                        default=Path("./models").resolve(),
                        required=False, help="Model output directory")
    args = parser.parse_args()

    # train model
    main(Path(args.mapfile).resolve(),
         Path(args.cv_index).resolve(),
         Path(args.results_dir).resolve(),
         Path(args.model_dir).resolve())
