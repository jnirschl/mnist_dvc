#   -*- coding: utf-8 -*-
#  Copyright (c)  2021. Jeffrey Nirschl. All rights reserved.
#
#  Licensed under the MIT license. See the LICENSE file in the project
#  root directory for  license information.
#
#  Time-stamp: <>
#   ======================================================================

import argparse
import os
import numpy as np
import pandas as pd

from pathlib import Path
import cv2

from src.img import transforms


def main(data_path, ext="png",
         img_shape=(28,28,1),
         output="./data/interim/",
         prefix="",
         na_rep="nan"):
    """Accept a numpy array of flattened images and
     save as images."""

    # create output director, if needed
    output = Path(output).resolve().joinpath(prefix)
    if not os.path.isdir(output):
        os.mkdir(output)

    # check for errors
    assert os.path.isfile(data_path), FileNotFoundError
    assert os.path.isdir(output), NotADirectoryError

    # remove period from ext
    ext = ext.replace(".", "")

    # read file
    img_array = pd.read_csv(data_path, sep=",",
                            header=0)

    # pop target column and save
    if "label" in img_array.columns:
        target = pd.DataFrame(img_array.pop("label"))
    else:
        target = pd.DataFrame({"label":np.full_like(img_array[img_array.columns[0]],
                                                    np.nan, dtype=np.float32)})

    # create mean image
    mean_image = transforms.mean_image(img_array)
    cv2.imwrite(str(output.joinpath("mean_image.png")), mean_image)

    # save individual images
    filenames = save_images(img_array, target, img_shape, ext, output)

    # save a mapfile with the filename and label
    mapfile = pd.DataFrame({"filenames":filenames,
                            target.columns[0]:target[target.columns[0]]},
                           index=target.index)
    mapfile.to_csv(output.joinpath(f"{prefix}_mapfile.csv"),
                   na_rep=na_rep)


def save_images(img_array, target, img_shape, ext, output):
    """Subfunction to process flattened images in dataframe"""
    # process dataframe line by line
    filenames = []
    for idx in range(img_array.shape[0]):
        if (idx+1) % 10000 == 0:
            print(f"Processed {idx+1} images")
        # select image and reshape
        temp_img = np.reshape(img_array.iloc[idx].to_numpy(),
                              img_shape).astype(np.float32)

        # set filename
        temp_name = f"{idx:06d}_{target.iloc[idx].to_numpy()[0]}.{ext}"

        filenames.append(str(output.joinpath(temp_name)))

        if not cv2.imwrite(filenames[idx], temp_img):
            raise SystemError

    print(f"Processed {idx + 1} images")
    return filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_data", dest="train_data",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test_data", dest="test_data",
                        required=True, help="Test CSV file")
    parser.add_argument("-ex", "--ext", dest="ext",
                        default=".png",
                        required=False, help="Train CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/interim").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # categorical variables into integer codes
    main(args.train_data, prefix="train", ext=args.ext, output=args.output_dir)

    main(args.test_data, prefix="test", ext=args.ext, output=args.output_dir)
