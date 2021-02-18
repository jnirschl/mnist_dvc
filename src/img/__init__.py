#   -*- coding: utf-8 -*-
#  Copyright (c)  2021.  Jeffrey Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE file in the project
#   root directory for  license information.
#
#   Time-stamp: <>
#   ======================================================================

# Standard library imports

# Image and array operations
import cv2

# Specify opencv optimization
cv2.setUseOptimized(True)


def to_float(img):
    return cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
