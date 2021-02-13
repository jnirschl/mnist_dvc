#   -*- coding: utf-8 -*-
#  Copyright (c)  2021.  Jeffrey Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE file in the project
#   root directory for  license information.
#
#   Time-stamp: <>
#   ======================================================================

import pytest
import requests

@pytest.fixture
def competition():
    return "digit-recognizer"

def test_url(competition):
    """Test if competition url is valid"""
    kaggle_url = f"https://www.kaggle.com/c/{competition}"
    web = requests.get(kaggle_url)
    assert web.status_code == 200
