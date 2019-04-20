from os.path import dirname
from pathlib import PurePath

import pytest


# Contains configurations to the tests

@pytest.fixture(scope="module")
def path_to_images() -> PurePath:
    # Obtain the location to the test images
    path: PurePath = PurePath(dirname(__file__))
    return path.parents[1].joinpath("tests").joinpath("resources").joinpath("images")
