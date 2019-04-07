import numpy as np

from image_genarators.input_output_generators import generate_image_data


# noinspection PyUnusedLocal
def test_input_output_generators(path_to_images):
    """
    Test that the generator will loop over,
    returns the right number of files in the batch, and has the right shape.
    """
    # Generate and test the images
    generator = generate_image_data(1, str(path_to_images) + "\\", 2, "*")
    max_number_of_runs = 100

    number_of_runs = 0
    image: np.ndarray
    mask: np.ndarray
    for image, mask in generator:
        assert image.shape == (1, 768, 768, 3)
        assert mask.shape == (1, 768 * 768, 1)
        if number_of_runs == max_number_of_runs:
            break
        number_of_runs = number_of_runs + 1


# noinspection PyUnusedLocal
def test_input_output_generator_batch_size(path_to_images):
    """
    Test that the generator will loop over,
    returns the right number of files in the batch.
    """
    # Generate and test the images
    generator = generate_image_data(2, str(path_to_images) + "\\", 2, "*")
    max_number_of_runs = 100

    number_of_runs = 0
    image: np.ndarray
    mask: np.ndarray
    for image, mask in generator:
        assert image.shape == (2, 768, 768, 3)
        assert mask.shape == (2, 768, 768, 1)
        if number_of_runs == max_number_of_runs:
            break
        number_of_runs = number_of_runs + 1
