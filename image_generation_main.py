import random
from pathlib import PurePath, Path
import cv2
import shutil

# Set these for the output and input directory
OUTPUT_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\segmentation'
INPUT_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\train'

CHANCE_TEST: float = 0.73
CHANCE_VALIDATION: float = 0.20


SHOULD_OVERWRITE: bool = True
RANDOM: random.Random = random.Random()


def create_directories(sub_directory_name: PurePath, overwrite=False):
    if not Path(sub_directory_name).exists():
        Path(sub_directory_name).mkdir(exist_ok=overwrite)
    elif overwrite:
        print("Removing overwritable directory")
        shutil.rmtree(str(sub_directory_name))
        create_directories(sub_directory_name,  overwrite)

if __name__ == '__main__':
    # Ensure no more accidental use
    return
    assert CHANCE_TEST + CHANCE_VALIDATION < 1.0, \
        "The chance of  a test plus  validation has to be less than 1."
    print(OUTPUT_DIRECTORY_LOCATION)
    root_output_directory: PurePath = PurePath(OUTPUT_DIRECTORY_LOCATION)
    print(root_output_directory)
    train_directory_output: PurePath = PurePath.joinpath(root_output_directory, "train")
    test_directory_output: PurePath = PurePath.joinpath(root_output_directory, "test")
    validation__directory_output: PurePath = PurePath.joinpath(root_output_directory, "validation")
    create_directories(train_directory_output, SHOULD_OVERWRITE)
    create_directories(test_directory_output, SHOULD_OVERWRITE)
    create_directories(validation__directory_output, SHOULD_OVERWRITE)

    total_number_of_samples: int = 0
    number_of_train_samples: int = 0
    number_of_test_samples: int = 0
    number_of_validation_samples: int = 0

    # Obtain the images from their directories
    root_input_directory: PurePath = PurePath(INPUT_DIRECTORY_LOCATION)
    sub_item: Path
    for sub_item in Path(root_input_directory).iterdir():
        # Just for error checking
        if sub_item.is_dir():
            print("Processing directory: {}".format(str(sub_item)))
            image_path: Path
            for image_path in sub_item.iterdir():
                is_validation_sample: bool = False
                is_testing_sample: bool = False

                # Determine if this will be a training, test, or validation image
                random_value: float = RANDOM.uniform(0.0, 1.0)
                if random_value <= CHANCE_TEST:
                    is_testing_sample = True
                elif random_value <= CHANCE_TEST + CHANCE_VALIDATION:
                    is_validation_sample = True

                # Load the image resized to be a power of 2 otherwise the u-net will stuff up
                image_original = cv2.resize(cv2.imread(str(image_path)), (768, 768))

                # Create the gray-scale and the rgb image
                image_grayscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
                image_rgb = image_original

                output_path: PurePath
                if is_validation_sample:
                    output_path = validation__directory_output
                    number_of_validation_samples += 1
                elif is_testing_sample:
                    output_path = test_directory_output
                    number_of_test_samples += 1
                else:
                    output_path = train_directory_output
                    number_of_train_samples += 1
                total_number_of_samples += 1

                # Generate the fish and mask directories
                output_fish_path = PurePath().joinpath(output_path, "Fish")
                output_mask_path = PurePath().joinpath(output_path, "Mask")

                # Check the directory exists
                create_directories(output_fish_path)
                create_directories(output_mask_path)

                # Write the images to the directories
                cv2.imwrite(str(PurePath.joinpath(output_fish_path,
                                              sub_item.name + "_" + image_path.name[:-4] + ".tif")),
                            image_rgb)
                cv2.imwrite(str(PurePath.joinpath(output_mask_path,
                                              sub_item.name + "_" + image_path.name[:-4] + ".tif")),
                            image_rgb)
                print(PurePath.joinpath(output_fish_path, sub_item.name + "_" + image_path.name[:-4] + ".tif"))

        else:
            print("That  was  a file, which is not intended to be in the root directory")

    print("Total number of images moved: {}".format(total_number_of_samples))
    print("Total number of training samples: {}\t{}".format(number_of_train_samples,
                                                            number_of_train_samples/total_number_of_samples))
    print("Total number of testing samples: {}\t{}".format(number_of_test_samples,
                                                           number_of_test_samples / total_number_of_samples))
    print("Total number of validation samples: {}\t{}".format(number_of_validation_samples,
                                                              number_of_validation_samples / total_number_of_samples))

    # Make a train, test, validation folder (if they don't exist) --- Done
    # Randomly select some of the images for test some of them for training and some for validation
    # Print the number of these
    # Go through and reshape images---Done
    # Convert them  all to black and white images. ---Done
    # Convert them all to tif images ---Done
    # Save image ---Done
