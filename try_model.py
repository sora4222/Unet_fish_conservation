from tensorflow.python import keras
import numpy as np
from image_genarators.input_output_generators import generate_image_data, SEGMENTATION_DIRECTORY_LOCATION, save_image
import logging
np.set_printoptions(threshold=np.nan)
if __name__ == '__main__':
    logging.basicConfig(filemode="w+", filename="prediction_logs.txt", level=logging.DEBUG)
    model:keras.Model = keras.models.load_model("Fish_unet_10_137.cnn")
    validation_generator = generate_image_data(1,
                                               SEGMENTATION_DIRECTORY_LOCATION + "\\train",
                                               number_of_images=20,
                                               glob_pattern="ALB*.tif")
    for i in range(0, 20):
        validation_data = next(validation_generator)
        logging.debug(f"image {i} processed")
        image_reshaped = np.reshape(validation_data[0][0],
                                    (1, 768, 768, 3))
        image_prediction: np.ndarray = model.predict(image_reshaped)
        logging.debug(f"image prediction shape {image_prediction[0].shape}")
        save_image(image_prediction[0],
                   number=i,
                   mask=True)
