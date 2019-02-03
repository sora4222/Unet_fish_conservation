import keras
import numpy as np
from input_output_generators import generate_image_data, SEGMENTATION_DIRECTORY_LOCATION, save_image
np.set_printoptions(threshold=np.nan)
if __name__ == '__main__':
    model:keras.Model = keras.models.load_model("Fish_unet_3_137.cnn")

    validation_generator = generate_image_data(1, SEGMENTATION_DIRECTORY_LOCATION + "\\validation", starting_with="ALB_img_00121.tif")
    validation_data = next(validation_generator)
    image_reshaped = np.reshape(validation_data[0][0], (1, 768, 768, 3))
    image_prediction: np.ndarray = model.predict(image_reshaped)
    print(image_prediction.shape)
    save_image(image_prediction[0], number=0, mask=True)
