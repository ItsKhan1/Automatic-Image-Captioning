from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import numpy as np

class InceptionModel:
    def __init__(self):
        self.model = InceptionV3(weights='imagenet')
        self.new_input = self.model.input
        self.hidden_layer = self.model.layers[-2].output
        self.model = Model(self.new_input, self.hidden_layer)

    def predict(self, image):
        # Preprocess the image
        image = image.resize((299, 299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0

        # Perform prediction using the Inception model
        pred = self.model.predict(image)
        print("Inception model output shape:", pred.shape)
        return pred
