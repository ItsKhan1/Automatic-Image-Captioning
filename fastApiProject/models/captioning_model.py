import numpy as np
import traceback
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CaptioningModel:
    def __init__(self, model_path, wordtoix, ixtoword, max_length):
        self.model = load_model(model_path)
        self.wordtoix = wordtoix
        self.ixtoword = ixtoword
        self.max_length = max_length

    def predict(self, image, sequence):
        try:
            in_text = 'startseq'
            for i in range(self.max_length):
                sequence = [self.wordtoix.get(word, 0) for word in in_text.split()]  # Default to 0 if word not found
                sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
                print(f"Sequence for prediction (iteration {i}): {sequence.shape}")

                print(f"Input image shape: {image.shape}")
                print(f"Input sequence shape: {sequence.shape}")

                yhat = self.model.predict([image, sequence], verbose=0)
                print(f"Prediction output shape: {yhat.shape}")

                yhat_index = np.argmax(yhat)
                print(f"Predicted index: {yhat_index}")

                # Handle out-of-vocabulary predictions
                word = self.ixtoword.get(yhat_index, '<UNK>')  # Use a placeholder for unknown indices
                print(f"Predicted word: {word}")

                if word == '<UNK>':
                    print(f"Predicted index {yhat_index} is out of vocabulary")

                in_text += ' ' + word
                if word == 'endseq':
                    break

            final_caption = in_text.split()[1:-1]
            final_caption = ' '.join(final_caption)
            return final_caption
        except Exception as e:
            print("Error in predict function:", str(e))
            raise e
