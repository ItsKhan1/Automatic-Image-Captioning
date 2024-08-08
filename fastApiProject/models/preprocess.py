from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_image(image, inception_model):
    # Get the feature vector for the image
    image_features = inception_model.predict(image)
    return image_features

def preprocess_sequence(sequence, wordtoix, max_length):
    # Convert sequence to integer encoding
    sequence = [wordtoix[word] for word in sequence.split() if word in wordtoix]
    sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
    return sequence


