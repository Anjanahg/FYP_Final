import cv2
import tensorflow as tf

def prepare(filepath):
    IMG_SIZE = 240  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-3, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.

def get_yes_no():
    model = tf.keras.models.load_model("Model_Yes_No/model_yes_no.model")

    prediction = model.predict([prepare('Images/Input/img_original.jpg')])

    result = int(prediction[0][0])
    return result
