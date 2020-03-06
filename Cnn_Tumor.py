from keras_segmentation.predict import predict
from keras.models import *

predict(
	checkpoints_path="./Model/model",
	inp="./Input/3 no.JPG",
	out_fname="./Output/12.JPG"
)
