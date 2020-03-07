from keras_segmentation.predict import predict


predict( 
	checkpoints_path="./Model/Model",
	inp="./Input/54-s.jpg",
	out_fname="./Output/img_segment_by_cnn.jpg"
)

