import pickle
import numpy as np
import cv2
from sklearn.externals import joblib
from Methods import*



# Calling methods  -------------------------------------------------------------------------------------------------
def get_value_array(img):
    mean = get_mean(img)
    entropy = get_entropy(img)
    STD = get_standard_deviation(img)
    skewness = get_skewness(img)
    kurtosis = get_kurtosis(img)
    contrast = get_contrast(img)
    homogeneity = get_homogeneity(img)
    co_relation = get_corelation(img)
    energy = get_energy(img)
    dissimilarity = get_dissimilarity(img)

    value_array = [[mean, entropy, kurtosis, STD, skewness,  contrast, homogeneity, co_relation, energy, dissimilarity]]
    return value_array


# Random forest classifier  -------------------------------------------------------
def get_rf_result(img):
    # resizing images
    image = resize(img, 200)
    image = np.asanyarray(image)
    # Load the model from the file
    rf_trained_model = joblib.load('./Model_Classification/rf_trained_model.pkl')
    # getting the value array
    values = get_value_array(image)
    # Use the loaded model to make predictions
    result = rf_trained_model.predict(values)
    return result, values



