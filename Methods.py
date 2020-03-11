import cv2
import skimage
from skimage.feature import greycomatrix
from skimage.measure import shannon_entropy


# resizing method. Weight facter can be given as 'W' ------------------------------------------------------------------
def resize(img, W):
    # Image resizing
    height, width = img.shape
    imgScale = W / width
    newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
    img = cv2.resize(img, (int(newX), int(newY)))
    return img

# Mean method for gray images------------------------------------------------------------------------------------------
def get_mean(img):
    # get rows and cols from image
    rows, cols = img.shape
    # defining variables
    row_val = 0
    total = 0
    for j in range(cols - 1):
        for i in range(rows - 1):
            row_val += img[i][j]
        total += row_val
        row_val = 0
    mean = total / (cols * rows)
    return mean

# Standard deviation for gray images  ----------------------------------------------------------------------------------
def get_standard_deviation(img):
    # calling mean function to get the mean
    mean = get_mean(img)
    # get rows and cols from image
    rows, cols = img.shape
    # defining variables
    row_val = 0
    total = 0
    for j in range(cols - 1):
        for i in range(rows - 1):
            row_val += (img[i][j] - mean) ** 2
        total += row_val
        row_val = 0
    std = abs(total / (cols * rows)) ** (1 / 2)
    return std

# Entropy for gray images    -------------------------------------------------------------------------------------------
def get_entropy(img):
    entropy = shannon_entropy(img)
    return entropy

# Skewness-------------------------------------------------------------------------------------------------------------
def get_skewness(img):
    # calling mean function to get the mean
    mean = get_mean(img)
    # calling standard deviation function to get the mean
    std = get_standard_deviation(img)
    # get rows and cols from image
    rows, cols = img.shape
    # get rows and cols from image
    row_val = 0
    total = 0
    for j in range(cols - 1):
        for i in range(rows - 1):
            row_val += (img[i][j]-mean)**3
        total += row_val
        row_val = 0
    skw = total/(cols*rows*std**3)
    return skw

# Kurtosis -------------------------------------------------------------------------------------------------------------
def get_kurtosis(img):
    # calling mean function to get the mean
    mean = get_mean(img)
    # calling standard deviation function to get the mean
    std = get_standard_deviation(img)
    # get rows and cols from image
    rows, cols = img.shape
    # get rows and cols from image
    row_val = 0
    total = 0
    for j in range(cols - 1):
        for i in range(rows - 1):
            row_val += (img[i][j] - mean) ** 4
        total += row_val
        row_val = 0
    k = total / (cols * rows * std ** 4)
    return k

# Energy ---------------------------------------------------------------------------------------------------------------
def get_energy(img):
    GLCM = greycomatrix(img, [1], [0], levels=256,
                        normed=True, symmetric=True)
    energy= skimage.feature.texture.greycoprops(GLCM, prop='energy')

    return energy[0][0]

# contrast -------------------------------------------------------------------------------------------------------------
def get_contrast(img):
    GLCM = greycomatrix(img, [1], [0], levels=256,
                        normed=True, symmetric=True)
    contrast = skimage.feature.texture.greycoprops(GLCM, prop='contrast')
    return contrast[0][0]

# homogeneity ----------------------------------------------------------------------------------------------------------
def get_homogeneity(img):
    GLCM = greycomatrix(img, [1], [0], levels=256,
                        normed=True, symmetric=True)
    homogeneity = skimage.feature.texture.greycoprops(GLCM, prop='homogeneity')
    return homogeneity[0][0]

# co-relation ----------------------------------------------------------------------------------------------------------
def get_corelation(img):
    GLCM = greycomatrix(img, [1], [0], levels=256,
                        normed=True, symmetric=True)
    co_rel = skimage.feature.texture.greycoprops(GLCM, prop='correlation')
    return co_rel[0][0]

# directional moment ---------------------------------------------------------------------------------------------------
def get_dissimilarity(img):
    GLCM = greycomatrix(img, [1], [0], levels=256,
                        normed=True, symmetric=True)
    dissimilarity = skimage.feature.texture.greycoprops(GLCM, prop='dissimilarity')
    return dissimilarity[0][0]
