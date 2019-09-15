import cv2
from skimage.feature import hog
import numpy as np
import joblib
import PIL
from PIL import Image

def get_score(SVMclf, image):

    image = get_corner(image)
    # Convert to grayscale and apply Gaussian filtering
    img_gray = cv2.cvtColor(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Threshold the image
    ret, img_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    nbrs = []
    # For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
    for rect in rects:

        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = img_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        nbr = SVMclf.predict(np.array([roi_hog_fd], 'float64'))
        nbrs = np.append(nbrs, nbr)
    #nbrs = np.array(nbrs,)
    print(nbrs)
    score = connec_nbrs(nbrs)

    return score

def get_corner(image):
# image = pyscreen.grab()
    #w, h = image.size
    w, h = 1920, 1080
    print(h, w)
    image = image.crop((w-202, 25, w-72, 72))
   # image.save('_0.png')
    return image

def connec_nbrs(nbrs):
    answer = nbrs[0]
    for i in range(1, len(nbrs)):
        tmp = nbrs[i] * 10 **i
        answer += tmp

    print('score', answer)
    return answer

#TESTS

# Load the classifier
#clf = joblib.load("score_SVMclf.pkl")
# Read the input image
#image = Image.open("images/gameplay/frame7.jpg")

#get_score(clf, image)

#connec_nbrs([3, 7,  5, 6])

#if __name__ == '__main__':
    #get_corner(image)

