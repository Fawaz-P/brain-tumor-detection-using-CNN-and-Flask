from flask import Flask, render_template, request, redirect
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
import os
import time

app = Flask(__name__)

class ImagePreprocessor:
    def __init__(self, target_size=(240, 240)):
        self.target_size = target_size

    def toGrayScale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def applyGaussianBlur(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred

    def applyThresholding(self, image, lower=0, upper=255, default=cv2.THRESH_BINARY):
        _, thresh = cv2.threshold(image, lower, upper, default)
        return thresh

    def applyAdaptiveThresholding(self, image):
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return thresh

    def applyErosion(self, image):
        eroded = cv2.erode(image, None, iterations=2)
        return eroded

    def applyDilation(self, image):
        dilated = cv2.dilate(image, None, iterations=2)
        return dilated

    def findContours(self, image):
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts

    def findExtremePoints(self, contours):
        c = max(contours, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        return extLeft, extRight, extTop, extBot

    def cropAndResizeImage(self, image, extLeft, extRight, extTop, extBot):
        cropped_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        resized_image = cv2.resize(cropped_image, dsize=self.target_size, interpolation=cv2.INTER_CUBIC)
        normalized_image = resized_image / 255.
        reshaped_image = normalized_image.reshape((1, *self.target_size, 3))
        return reshaped_image


class DisplayTumor:
    def __init__(self, img):
        self.orig_img = np.array(img)
        self.cur_img = np.array(img)
        self.kernel = np.ones((3, 3), np.uint8)
        self.thresh = None

    def remove_noise(self):
        imagePre = ImagePreprocessor()
        gray = imagePre.toGrayScale(self.orig_img)
        ret, self.thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(self.thresh, cv2.MORPH_OPEN, self.kernel, iterations=2)
        self.cur_img = opening

    def display_tumor(self):
        if self.thresh is None:
            self.remove_noise()

        # sure background area
        sure_bg = cv2.dilate(self.cur_img, self.kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(self.cur_img, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.orig_img, markers)
        self.orig_img[markers == -1] = [255, 0, 0]

        tumor_image = cv2.cvtColor(self.orig_img, cv2.COLOR_HSV2BGR)
        self.cur_img = tumor_image

    def get_current_image(self):
        return self.cur_img

   

def make_prediction(img):
    model = load_model('BrainTumorBig.h5')
    input_img = np.expand_dims(img, axis=0)
    res = (model.predict(input_img) > 0.5).astype("int32")
    return res
    

def show_result(img_path):
    image = cv2.imread(img_path)
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)



    pred = make_prediction(img)
    if pred:
        label = "Tumor Detected"
        print("Tumor Detected")
        heatmap = np.squeeze(pred)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        timestamp = int(time.time())  # Generate timestamp
        result_path = f'static/tmp/result_image.jpg?{timestamp}'

    else:
        label = "No Tumor"
        timestamp = int(time.time())  # Generate timestamp
        result_path = f'static/tmp/result_image.jpg?{timestamp}'
        print("No Tumor")

    plt.show()
    return result_path,label



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        # Save the uploaded file to a temporary location
        file_path = 'static/tmp/temp_image.jpg'
        file.save(file_path)

        # Perform image preprocessing and tumor detection
        image = cv2.imread(file_path)
        preprocessor = ImagePreprocessor()
        gray_image = preprocessor.toGrayScale(image)
        blurred_image = preprocessor.applyGaussianBlur(gray_image)
        thresholded_image = preprocessor.applyThresholding(blurred_image)
        contours = preprocessor.findContours(thresholded_image)
        extLeft, extRight, extTop, extBot = preprocessor.findExtremePoints(contours)
        cropped_image = preprocessor.cropAndResizeImage(image, extLeft, extRight, extTop, extBot)

        # Tumor detection and visualization
        display_tumor = DisplayTumor(image)
        display_tumor.display_tumor()
        current_image = display_tumor.get_current_image()

        # Save the current image with tumor visualization
        result_path = 'static/tmp/result_image.jpg'
        cv2.imwrite(result_path, current_image)

        # Delete the temporary uploaded image
        os.remove(file_path)

        # Redirect to the result display route
        return redirect('/result')

    return render_template('index.html')

@app.route('/result')
def display_result():
    # Get the path of the result image
    result_path = 'static/tmp/result_image.jpg'

    # Perform tumor prediction and display the result
    result_image_url,label = show_result(result_path)

    # Render the result template with the image path
    return render_template('result.html', image_path=result_image_url,label=label)

if __name__ == '__main__':
    app.run(debug=True)
