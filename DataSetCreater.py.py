# REFRENCED CODE:
#   https://www.datacamp.com/community/tutorials/face-detection-python-opencv
#
# DATE: October 12th, 2021
# DESCRIPTION: This program is meant to provide facial-recognition capabilities to the rest of the project.
#
# DEPENDENCIES:
#   OpenCV | pip install opencv-python \\For computer vision
#   matplotlib | pip install -U matplotlib \\For testing
#   numpy | pip install numpy \\For matrix manipulation


__opencv_version__ = r'4.1.1'

import sys
import time
import cv2 as cv
import math as ma
# import BotInterface as bi

#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

size =(1300,800)

if(__opencv_version__ != cv.__version__):
    print('WARNING: The OpenCV version being used ({}) is different from the OpenCV version this module was written in! ({})'.format(cv.__version__, __opencv_version__))
__training_xml__ = cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
__cascade__ = cv.CascadeClassifier(__training_xml__) #load the already-trained facial-recongition model


#specify which detection model to use
def use_model(cascade):
    __cascade__ = cascade

#given an image and a scale factor, find faces in an image and return that image with rectangles around the faces
def detect_faces(image):
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY) #create a grayscale copy of the image (used for face detection)

    #find the faces.
    #TODO: Have an explanation as to how 'detectMultiScale' works so we can better-utilize scaleFactor and minNeighbors

    #** scaleFactor is a parameter specifying how much the image size is reduced at each image scale.
    #** minNeighbors is a parameter specifying how many neighbors each candidate rectangle should have to retain it.
    faces_rects = __cascade__.detectMultiScale(image_gray, scaleFactor = 1.3, minNeighbors = 5)

    #print how many faces were found
    # print('Faces found: {}'.format(len(faces_rects)))
    return faces_rects


def createLine(im,st, en,co):
    cv.line(im,st, en,co , 5)
def draw_rects(image, rects):
    #draw rectangles around the bounds of the detected faces
    xx,yy,ww,hh = 0,0,0,0
    area = 0
    out_image = image.copy()

    stat = False
    for (x, y, w, h) in rects:
        stat = True
        #draws a red rectangle with a thickness of 2
        if w*h > area:
            area = w*h
            xx,yy,ww,hh = x,y,w,h
        cv.rectangle(out_image, (x, y), (x + w - 15, y + h - 15), (0, 255, 0), 2)

    if(stat):
        start = ((int)(size[0] / 2),(int) (size[1] / 2))
        tl = ((int)(xx + (ww / 2)), (int)(yy + (hh / 2)))

        th = ((int)(size[0] / 2), (int)(yy + (hh / 2)))
        thd = start[1] - th[1]

        tw = ((int)(xx + (ww / 2)), (int)(size[1] / 2))
        twd = start[0] - tl[0]

        createLine(out_image, start, tl , (255,0,0))
        createLine(out_image, start, th, (0, 255, 0))
        createLine(out_image, th, tl, (0, 0, 255))

        print("Angle = ",ma.tan(twd/(thd+0.001)))
        cv.rectangle(out_image, (xx, yy), (xx + ww, yy + hh), (0, 0, 255), 2) # The closest person has a red rectangle

    return out_image



if __name__ == "__main__":
    # bi.Calibrat()
    # Using camera
    cap = cv.VideoCapture(0) # The Parameter is the index of the camera
    if not cap.isOpened():
        print("Unable to capture camera")
    num_faces = 0
    max_faces = 10
    while cap.isOpened() and num_faces < max_faces:
        ret, image = cap.read()  # Read each frame as an image
        # test_image = cv.imread(image)
        if ret:
            face_rects = detect_faces(image)  # detect the faces
            if len(face_rects) > 0:
                num_faces += 1
                x,y,w,h = face_rects[0]
                #TODO: Save image
                image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                face_gray = image_gray.copy()
                face_gray = face_gray[y:y+h, x:x+w]
                filepath = "{}/face_{}_{}.png".format("thor", "thor", num_faces)
                print("Saving image {} out of {} to `{}`", num_faces, max_faces, filepath)
                cv.imwrite(filepath, face_gray)
                time.sleep(0.5)

            #TODO: change from an image display to a video display
            # #show the image and waits for a key press before exiting
            draw_image = draw_rects(image, face_rects)
            draw_image = cv.resize(draw_image, size)
            cv.imshow('Detected Faces', draw_image)
            key = cv.waitKey(1) #waits for 2 milliseconds for a key press on a OpenCV window
            if key == 113: # it breaks when q is pressed
                break
        else:
            print("Unable to get video frame from camera")
            break
    cap.release()
    cv.destroyAllWindows()


# if __name__ == '__main__':
#     import sys
#     test_image = cv.imread(sys.argv[1])  #load the image from the file specified from the command line
#     result_image = detect_faces(test_image)  #detect the faces
#     # #show the image and waits for a key press before exiting
#     cv.imshow('Detected Faces', result_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# For testing from code
# image = r'/Users/rediettadesse/Documents/GitHub/ProjectX0/image3.jpeg' #image full path
# test_image =  cv.imread(image) #load the image from the file specified from the command line
# result_image = detect_faces(test_image)  #detect the faces
# draw_rects(test_image,result_image)
# #show the image and waits for a key press before exiting

# cv.imshow('Detected Faces', test_image)
# # # print(cv.waitKey(0))
# cv.waitKey(0)
# cv.destroyAllWindows()
