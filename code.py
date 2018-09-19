#Created by Shikha Singh

#importing the required libraries
import cv2,numpy as np

#loading the image containing blobs
img = cv2.imread('C:/Users/pluto/folder_img/blobs2.jpg',0)
#showing the original image
cv2.imshow('original image',img)
#waitKey(0) defines that the window'll exit for any key pressed by the user.
cv2.waitKey(0)

#creating an object named 'detctor' for SimpleBlobDetector() class
#for openCV versions 3.x & above this class is replaced by SimpleBlobDetector_create() class
detector = cv2.SimpleBlobDetector_create()
#detect blobs by passing input image into the detector
#keypoints refers to the spatial locations or points in the image that have something in common, for e.g. threshold value
#keypoints are special because they are scale-invariant i.e. whether the image rotates,translates,shrinks/expands or distorts,
#the keypoints remains same even in the modified image
keypoints = detector.detect(img)

#draw blobs on our image as red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(img,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#to display total number of blobs detected
number_of_blobs = len(keypoints)
text = "total number of blobs" + str(len(keypoints))
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

#displaying image with blob keypoints
cv2.imshow('blobs using default parameters',blobs)
cv2.waitKey(0)

#to diffrentiate the circular blobs from non-circular blobs, we need to set some parameters
#these parameters can be set by using the functions of SimpleBlobDetector_Params() class

#setting filtering parameters
#firstly, creating an object to access SimpleBlobDetector_params() class functions
params = cv2.SimpleBlobDetector_Params()

#set area filtering parameters 
#in this, we can define both min as well as max area for the blob
params.filterByArea = True
params.minArea = 100

#set circularity filtering parameters
#the minCircularity value =1 defines a perfect circle & 0 defines it's opposite
params.filterByCircularity = True
params.minCircularity = 0.9

#set convexity filtering parameters 
params.filterByConvexity = False
params.minConvexity = 0.2

#setting inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

#creating a detector with the specified parameters
detector = cv2.SimpleBlobDetector_create(params)
#detect circular blobs
keypoints = detector.detect(img)

#draw blobs on our image as red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(img,keypoints,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#displaying number of circular blobs detected
number_of_blobs = len(keypoints)
text = "number of circular blobs" + str(len(keypoints))
cv2.putText(blobs,text,(20,550),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

#highlighting circular blobs among several blobs
cv2.imshow('filtering circular blobs only',blobs)
cv2.waitKey(0)

cv2.destroyAllWindows()
