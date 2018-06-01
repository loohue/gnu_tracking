
import cv2
import numpy as np
import pandas as pd
import os
import re
from math import *
import math
import time

# 4k camera matrix
camera_matrix = np.array( [[  2467.726893, 0,  1936.02964], [0, 2473.06961, 1081.48243], [0, 0,1.0]])
dc = np.array( [ -1.53501973e-01,3.04457563e-01,8.83127622e-05,6.93998940e-04,-1.90560255e-01])

# 1080 camera matrix
#camera_matrix = np.array([[1.18208317e+03,0, 9.52682573e+02],[0, 1.18047659e+03, 5.37914578e+02],  [ 0,0,1.0]])
#dc = np.array([-0.15515428,  0.2575828,   0.00030817,  0.00119713, -0.21363664])

S = (4096,2160)
# *****************************************************
# setup the rotation matrix to correct for the angle
# *****************************************************
# set the angle of the camera here (subtract from 180 depending on the axis)
angle= 180-131.5
inputName = '284STAB.AVI'
alpha_=angle
beta_ = 90
gamma_=90
alpha = (alpha_ - 90.)*pi/180;
beta = (beta_ - 90.)*pi/180;
gamma = (gamma_ - 90.)*pi/180;
f = 12447
dist = 800
w = S[0]
h= S[1]
f=camera_matrix[0,0]
A1 = np.array([[1, 0, -w/2],[0,1, -h/2],[0, 0,0],[0, 0,1]])
RX = np.array([[1,0,0,0],[0, cos(alpha), -sin(alpha), 0],[0,sin(alpha), cos(alpha),0],[0,0,0,1]])
R = RX #* RY * RZ; 
T = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0,0, 1, dist],[0, 0, 0, 1]])
A2 = np.array([[f, 0, w/2,0],[0, f, h/2, 0], [0,0,1,0]])
A2[0:3,0:3]=camera_matrix
M2 = np.dot(A2,np.dot(T, np.dot(R,A1)))
# *****************************************************
# *****************************************************



warp_mode = cv2.MOTION_HOMOGRAPHY
number_of_iterations = 20

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = -1e-16;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)


# set filenames for input and for saving the stabilized movie
inputName = 'DJI_0284.MP4'
outputName = '284STAB2.AVI'

# open the video
cap = cv2.VideoCapture(inputName)
fps = round(cap.get(cv2.CAP_PROP_FPS))

fStop = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# reduce to 12 frames a second - change number to required frame rate
ds = math.ceil(fps/12)

out = cv2.VideoWriter(outputName, cv2.VideoWriter_fourcc('M','J','P','G'), fps/ds, S, True)

  
im1_gray = np.array([])
first = np.array([])
fStop = 600
warp_matrix = np.eye(3, 3, dtype=np.float32) 
#warp_matrix = np.eye(2, 3, dtype=np.float32) 
full_warp = np.eye(3, 3, dtype=np.float32)
for tt in range(fStop):
    # Capture frame-by-frame
    _, frame = cap.read()

    if (tt%ds!=0):
        continue
    print(tt,fStop)
    if not(im1_gray.size):
        # enhance contrast in the image
        im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        first = frame.copy()
    
    im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    

    try:
        # find difference in movement between this frame and the last frame
        (cc, warp_matrix) = cv2.findTransformECC(im2_gray,im1_gray,warp_matrix, warp_mode, criteria)    
    except cv2.error as e:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        
    # this frame becames the last frame for the next iteration
    im1_gray =im2_gray.copy()
    
    # alll moves are accumalated into a matrix
    #full_warp = np.dot(full_warp, np.vstack((warp_matrix,[0,0,1])))
    full_warp = np.dot(full_warp, warp_matrix)
    # create an empty image like the first frame
    im2_aligned = np.empty_like(frame)
    np.copyto(im2_aligned, first)
    # apply the transform so the image is aligned with the first frame and output to movie file
    #im2_aligned = cv2.warpAffine(frame, full_warp[0:2,:], (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
    im2_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), dst=im2_aligned, flags=cv2.INTER_LINEAR  , borderMode=cv2.BORDER_TRANSPARENT)
    # now rotate the image
    und_im = cv2.undistort(im2_aligned,camera_matrix,dc)
    rotated = cv2.warpPerspective(und_im, M2, (S[0],S[1]),flags=cv2.WARP_INVERSE_MAP)
    out.write(rotated)
#    out.write(im2_aligned)
    #cv2.imwrite(str(tt)+'stab.jpg',im2_aligned)

cap.release()
out.release()


