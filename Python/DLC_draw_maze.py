# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:35:14 2024

@author: shukl
"""


#%%

import cv2
import numpy as np
import os

points = []  # To store clicked points
connected_lines = []  # To store connected lines

def click_event(event, x, y, flags, param):
    global points, connected_lines
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        draw_circles()  # Draw circles at clicked points
        print(f"Clicked at: ({x}, {y})")

        if len(points) == 2:
            connect_nodes()
            draw_w_shape()
        elif len(points) == 4:
            connect_nodes()
            draw_w_shape()
        elif len(points) == 6:
            connect_nodes()
            draw_w_shape()

def draw_circles():
    for point in points:
        cv2.circle(image, point, 5, (0, 0, 255), -1)  # Draw red circles at clicked points

def connect_nodes():
    if len(points) == 2:
        # Connect nodes 1-2
        connected_lines.append((points[0], points[1]))
    elif len(points) == 4:
        # Connect nodes 3-4
        connected_lines.append((points[2], points[3]))
    elif len(points) == 6:
        # Connect nodes 5-6
        connected_lines.append((points[4], points[5]))
        # Connect nodes 2-6 with a straight line
        connected_lines.append((points[1], points[3]))
        connected_lines.append((points[3], points[5]))

def draw_w_shape():
    # Draw lines connecting nodes without closing the shape
    for line in connected_lines:
        cv2.line(image, line[0], line[1], (0, 0, 255), 2)

    # Save pixel positions
    # np.save(os.path.join('C:\\Users\\shukl\\Downloads', 'pixel_positions.npy'), np.array(points))
    print("Pixel Positions:")
    for point in points:
        print(f"({point[0]}, {point[1]})")

# if __name__ == "__main__":
#     image_path = 'C:\\Users\\shukl\\Downloads\\download.jpg'
#     image = cv2.imread(image_path)
#     cv2.imshow("Image", image)
#     cv2.setMouseCallback("Image", click_event)

#     while True:
#         cv2.imshow("Image", image)
#         key = cv2.waitKey(1) & 0xFF

#         if key == 27:  # Press 'Esc' to exit
#             break

#     cv2.destroyAllWindows()
    
    


#%% Do for rat 1 for cohort 2

import os
import glob

cohort = 'Cohort 1'
dataFolder = 'E:/Jadhav lab data/Behavior/' + cohort + '/Social W/08-28-2023'
os.chdir(dataFolder)

# Filter all .mp4 files without "DLC" in their filenames
mp4_files = [os.path.join(dataFolder, file) for file in glob.glob('*.mp4') if 'DLC' not in file]
# Filter all .mp4 files with "Image" in their filenames but not "DLC"
# mp4_files = [os.path.join(dataFolder, file) for file in glob.glob('*.mp4') if 'Image' in file and 'DLC' not in file]

# Print the list of selected files with full paths
for mp4_file in mp4_files:
    print(mp4_file)
    sessionVideo = mp4_file
    
    # if os.path.isfile(sessionVideo.replace(".mp4",".jpeg"))==False:
    vidcap = cv2.VideoCapture(sessionVideo)
    success,image = vidcap.read(1)
    cv2.imwrite(sessionVideo.replace(".mp4",".jpeg"), image)     # save frame as JPEG file    
    frame = sessionVideo.replace(".mp4",".jpeg")  
        
    # frame_rat1 = frame.replace(".1.jpeg","-Rat1.npy")
    frame_rat2=frame.replace(".1.jpeg","-Rat2.npy")
        
    if os.path.isfile(frame_rat2)==False:
        nodes1 = []
        # nodes2=[]
        preview = None
        lastNode = (-1, -1)




        # set the named window and callback    
        image = cv2.imread(frame)
        points = []
        connected_lines = []
        cv2.namedWindow(frame)
        cv2.setMouseCallback(frame, click_event)

        while (True):
        # if we are drawing show preview, otherwise the image
            if preview is None:
                cv2.imshow(frame,image)
            else:
                cv2.imshow(frame,preview)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break;

            elif len(nodes1)==6:
                break

        cv2.destroyAllWindows()
        cv2.waitKey(0)
        
        # np.save(frame_rat1,np.array(points))
        np.save(frame_rat2,np.array(points))
        
#%%
        
