import cv2
import numpy as np 
import mediapipe as mp
import matplotlib.pyplot as plt
import itertools





# mp_face_mesh = mp.solutions.face_mesh

# face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
#                                          min_detection_confidence=0.5)

# mp_drawing = mp.solutions.drawing_utils

# mp_drawing_styles = mp.solutions.drawing_styles
# cap = cv2.VideoCapture(0)
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'.
#             continue
        

#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)
        

#         LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
#         RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
#         #NOSE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_NOSE)))
#         MOUTH_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
#         LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
#         RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
    

#get facial classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

#read images
filter = cv2.imread('jason.png')

#get shape of witch
original_filter_h,original_filter_w,filter_channels = filter.shape

#convert to gray
filter_gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of witch
ret, original_mask = cv2.threshold(filter_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]
# while True:
#     #read video
    
#     

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     #for each face
#     for (x,y,w,h) in faces:
#         #draw rectangle around face
#         img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         #select face as region of interest 
#         roi_g = gray[y:y+h,x:x+h]
#         roi_c = img[y:y+h,x:x+h]
#         #within region of interest find eyes
#         eyes = eye_cascade.detectMultiScale(roi_g)
#         #for each eye
#         for (ex,ey,ew,eh) in eyes:
#             #draw retangle around eye
#             cv2.rectangle(roi_c, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#         cv2.imshow('img',img) #shows image
#         keypressed = cv2.waitKey(1) & 0xFF
#         if keypressed == 0x1b: #escape key
#             break
#         #cv2.waitKey(0) #waits until a key is pressed to progress

while True:   #continue to run until user breaks loop
    
    #read each frame of video and convert to gray
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #for every face found:
    for (x,y,w,h) in faces:
        #draw rectangle around face
        print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #retangle for testing purposes
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #filter size in relation to face by scaling
        filter_width = int(1.5 * face_w)
        filter_height = int(1*filter_width * original_filter_h / original_filter_w)
        
        #setting location of coordinates of filter
        filter_x1 = face_x2 - int(face_w/2) - int(filter_width/2)
        filter_x2 = filter_x1 + filter_width
        filter_y1 = face_y1 - int(face_h) + int(filter_height/2)
        filter_y2 = filter_y1 + filter_height 

        #check to see if out of frame
        if filter_x1 < 0:
            filter_x1 = 0
        if filter_y1 < 0:
            filter_y1 = 0
        if filter_x2 > img_w:
            filter_x2 = img_w
        if filter_y2 > img_h:
            filter_y2 = img_h

        #Account for any out of frame changes
        filter_width = filter_x2 - filter_x1
        filter_height = filter_y2 - filter_y1

        #resize witch to fit on face
        filter = cv2.resize(filter, (filter_width,filter_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (filter_width,filter_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (filter_width,filter_height), interpolation = cv2.INTER_AREA)

        #take ROI for witch from background that is equal to size of witch image
        roi = img[filter_y1:filter_y2, filter_x1:filter_x2]

        #original image in background (bg) where witch is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(filter,filter,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[filter_y1:filter_y2, filter_x1:filter_x2] = dst

        break
        
    #display image
    cv2.imshow('img',img) 

    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windows
