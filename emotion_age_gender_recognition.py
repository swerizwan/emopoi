import time
import cv2
import image_labels
import os
import random
import subprocess
import pyglet
import pygame
import time
import threading


# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set the size of the face detector window
size = 4

# Add the Graphviz binary path to the system environment variable
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

# Define the path to the input video and load the face cascade classifier
video_path = 'videos/happysmile.mp4'
classifier = cv2.CascadeClassifier('emopoi/emopoi_frontalface_alt.xml')
global text

# Open a video capture object
webcam = cv2.VideoCapture(video_path) 

def faceBox(faceNet, frame):
    """
    Detect faces in a frame using a pre-trained deep learning model.

    Args:
    - faceNet: Face detection neural network model
    - frame: Input image frame

    Returns:
    - frame: Image frame with rectangles around detected faces
    - bboxs: List of bounding boxes for detected faces
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Define file paths for face, age, and gender detection models
faceProto = "emopoi/emopoi_opencv_face_detector.pbtxt"
faceModel = "emopoi/emopoi_opencv_face_detector_uint8.pb"
ageProto = "emopoi/emopoi_age_deploy.prototxt"
ageModel = "emopoi/emopoi_age_net.caffemodel"
genderProto = "emopoi/emopoi_gender_deploy.prototxt"
genderModel = "emopoi/emopoi_gender_net.caffemodel"

# Load pre-trained models for face, age, and gender detection using OpenCV's dnn module
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define constants for age and gender classification
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
# Categories of POIs
# Art Galleries and Historical Museums
AH = ['Zhejiang Art Museum', 'West Lake Museum', 'Zhejiang Museum of Natural History', 'Hangzhou Arts and Crafts Museum', 'Southern Song Dynasty Government Kiln Museum', 'Hangzhou Museum', 'Liangzhu Museum', 'China Fan Museum', 'National Wetland Museum of China', 'Zhejiang Science and Technology Museum', 'China Umbrella Museum']

# Playgrounds and Parks
PP = ['Hangzhou Songcheng', 'Hangzhou Botanical Garden', 'Hangzhou Paradise', 'Crazy Apple land. Amusement & Theme Parks', 'Hangzhou Orient Culture Park. Amusement & Theme Parks', 'Hangzhou Polar Ocean World. Amusement & Theme Parks', 'Hangzhou Ecopark', 'Hangzhou Luyu Spring Cultural Amusement/theme park']

# Restaurants and Ice ream Shops
RI = ['Dairy Queen', 'Lanting Chinese Restaurant', 'The Grandmas', 'Dongyishun', 'Louwailou', 'Hubin 28 Restaurant', 'Dragon Well Manor','Lètiān mǎ tè', 'Wòērmǎ gòuwù guǎngchǎng', 'Xinbailu Restaurant']

# Trendy Cafés and Coffee Shops
TC = ['Starbucks', 'Costa Coffee', 'Cafe Fiocco', 'C.straits Café', 'Caffe Bene', 'MoMo To Go', 'Funky Soul', 'Dio Coffee', 'Green Garden Hangzhou', 'C.straits Café']

# Live Music Venues and Concert Halls
LC = ['Huanglou JZ Club', 'Traveller Bar', 'Old Captain Lounge Bar', 'Party Space Bar', 'Grand Hyatt Hangzhou', 'Zhejiang Broadcast Concert Hall', 'Xintiandi Sun Theatre', 'Hangzhou Grand Theatre'] 

# Shopping Malls and Game Stores
SG = ['Hangzhou Building Shopping Center', 'Hangzhou Paradise', 'Hangzhou in City Plaza', 'Hangzhou Department Store', 'Hangzhou Culture Shopping Mall', 'Meijia Square', 'Hangzhou Hanglung Palace']

# Heritage Sites and Monuments
HM  = ['Gaijiaotian Former Residence', 'Tomb of Yue Fei', 'Yue Fei Temple', 'Leifeng Pagoda', 'Longhong Cave', 'Lingyin Temple', 'City God Pavilion', 'Xixi National Wetland Park', 'Leifeng Pagoda', 'Lingyin Temple']

# Landscape and Countryside
LC = ['Leifeng Tower', 'Hangzhou Lingyin Temple and Feilai Peak Scenic Spot', 'Quyuanfenghe', 'Liulangwenying', 'Gushan', 'Yunqi Bamboo Trail', 'Three Pools Mirroring the Moon', 'Lingering Snow on the Broken Bridge', 'Sir Georg Solti in Early Spring']

# Clubs and Bars
CB = ['G-Plus', 'Basement', 'Shares Bar', 'Asuka Karaoke Bar', 'Vesper Bar', 'JZ Club', 'Aurora Cocktail Lounge', '9-Club', 'H·Linx', 'AA international cartoon cute Club']

# Yoga Retreats and Fitness Centers
YF = ['Hangzhou Jingyuan Yoga', 'Hangzhou Sandi Yoga Limited Company', 'Soho Gym', 'Weider-Tera Fitness Club Club', 'Huanglong Fitness Club', 'Hangzhou Jiadongle City Window', 'Guǎng xìng jiànshēn bīnjiāng fēn guǎn bīnjiāng fēn guǎn']

# Set the padding value
padding = 20

# Get the current time
now = time.time() 
# Set a future time (e.g., 60 seconds from now)
future = now + 60

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = webcam.read()

    # Get face bounding boxes
    frame, bboxs = faceBox(faceNet, frame)

    # Process each face
    for bbox in bboxs:
        # Extract the face region with padding
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        # Preprocess the face image
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        gender = gender.title()

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        # Categorize age groups
        if age == '(0-2)' or age == '(4-6)' or age == '(8-12)':
            age = 'Child'
        elif age == '(25-32)' or age == '(15-20)' or age == '(38-43)':
            age = 'Adult'
        elif age == '(60-100)' or age == '(60-100)':
            age = 'Old'
        else:
            age = 'Over 100'
        age = age.title()

        # Save the face image to a file and label it using an external module
        FaceFileName = "emopoi_test.jpg"
        cv2.imwrite(FaceFileName, face)
        text = image_labels.main(FaceFileName)
        text = text.title()

        # Prepare the label for display
        label = "{},{},{}".format(gender, age, text)

        # Display information based on detected emotion
        for emotion in ['Happy', 'Worried', 'Sad', 'Excited', 'Exhausted', 'Bored', 'Frustrated', 'Neutral']:
            if text == emotion:
                cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)

    # Resize and display the frame
    frame = cv2.resize(frame, (1000, 700))
    cv2.imshow('Gender, Agr Group, and Emotion', frame)

    # Wait for a key press and exit the loop if needed
    key = cv2.waitKey(30) & 0xff

# Check if the specified future time has passed
    if time.time() > future:
        try:
            # Close all OpenCV windows
            cv2.destroyAllWindows()

            # Determine the emotion, gender, and age categories, then select a random file based on the conditions
            if (text == 'Happy' or text == 'Excited') and (gender == 'Male' or gender == 'Female') and age == 'Child':
                randomfile = random.choice(PP)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Worried' or text =='Sad') and (gender == 'Male' or gender == 'Female') and age == 'Child':
                randomfile = random.choice(RI)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Exhausted' or text =='Frustrated') and (gender == 'Male' or gender == 'Female') and age == 'Child':
                randomfile = random.choice(PP)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Bored' and (gender == 'Male' or gender == 'Female') and age == 'Child':
                randomfile = random.choice(PP)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Neutral' and (gender == 'Male' or gender == 'Female') and age == 'Child':
                randomfile = random.choice(RI)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Happy' or text =='Excited') and gender == 'Male' and age == 'Adult':
                randomfile = '1: ' + random.choice(CB)
                randomfile = randomfile + ' 2: ' + random.choice(YF)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Worried' or text =='Sad') and gender == 'Male' and age == 'Adult':
                randomfile = random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Exhausted' or text =='Frustrated') and gender == 'Male' and age == 'Adult':
                randomfile = random.choice(CB)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Bored' and gender == 'Male' and age == 'Adult':
                randomfile = random.choice(YF)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Neutral' and gender == 'Male' and age == 'Adult':
                randomfile = '1: ' + random.choice(CB)
                randomfile = randomfile + ' 2: ' + random.choice(YF)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Happy' or text =='Excited') and (gender == 'Male' or gender == 'Female') and age == 'Old':
                randomfile = random.choice(AH)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Worried' or text =='Sad') and gender == 'Male' and age == 'Old':
                randomfile = random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Exhausted' or text =='Frustrated') and gender == 'Male' and age == 'Old':
                randomfile = random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Bored' and gender == 'Male' and age == 'Old':
                randomfile = '1: ' + random.choice(HM)
                randomfile =  randomfile + ' 2: ' + random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Neutral' and gender == 'Male' and age == 'Old':
                randomfile = random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Happy' or text =='Excited') and gender == 'Female' and age == 'Adult':
                randomfile = '1: ' + random.choice(TC)
                randomfile =  randomfile + ' 2: ' + random.choice(LC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Worried' or text =='Sad') and gender == 'Female' and age == 'Adult':
                randomfile = '1: ' + random.choice(CB)
                randomfile =  randomfile + '2: ' + random.choice(YF)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Exhausted' or text =='Frustrated') and gender == 'Female' and age == 'Adult':
                randomfile = random.choice(TC)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Bored' and gender == 'Female' and age == 'Adult':
                randomfile = random.choice(CB)
                print('You are bored,' + randomfile)
                
            if text =='Neutral' and gender == 'Female' and age == 'Adult':
                randomfile = '1: ' + random.choice(RI)
                randomfile =  randomfile + '2: ' + random.choice(TC)
                randomfile =  randomfile + '3: ' + random.choice(LC)
                randomfile =  randomfile + '4: ' + random.choice(SG)
                randomfile =  randomfile + '5: ' + random.choice(HM)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Worried' or text =='Sad') and gender == 'Female' and age == 'Old':
                randomfile = '1: ' + random.choice(RI)
                randomfile =  randomfile + '2: ' + random.choice(TC)
                randomfile =  randomfile + '3: ' + random.choice(LC)
                randomfile =  randomfile + '4: ' + random.choice(CB)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if (text =='Exhausted' or text =='Frustrated') and gender == 'Female' and age == 'Old':
                randomfile = random.choice(HM)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Bored' and gender == 'Female' and age == 'Old':
                randomfile = random.choice(HM)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
                
            if text =='Neutral' and gender == 'Female' and age == 'Old':
                randomfile = '1: ' + random.choice(LC)
                randomfile =  randomfile + '2: ' + random.choice(YF)
                print('You are ' + gender +  ', ' + age + ' and ' + text +  ' The recommended POIs are ' + randomfile)
            # Break out of the loop after processing the emotion       
            break

        # Handle any exceptions and print an error message
        except :
            print('Please stay focus in Camera frame atleast 15 seconds & run this program again :)')
            break

    # Check if the 'Esc' key was pressed to exit the loop
    if key == 27:  
        break
