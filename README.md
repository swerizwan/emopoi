# EmoPOI: Emotion-Based Point of Interest Recommendation

# About Project
This research highlights the significance of understanding users' emotional states to enhance the user experience in recommendation applications, particularly in Point of Interest (POI) recommendations. Previous studies have overlooked emotions and lacked comprehensive datasets. In response, we propose an EmoPOI dataset and a novel approach that integrates facial feature extraction using Convolutional Neural Networks (CNNs) and emotion analysis through Long Short-Term Memory (LSTM) layers. Our method excels in accuracy compared to state-of-the-art techniques, leveraging FER-2013 and EmoPOI datasets.

# Workflow
The first step involves importing essential libraries by installing them on your system. The primary libraries necessary for code execution include:

1.	TensorFlow
2.	OpenCV
3.	Numpy
4.	Dlib
5.	EmoPOI files
6.	GUI (Tkinter)
   
# Installation
1.	pip install tensorflow
2.	pip install opencv-python
3.	pip install numpy
4.	Option 1: Download the Dlib and EmoPOI files as external zip files and extract them into the main project directory/folder where your project files are stored.
Option 2: If you're using PyCharm IDE, you can install the Dlib library directly in the project settings by navigating to Configure Interpreter and adding your required libraries from there.

# Datasets
FER-2013: You can download the FER2013 dataset from the link https://www.kaggle.com/datasets/msambare/fer2013
EmoPOI: EmoPOI dataset will be available upon request for research purpose after the paper will get accepted. 

# Step to Run the demo

•	Download this repository.
•	Create an 'Images' folder in your project and make subfolders for emotions such as Happy, Worried, Sad, Excited, Exhausted, Frustrated, Bored, and Neutral.
•	Utilize the 'video_frames.py' file to convert your own or any live video into frames of images, thereby generating a large dataset.
•	Place the 'image_augmentation.py' and 'emopoi_frontalface_alt.xml' files in each type of image folder. For example, place these files in the "happy" image folder and execute the program. It will detect faces from images, convert them into grayscale, and create new images in the same folder.
•	Next, create the model. You can copy the code from the 'training_inputs.txt' file, open the terminal in your project folder, paste the code, and hit enter.
•	Training the model will take approximately 20-25 minutes to complete. Use a large number of datasets for optimal accuracy. 
•	Upon training completion, two files named 'emopoi_retrained_graph.pb' and 'emopoi_retrained_labels.txt' will be generated.
•	Finally, run 'emotion_age_gender_recognition.py' (provide the proper path to your video). Based on the recognized emotion, gender, and age group, you will receive personalized POI recommendations suited to the user's preferences.
