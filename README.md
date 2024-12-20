# EmotionViA: Emotion Recognition from Videos via Attention-Enhanced CNN-LSTM Architecture

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

# Overview

This paper presents EmotionViA, a real-time framework for recognizing emotions from facial expressions in videos. It combines CNNs for spatial feature extraction, LSTMs for capturing temporal dynamics, and an attention mechanism to integrate these features effectively. The EmotionViA dataset, featuring diverse emotional expressions, enhances the framework's performance and adaptability. Experiments on EmotionViA, FER-2013, AffectNet, and RAF-DB datasets show that it surpasses state-of-the-art methods. EmotionViA has wide applications in fields like education, healthcare, entertainment, and marketing, improving user experiences.

# 👁️💬 Architecture

This section outlines the methodology for facial emotion recognition from videos, integrating CNNs, LSTMs, and an attention mechanism for enhanced performance. CNNs extract key facial features from video frames, while LSTMs capture temporal dependencies to track evolving emotions. An intra-modal attention mechanism highlights relevant facial cues, supported by tensor construction, tensor train layers for efficient high-dimensional processing, and dense layers for pattern refinement. This approach ensures accurate recognition by focusing on expressive facial regions like the eyes, nose, and mouth.

<img style="max-width: 100%;" src="https://github.com/swerizwan/emopoi/blob/main/resources/architecture.png" alt="EMOPOI Overview">

# Workflow

To get started, ensure you have the necessary libraries installed:

1. TensorFlow
2. OpenCV
3. Numpy
4. Dlib
5. EmoPOI files
6. GUI (Tkinter)

# Installation

1. Install TensorFlow:
   ```
   pip install tensorflow
   ```

2. Install OpenCV:
   ```
   pip install opencv-python
   ```

3. Install Numpy:
   ```
   pip install numpy
   ```

4. **Option 1:** Download the Dlib and EmoPOI files as external zip files and extract them into the main project directory/folder where your project files are stored.

   **Option 2:** If you're using PyCharm IDE, you can install the Dlib library directly in the project settings by navigating to Configure Interpreter and adding your required libraries from there.

# Datasets

- **FER-2013:** You can download the FER2013 dataset from [here](https://www.kaggle.com/datasets/msambare/fer2013).

- **EmoPOI:** EmoPOI dataset samples are available [here](https://drive.google.com/file/d/1TtJNkrWSFkIMW72-xnpBuRD7LTuxSEal/view?usp=sharing). The complete EmoPOI dataset will be available upon request for research purposes after the paper is accepted.

- **AffectNet:** You can download the AffectNet dataset from [here](http://mohammadmahoor.com/affectnet/).

- **RAF-DB:** You can download the RAF-DB dataset from [here](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset).

# Steps to Run the Demo

1. Download this repository.
2. Download the `emopoi_files` folder from [here](https://drive.google.com/drive/folders/13aIqYTp4tY5NiusXMcKxyNWiWDp80irP?usp=sharing) and put it in the root directory.
3. Create an 'Images' folder in your project and make subfolders for emotions such as Happy, Worried, Sad, Excited, Exhausted, Frustrated, Bored, and Neutral.
4. Utilize the `video_frames.py` file to convert your own or any live video into frames of images, thereby generating a large dataset.
5. Place the `image_augmentation.py` and `emopoi_frontalface_alt.xml` files in each type of image folder. For example, place these files in the "happy" image folder and execute the program. It will detect faces from images, convert them into grayscale, and create new images in the same folder.
6. Next, create the model. You can copy the code from the `training_inputs.txt` file, open the terminal in your project folder, paste the code, and hit enter.
7. Training the model will take approximately 20-25 minutes to complete. Use a large number of datasets for optimal accuracy.
8. Upon training completion, two files named `emopoi_retrained_graph.pb` and `emopoi_retrained_labels.txt` will be generated.
9. Finally, run `emotion_age_gender_recognition.py` (provide the proper path to your video). Based on the recognized emotion, gender, and age group, you will receive personalized POI recommendations suited to the user's preferences.
