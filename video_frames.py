import cv2
import os


# Add the path to the Graphviz binary to the system environment variable
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

# Define the path to the input video and the output directory for extracted frames
video_path = 'EmoPOI_V/riz.mp4'
output_path = 'EmoPOI_I/Happy'

# Create the output directory if it does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Open the video file for reading
cap = cv2.VideoCapture(video_path)

# Initialize an index counter for the extracted frames
index = 0

# Loop through each frame in the video
while cap.isOpened():
    # Read the next frame from the video
    Ret, Mat = cap.read()

    # Check if the frame was successfully read
    if Ret:
        # Increment the index counter
        index += 1

        # Skip frames that are not multiples of 9
        if index % 9 != 0:
            continue

        # Save the current frame as a PNG image in the output directory
        cv2.imwrite(output_path + '/' + str(index) + '.png', Mat)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object
cap.release()
