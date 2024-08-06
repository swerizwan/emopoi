import cv2
import glob
# Fetch a list of all PNG images in the current directory
images = glob.glob("*.png")

# Loop through each image in the list
for image in images:
    # Load the pre-trained Haar Cascade classifier for face detection
    facedata = "emopoi/emopoi_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    # Read the image in grayscale
    img = cv2.imread(image, 0)

    # Resize the image to its original dimensions
    re = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

    # Detect faces in the resized image
    faces = cascade.detectMultiScale(re)

    # Loop through each detected face
    for f in faces:
        # Extract coordinates of the face bounding box
        x, y, w, h = [v for v in f]

        # Draw a green rectangle around the detected face in the original image
        Rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) containing the detected face
        sub_face = img[y:y + h, x:x + w]

        # Extract the filename from the image path
        f_name = image.split('/')
        f_name = f_name[-1]

        # Display the detected face in a window named "checking"
        cv2.imshow("checking", sub_face)

        # Wait for 500 milliseconds and then close the window
        cv2.waitKey(500)
        cv2.destroyAllWindows()

        # Save the detected face as a new image with a filename prefix "resized_"
        cv2.imwrite("resized_" + image, sub_face)
