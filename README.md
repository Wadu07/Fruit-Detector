# Fruit-Detector
Fruit Detection application using computer vision

About dataset:
After preprocessing the images, removing duplicates, deleting low-resolution images, as well as rotating and flipping images, we have concluded the preparations for model training.
The dataset is represented by three directories: training, validation, and testing.
The training set contains 3066 images.
1348 images are located in the validation set.
The testing set comprises images that are not found in the other two directories, to provide a realistic perspective for fruit detection.

About the application:
The user has the option to choose between images or videos to test.
To test a video, the user needs to check the box designated for video testing.
They can search for an image in local memory and can use the "drag and drop" method to select the image.
Upon clicking the "submit" button, the detections of the apples are displayed, the number of fruits present in the image is shown.

About the code:
I used a function in order to get a BGR image with the detection results.
If the user opts to test a video clip, real-time detection is displayed. The image is converted to RGB when there are detections found by the function, as YOLO operates with BGR images.
If the user chooses to test an image, it is initially converted to the BGR channel for the function's use, and later reconverted for normal display on the screen.
