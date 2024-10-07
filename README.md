# SNR_UAS

# Search and Rescue Image Segmentation
This project is part of UAS-DTU Round 2 for the Unmanned Aerial Systems - Delhi Technological University recruitment. The task focuses on using image segmentation and feature detection to assist in a simulated search-and-rescue mission. The goal is to process UAV-collected images, identifying and classifying various features to prioritize rescue efforts in a disaster-stricken area.

# Problem Statement
The task is to analyzing aerial images to differentiate between:

Burnt grass and green grass,
Identifying houses (marked as blue and red triangles) and their priority levels.
The key deliverable includes:

Process images and overlay two distinct colors to represent burnt and green grass.
Determine the number of houses on burnt and green grass, and calculate their priorities.
Calculate and return a # rescue ratio (priority ratio between burnt and green grass areas).
Output a sorted list of images based on their rescue ratio.
# Input and Output
# Input:
A list of 10 images similar to the provided sample image.
Image features:
Brown areas: Burnt grass
Green areas: Green grass
Red triangles: Priority 1 houses
Blue triangles: Priority 2 houses
# Expected Output:
Overlay Image: A color-enhanced version of each image, highlighting burnt and green grass areas.
House Count: A list with the number of houses on burnt grass and green grass (e.g., [[Hb, Hg]]).
Priority Scores: A list showing the total priority on burnt and green grass (e.g., [[Pb, Pg]]).
Rescue Ratio: The ratio of priorities (e.g., [Pr = Pb / Pg]).
Image Ranking: A list of image names sorted by rescue ratio.
# Approach
The solution will be developed using the following technologies:

Python3: For the core processing logic.</br>
NumPy: To handle numerical computations and array operations.</br>
OpenCV: For image processing, including loading images and applying segmentation.
