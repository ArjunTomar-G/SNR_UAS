import cv2
import numpy as np
import os

# Loading all images
def load_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            images.append((filename, img))
    return images   

image_folder = '/Users/arjuntomar/Documents/uastaskimg'   #image folder location , I have entered for my case

#Or if I want to select manually my image folder then
#import tkinter as tk
#from tkinter import filedialog
#root = tk.Tk()
#root.withdraw()  
#image_folder = filedialog.askdirectory(title='Select Folder with Images')
#print(f"Selected folder: {image_folder}")

images = load_images(image_folder)

# Segment burnt and green grass using color thresholding
def segment_grass(img):
    # Convert to HSV for better color segmentation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for burnt grass (brown)
    burnt_lower = np.array([10, 100, 20])
    burnt_upper = np.array([20, 255, 200])
    burnt_mask = cv2.inRange(hsv_img, burnt_lower, burnt_upper)

    # Define the range for green grass
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_img, green_lower, green_upper)

    return burnt_mask, green_mask

# Detect houses
def detect_houses(img):
    # Assuming houses are triangular and colored red and blue
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    blue_mask = cv2.inRange(img, lower_blue, upper_blue)
    red_mask = cv2.inRange(img, lower_red, upper_red)

    return blue_mask, red_mask
def count_houses_on_grass(burnt_mask, green_mask, blue_mask, red_mask):
    # Find houses on burnt and green grass
    burnt_houses = cv2.bitwise_and(burnt_mask, blue_mask + red_mask)
    green_houses = cv2.bitwise_and(green_mask, blue_mask + red_mask)

    # Count houses on burnt grass
    num_burnt_blue = np.sum(cv2.bitwise_and(burnt_mask, blue_mask) > 0)
    num_burnt_red = np.sum(cv2.bitwise_and(burnt_mask, red_mask) > 0)
    num_green_blue = np.sum(cv2.bitwise_and(green_mask, blue_mask) > 0)
    num_green_red = np.sum(cv2.bitwise_and(green_mask, red_mask) > 0)

    # Total number of houses
    num_burnt = num_burnt_blue + num_burnt_red
    num_green = num_green_blue + num_green_red

    # Calculate priorities
    priority_burnt = num_burnt_red * 1 + num_burnt_blue * 2
    priority_green = num_green_red * 1 + num_green_blue * 2

    return num_burnt, num_green, priority_burnt, priority_green
def calculate_rescue_ratio(priority_burnt, priority_green):
    if priority_green == 0:  # Avoid division by zero
        return float('inf')  # Highest possible ratio
    return priority_burnt / priority_green

# Store results
n_houses = []
priority_houses = []
priority_ratio = []
image_names = []

for filename, img in images:
    burnt_mask, green_mask = segment_grass(img)
    blue_mask, red_mask = detect_houses(img)

    num_burnt, num_green, priority_burnt, priority_green = count_houses_on_grass(burnt_mask, green_mask, blue_mask, red_mask)
    
    ratio = calculate_rescue_ratio(priority_burnt, priority_green)
    
    n_houses.append([num_burnt, num_green])
    priority_houses.append([priority_burnt, priority_green])
    priority_ratio.append(ratio)
    image_names.append(filename)

# Sort by rescue ratio in descending order
sorted_images = [x for _, x in sorted(zip(priority_ratio, image_names), reverse=True)]
print("Number of Houses: ", n_houses)
print("House Priorities: ", priority_houses)
print("Rescue Ratios: ", priority_ratio)
print("Images sorted by rescue ratio: ", sorted_images)


#If the image has a lot of noise then we can apply
#def apply_gaussian_blur(img):
 #   return cv2.GaussianBlur(img, (5, 5), 0)  # Kernel size (5x5) can be adjusted
#def apply_median_blur(img):
 #   return cv2.medianBlur(img, 5)  # Kernel size 5 (can be changed based on noise level)



#function to increase brightness of output images in order to match the example given in task pdf
def increase_brightness(img, value=250):
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Add value to the V channel (Brightness)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255  # values stay within [0, 255]
    
    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img
# Function to generate coloured output images
def color_and_overlay(img, burnt_mask, green_mask, blue_mask, red_mask):
    # Create color overlays
    burnt_color = np.zeros_like(img)
    green_color = np.zeros_like(img)
    house_blue = np.zeros_like(img)
    house_red = np.zeros_like(img)
    
    # Assign colors: Burnt grass = yello, Green grass = cyan, Blue houses = blue, Red houses = red
    burnt_color[burnt_mask > 0] = [0, 255, 255]  # yellow (bgr)
    green_color[green_mask > 0] = [255, 255, 0]    # cyan
    house_blue[blue_mask > 0] = [255, 0, 0]      # Blue
    house_red[red_mask > 0] = [0, 0, 255]        # Red

    # Combine the masks with weighted overlay
    combined_img = cv2.addWeighted(img, 0.5, burnt_color, 0.5, 0)
    combined_img = cv2.addWeighted(combined_img, 1, green_color, 0.5, 0)
    combined_img = cv2.addWeighted(combined_img, 1, house_blue, 0.7, 0)
    combined_img = cv2.addWeighted(combined_img, 1, house_red, 0.7, 0)

    return combined_img

#to display the output images
for filename, img in images:
    # Process the image (e.g., segmentation)
    burnt_mask, green_mask = segment_grass(img)
    blue_mask, red_mask = detect_houses(img)
    
    # Color and overlay the masks on the original image
    colored_output = color_and_overlay(img, burnt_mask, green_mask, blue_mask, red_mask)
    colored_output = increase_brightness(colored_output)

    # Display the image with OpenCV
    cv2.imshow(filename, colored_output)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window when done




