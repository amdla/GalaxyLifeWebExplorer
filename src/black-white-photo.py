import cv2
from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "../screenshots/opened_galaxy.png"  # Path to the provided image file
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")


# Function to extract text using Tesseract
def extract_text(image):
    custom_config = r'--oem 1 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)


# Function to enhance brightness
def enhance_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image


# Enhance brightness for the purple text
enhanced_image = enhance_brightness(image, value=50)

# Convert to grayscale
gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to get a binary image
binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Define regions for each player
regions = {
    "player_1": (32, 90, 104, 108),
    "player_2": (116, 90, 188, 108),
    "player_3": (200, 90, 272, 108),
    "player_4": (284, 90, 356, 108),
    "player_5": (368, 90, 440, 108),
    "player_6": (452, 90, 524, 108),
    "player_7": (32, 206, 104, 224),
    "player_8": (116, 206, 188, 224),
    "player_9": (200, 206, 272, 224),
    "player_10": (284, 206, 356, 224),
    "player_11": (368, 206, 440, 224),
    "player_12": (452, 206, 524, 224),
}

# Extract text from each region
extracted_texts = {}
for player, (x1, y1, x2, y2) in regions.items():
    # Crop the region from the binary image
    region = binary_image[y1:y2, x1:x2]

    # Convert to PIL Image for Tesseract
    pil_image = Image.fromarray(region)

    # Extract text
    text = extract_text(pil_image).strip()
    extracted_texts[player] = text

    # Optional: Display each region
    plt.figure(figsize=(5, 3))
    plt.imshow(region, cmap='gray')
    plt.title(f"{player}: {text}")
    plt.axis('off')
    plt.show()

# Output the extracted text
extracted_texts
