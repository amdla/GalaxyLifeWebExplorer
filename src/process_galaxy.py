import cv2
from PIL import Image
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "../screenshots/opened_galaxy_prep.png"  # Path to your saved image file
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")


# Function to extract text using Tesseract
def extract_text(image):
    custom_config = r'--oem 1 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config)


# Function to preprocess image by isolating the text colors
def preprocess_image_for_text(image, lower_color, upper_color):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Bitwise-AND mask and original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to grayscale
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply morphological operations to remove small noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(morph_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image, mask, filtered_image


# Define color ranges for text colors (in HSV)
lower_purple = np.array([115, 30, 30])  # Further adjustment
upper_purple = np.array([165, 255, 255])  # Further adjustment

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 20, 255])

# Define regions to crop (coordinates adjusted as per the image layout)
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

# Extract text from defined regions and annotate the image
extracted_texts = {}
for region_name, (x1, y1, x2, y2) in regions.items():
    # Ensure the coordinates are within the bounds of the image
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    # Crop the region
    cropped_image = image[y1:y2, x1:x2]

    # Preprocess for both purple and white text
    preprocessed_purple, mask_purple, filtered_purple = preprocess_image_for_text(cropped_image, lower_purple,
                                                                                  upper_purple)
    preprocessed_white, mask_white, filtered_white = preprocess_image_for_text(cropped_image, lower_white, upper_white)

    # Convert to PIL Image for Tesseract
    pil_image_purple = Image.fromarray(preprocessed_purple)
    pil_image_white = Image.fromarray(preprocessed_white)

    # Extract text
    text_purple = extract_text(pil_image_purple).strip()
    text_white = extract_text(pil_image_white).strip()

    # Combine extracted text
    extracted_texts[region_name] = f"{text_purple} {text_white}".strip()

    # Draw rectangle and put text on the original image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, extracted_texts[region_name], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Visualize the masks and filtered images for debugging
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mask_purple, cmap='gray')
    axes[0].set_title(f'{region_name} Purple Mask')
    axes[1].imshow(mask_white, cmap='gray')
    axes[1].set_title(f'{region_name} White Mask')
    axes[2].imshow(cv2.cvtColor(filtered_purple, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'{region_name} Filtered Purple Image')
    plt.show()

# Save the final annotated image
output_image_path = "../screenshots/opened_galaxy_processed.png"
cv2.imwrite(output_image_path, image)

# Display the image with annotations
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Output the extracted text from all regions
for region, text in extracted_texts.items():
    print(f"{region}: {text}")
