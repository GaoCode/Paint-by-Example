import cv2

# Load image, create mask, and draw white circle on mask
image_path = "examples/noosa/1559167625_+00240.jpg"
image = cv2.imread(image_path)

mask_path = "examples/noosa/1559167625_+00240.png"
mask = cv2.imread(mask_path)
# mask = 255 - mask
mask[mask < 128] = 0
mask[mask >= 128] = 255

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask == 0] = 255  # Optional

sample_path = "examples/noosa/reference.jpg"
cv2.imwrite(sample_path, result)
