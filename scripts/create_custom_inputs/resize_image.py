import cv2

# Load image, create mask, and draw white circle on mask
image_path = "examples/noosa/0000000002.jpg"
image = cv2.imread(image_path)
resized_img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
save_img = "examples/noosa/background.jpg"
cv2.imwrite(save_img, resized_img)

mask_path = "examples/noosa/1559167625_+00240.png"
mask = cv2.imread(mask_path)
resized_mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)
save_img = "examples/noosa/mask.jpg"
cv2.imwrite(save_img, resized_mask)


resized_mask = 255 - resized_mask
resized_mask[resized_mask < 128] = 0
resized_mask[resized_mask >= 128] = 255
# Mask input image with binary mask
print(resized_img.shape, resized_mask.shape)
result = cv2.bitwise_and(resized_img, resized_mask)
# Color background white
result[resized_mask == 0] = 255  # Optional
sample_path = "examples/noosa/masked_img.jpg"
cv2.imwrite(sample_path, result)
