import cv2
import numpy as np
import os

MOUNT_POINT = os.path.join(os.environ["HOME"], "bushfire")

image_path = os.path.join(
    MOUNT_POINT,
    "u5155914/models/Paint-by-Example/background_images/background.jpg",
)
image = cv2.imread(image_path)

mask = np.zeros((512, 512), dtype=np.uint8)
# cv2.rectangle(image, start_point, end_point, color, thickness)
mask = cv2.rectangle(mask, (190, 240), (270, 310), (255, 255, 255), -1)

mask_rgb = np.zeros_like(image)
mask_rgb[:, :, 0] = mask
mask_rgb[:, :, 1] = mask
mask_rgb[:, :, 2] = mask
save_img = os.path.join(
    MOUNT_POINT,
    "u5155914/models/Paint-by-Example/background_images/mask_small.jpg",
)
cv2.imwrite(save_img, mask_rgb)

mask_rgb = 255 - mask_rgb
mask_rgb[mask_rgb < 128] = 0
mask_rgb[mask_rgb >= 128] = 255
# Mask input image with binary mask
print(image.shape, mask_rgb.shape)
result = cv2.bitwise_and(image, mask_rgb)
sample_path = os.path.join(
    MOUNT_POINT,
    "u5155914/models/Paint-by-Example/background_images/small_masked_image.jpg",
)
cv2.imwrite(sample_path, result)
