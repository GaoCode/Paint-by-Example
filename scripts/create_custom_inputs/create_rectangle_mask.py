import cv2
import numpy as np

image_path = "examples/figlib/1495302789_-01680.jpg"
image = cv2.imread(image_path)
resized_img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
save_img = "examples/figlib/background.jpg"
cv2.imwrite(save_img, resized_img)

mask = np.zeros((512, 512), dtype=np.uint8)
# cv2.circle(image, center_coordinates, radius, color, thickness)
# mask = cv2.circle(mask, (260, 300), 225, (255,255,255), -1)
# cv2.rectangle(image, start_point, end_point, color, thickness)
mask = cv2.rectangle(mask, (50, 50), (400, 300), (255, 255, 255), -1)

mask_rgb = np.zeros_like(resized_img)
mask_rgb[:, :, 0] = mask
mask_rgb[:, :, 1] = mask
mask_rgb[:, :, 2] = mask
save_img = "examples/figlib/mask.jpg"
cv2.imwrite(save_img, mask_rgb)

mask_rgb = 255 - mask_rgb
mask_rgb[mask_rgb < 128] = 0
mask_rgb[mask_rgb >= 128] = 255
# Mask input image with binary mask
print(resized_img.shape, mask_rgb.shape)
result = cv2.bitwise_and(resized_img, mask_rgb)
# Color background white
# result[mask == 0] = 255  # Optional
sample_path = "examples/figlib/masked_img.jpg"
cv2.imwrite(sample_path, result)
