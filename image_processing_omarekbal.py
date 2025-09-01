import cv2
import numpy as np
import matplotlib.pyplot as plt
ideal_img = cv2.imread(r"D:\alex_imageprocessing\ideal.jpg", cv2.IMREAD_GRAYSCALE)
sample_img = cv2.imread(r"D:\alex_imageprocessing\sample2.jpg", cv2.IMREAD_GRAYSCALE)
if ideal_img is None:
    print("❌ Failed to load ideal.png")
if sample_img is None:
    print("❌ Failed to load sample2.png")

ideal_blur = cv2.GaussianBlur(ideal_img, (5, 5), 0)
sample_blur = cv2.GaussianBlur(sample_img, (5, 5), 0)
_, ideal_thresh = cv2.threshold(ideal_blur, 100, 255, cv2.THRESH_BINARY_INV)
_, sample_thresh = cv2.threshold(sample_blur, 100, 255, cv2.THRESH_BINARY_INV)
diff = cv2.absdiff(ideal_thresh, sample_thresh)
kernel = np.ones((3, 3), np.uint8)
diff_cleaned = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
plt.imshow(diff_cleaned, cmap='gray')
plt.title("Difference Image (Defects Highlighted)")
contours, _ = cv2.findContours(diff_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
broken = 0
worn = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue  

    circularity = 4 * np.pi * (area / (perimeter ** 2))

    
    if area > 100 and circularity < 0.8:  
        if area > 350:
            broken += 1
        else:
            worn += 1

print("🦷 Broken Teeth:", broken)
print("🦷 Worn Teeth  :", worn)

ideal_inv = cv2.bitwise_not(ideal_thresh)
sample_inv = cv2.bitwise_not(sample_thresh)
diff_inner = cv2.absdiff(ideal_inv, sample_inv)


_, diff_inner_thresh = cv2.threshold(diff_inner, 30, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(diff_inner_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


output_img = cv2.cvtColor(diff_inner_thresh, cv2.COLOR_GRAY2BGR)
h, w = diff_inner_thresh.shape
center_x, center_y = w // 2, h // 2
max_dist = 70  

found_near_center = False
sample_larger = False
sample_smaller = False

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    dist_to_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    if circularity > 0.7 and area > 50 and dist_to_center < max_dist:
        found_near_center = True
        cv2.drawContours(output_img, [cnt], -1, (0, 255, 0), 2)
        mask = np.zeros_like(diff_inner_thresh)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        ideal_mean = cv2.mean(ideal_inv, mask=mask)[0]
        sample_mean = cv2.mean(sample_inv, mask=mask)[0]

        if sample_mean > ideal_mean:
            sample_larger = True
        elif sample_mean < ideal_mean:
            sample_smaller = True
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title("Difference in Inner Hole Area (Ideal vs Sample)")
plt.axis('off')
plt.show()
if not found_near_center:
    print("✅ Inner diameter is IDENTICAL (no difference near center).")
elif sample_larger and not sample_smaller:
    print("⚠️ Sample gear has a smaller inner diameter than ideal.")
elif sample_smaller and not sample_larger:
    print("⚠️ Sample gear has a larger inner diameter than ideal.")
else:
    print("⚠️ Inner diameter CHANGED, but contains both larger and smaller regions.")

