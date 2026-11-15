"""
ZENITHFORGE - Visual AI Pipeline
Extracts physical features from microgravity flame images.
"""
import cv2
import numpy as np
import os

def analyze_flame_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not read image"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"detected": False, "flame_area_px": 0, "brightness": float(np.mean(gray))}
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = float(w) / h if h > 0 else 0
    return {
        "detected": True,
        "flame_area_px": int(area),
        "aspect_ratio": round(aspect_ratio, 2),
        "brightness": round(float(np.mean(gray)), 1),
        "bounding_box": (x, y, w, h),
    }

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    path = 'data/processed/mock_flame.jpg'
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 50, (255, 100, 0), -1)
    cv2.imwrite(path, img)
    print(f"Generated mock image at {path}")
    print("Analyzing visual features...")
    print(analyze_flame_image(path))
