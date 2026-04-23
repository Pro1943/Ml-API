import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import normalize_landmarks

# --- CONFIGURATION ---
IMAGE_DIR = "dataset/images"
OUTPUT_CSV = "dataset/landmark/asl_landmarks.csv"
MODEL_PATH = 'assets/hand_landmarker.task'
# ---------------------

def convert():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: {IMAGE_DIR} not found.")
        return

    # Initialize MediaPipe Tasks Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE
    )

    landmark_data = []
    total_images = 0
    skipped_no_hand = 0
    # C.1.7: Track corrupt images separately so we know data quality
    skipped_corrupt = 0

    # Use 'with' to ensure the detector closes properly
    with vision.HandLandmarker.create_from_options(options) as detector:
        class_folders = [f for f in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, f))]
        print(f"Found {len(class_folders)} classes.")

        for class_name in class_folders:
            class_path = os.path.join(IMAGE_DIR, class_name)
            images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Processing Class '{class_name}': {len(images)} images...")
            
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                total_images += 1

                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    # C.1.7: Log every unreadable image — silent skips hide data quality issues
                    print(f"  [WARN] Could not read image (corrupt/missing): {img_path}")
                    skipped_corrupt += 1
                    continue

                # Convert BGR to RGB (MediaPipe requirement)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to MediaPipe Image Object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Process with MediaPipe Tasks
                results = detector.detect(mp_image)

                if results.hand_landmarks:
                    # results.hand_landmarks is a list of hands; we take the first [0]
                    hand_landmarks = results.hand_landmarks[0]
                    
                    # Convert landmarks to list of dicts
                    raw_list = [{"x": l.x, "y": l.y, "z": l.z} for l in hand_landmarks]
                    
                    # Normalize using your shared logic
                    normalized = normalize_landmarks(raw_list)
                    
                    # Append features + label
                    landmark_data.append(list(normalized) + [class_name])
                else:
                    skipped_no_hand += 1

            print(f"Done with '{class_name}'. Current Success: {len(landmark_data)} | No-hand skips: {skipped_no_hand} | Corrupt: {skipped_corrupt}")

    # Save to CSV
    if landmark_data:
        cols = [f"feat_{i}" for i in range(63)] + ["label"]
        df = pd.DataFrame(landmark_data, columns=cols)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFinal Success!")
        print(f"Processed {total_images} total images.")
        print(f"Saved {len(landmark_data)} valid samples to {OUTPUT_CSV}")
    else:
        print("No landmarks were detected. Check if hand_landmarker.task is in the correct folder.")

if __name__ == "__main__":
    convert()
