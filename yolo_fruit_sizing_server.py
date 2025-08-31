#Mango Fruit Sizing using YOLOv8 Segmentation model
#Note: Run this server first before running the client script. No API security implemented as the server is intended to run locally.
#Requires: pip install fastapi uvicorn ultralytics opencv-python numpy


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import csv
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

##Images Folder - "D:\2024-03-05\images"
app = FastAPI(title="Fruit sizing processing pipeline ")
# mango_seg_model = "yolov8s_mango-seg.pt"
mango_seg_model = "yolov8m_mango-seg.pt"
# Load seg model
model = YOLO(mango_seg_model, task = "segment").to("cuda")# if 1 else "cpu")

# Output dirs
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
CSV_FILE = Path("output/results"+datetime.now().strftime("%Y%m%d_%H%M%S")+ ".csv")

# Defining filtering thresholds
MIN_AREA = 200
MAX_AREA = 10000
ELLIPSE_RATIO_THRESH = 0.85
MAX_ECCENTRICITY = 0.80
MIN_DEPTH = 400
MAX_DEPTH = 2500

# Preparing CSV headers
# if not CSV_FILE.exists():
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "file","fruit_no", "height_pixel", "width_pixel",
        "height_real", "width_real", "masked_area",
        "ellipse area", "msk_e_area_ratio", "ecc"
    ])

def compute_eccentricity(ellipse):
    (x, y), (MA, ma), _ = ellipse #skipping angle
    a = max(MA, ma) / 2.0
    b = min(MA, ma) / 2.0
    return np.sqrt(1 - (b ** 2 / a ** 2)) if a > 0 else 1.0

@app.post("/prediction")
async def prediction(rgb: UploadFile = File(...), depth: UploadFile = File(...)):
    # Load image
    rgb_bytes = await rgb.read()
    if rgb_bytes is not None:
        depth_bytes = await depth.read()
    np_rgb = np.frombuffer(rgb_bytes, np.uint8)
    np_depth = np.frombuffer(depth_bytes, np.uint8)
    depth_image_orig = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)
    depth_image = cv2.rotate(depth_image_orig, cv2.ROTATE_90_COUNTERCLOCKWISE) if depth_image_orig is not None else None
    rgb_image_orig = cv2.imdecode(np_rgb, cv2.IMREAD_COLOR)
    rgb_image = cv2.rotate(rgb_image_orig, cv2.ROTATE_90_COUNTERCLOCKWISE) if rgb_image_orig is not None else None

    # Run YOLO segmentation
    results = model.predict(rgb_image, imgsz=(1920, 1088), conf=0.5, iou=0.45, device= 0)
    # masks = results[0].masks
    masks = results[0].masks.data.cpu().numpy()
    
    annotated = rgb_image.copy()
    output_data = []
    fruit_id = 0
    # logging.info(f"MASKS: {masks[0]}")
    if masks is not None:
        logging.info(f"No detections in {rgb.filename}, skipping...")
        for mask in masks:
            mask = (mask > 0.5).astype(np.uint8)
            depth_val = None
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours)< 5:
                continue
            cnt = max(contours, key=cv2.contourArea)
            mask_area = cv2.contourArea(cnt)
            if depth_image is not None:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    depth_val = int(depth_image[cy, cx])
                    if not (MIN_DEPTH <= depth_val <= MAX_DEPTH):
                        continue
                    logging.info(f"DEPTH_VALUE: {depth_val}")
            if not (MIN_AREA <= mask_area <= MAX_AREA):
                continue

            ellipse = cv2.fitEllipse(cnt)
            ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
            ellipse_ratio = mask_area / ellipse_area if ellipse_area > 0 else 0
            eccentricity = compute_eccentricity(ellipse)

            if ellipse_ratio < ELLIPSE_RATIO_THRESH:
                continue
            if eccentricity < MAX_ECCENTRICITY:
                continue
            
            # extracting mask enclosing bounding box size (considering upright rect for mango only)
            x, y, w, h = cv2.boundingRect(cnt)

            # Saving results
            fruit_id += 1
            output_data.append({
                "filename": rgb.filename,
                "fruit_id": fruit_id,
                "height_px": int(h),
                "width_px": int(w),
                "h_real": round(h * depth_val / 960, 2) if depth_val else None,
                "w_real": round(w * depth_val / 960, 2) if depth_val else None,
                "mask_area": float(mask_area),
                "ellipse_area": float(ellipse_area),
                "ellipse_ratio": float(ellipse_ratio),
                "eccentricity": float(eccentricity)
            })

            # Draw on output image
            cv2.putText(annotated, str(fruit_id), (x-10, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.ellipse(annotated, ellipse, (255, 0, 0), 2)

    # Saving output image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_img = OUTPUT_DIR / f"{Path(rgb.filename).stem}_{timestamp}.png"
    cv2.imwrite(str(out_img), annotated)

    # Write result data to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for r in output_data:
            writer.writerow([
                r["filename"], r["fruit_id"],
                r["height_px"], r["width_px"], r["h_real"], r["w_real"], r["mask_area"],
                r["ellipse_area"], r["ellipse_ratio"], r["eccentricity"]
            ])
    logging.info(f"Detections: {output_data}")
    logging.info(f"Saved annotated image to {out_img}")

    return JSONResponse({
        "filename": rgb.filename,
        "saved_image": str(out_img),
        "results": output_data
    
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
