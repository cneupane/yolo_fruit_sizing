# yolo_fruit_sizing_client.py
import requests
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python yolo_fruit_sizing_client.py <folder_path>")
    sys.exit(1)

url = "http://localhost:8000/prediction"  # adjust if different

folder = Path(sys.argv[1])

# Loop through color images and find matching depth
for rgb_file in folder.glob("color_*.png"):
    suffix = rgb_file.name.replace("color_", "")
    depth_file = folder / f"depth_{suffix}"

    if not depth_file.exists():
        print(f"⚠️ No matching depth file for {rgb_file.name}, skipping...")
        continue

    with open(rgb_file, "rb") as f_rgb, open(depth_file, "rb") as f_depth:
        files = {
            "rgb": (rgb_file.name, f_rgb, "image/png"),
            "depth": (depth_file.name, f_depth, "image/png"),
        }
        response = requests.post(url, files=files)

    try:
        print(f"Processed {rgb_file.name}: {response.json()}")
    except Exception:
        print(f"Error processing {rgb_file.name}: {response.text}")





