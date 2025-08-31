# Mango Fruit Sizing with YOLOv8 Segmentation + RGBDepth

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com/)  
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-blue)](https://docs.ultralytics.com/)  
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange)](https://www.python.org/)  

---

## Overview
This project provides a **FastAPI-based inference server** and a **Python client** for automated fruit sizing (configured for mangoes).  
The system detects fruits in **RGB images** using a YOLOv8 segmentation model, combines them with associated **depth maps**, and outputs measurements such as:

- Fruit **height & width** in pixels and real-world units  
- **Mask area** and **ellipse-fit area**  
- **Mask-to-ellipse ratio**  
- **Ellipse eccentricity**  

Outputs include **annotated images** and a consolidated **CSV file**.

---

## Project Structure
bash```

├── yolo_fruit_sizing_server.py # FastAPI inference server

├── yolo_fruit_sizing_client.py # Client script for batch processing

├── yolov8m_mango-seg.pt # YOLOv8 segmentation model (not included)

├── output/ # Annotated images + CSV results

└── README.md # Documentation
```
---

## Requirements
Install dependencies:

```bash
pip install fastapi uvicorn ultralytics opencv-python requests
```

**(Optional)** Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

**Running the Server**
Start the FastAPI inference server (loads YOLOv8 segmentation model once):
python yolo_fruit_sizing_server.py

API endpoint: http://0.0.0.0:8000/predict
Annotated images and CSV file will be saved in output/.
Uses GPU (cuda) if available, otherwise falls back to CPU.

**Running the Client**
The client script loops through an images folder.

RGB images must be named as: color_<id>.png
Depth images must be named as: depth_<id>.png
Both must share the same <id> (e.g., color_123.png ↔ depth_123.png)

**Example usage:**
python yolo_fruit_sizing_client.py /path/to/images

The client sends each RGB+depth pair to the server, receives results, and prints them.

**Annotated images** in output/:
Bounding box
Instance ID (top-left corner)
Fitted ellipse

**Filtering Rules**
Detections are validated using thresholds defined in the server code:
MIN_AREA, MAX_AREA: valid mask area range
MIN_DEPTH, MAX_DEPTH: valid depth range
ELLIPSE_RATIO_THRESH: mask vs ellipse area ratio
MAX_ECCENTRICITY: restricts elongated detections
Only detections passing these rules are recorded.

**Security (Optional)**
Currently, the API is not secured (intended for local use).
For production use:
Add API key validation in FastAPI
Enable HTTPS
Restrict IP access

**Example Workflow**
- Collect RGB + depth image pairs in a folder.
- Run the server: python yolo_fruit_sizing_server.py
- Run the client pointing to the folder: python yolo_fruit_sizing_client.py ./images
- Check results in the output/ directory.

**Notes**
- The YOLOv8 segmentation model (e.g. yolov8m_mango-seg.pt) must be trained separately and placed in the project folder. Replace your model name in the server code.
- GPU acceleration requires CUDA-enabled PyTorch.
- Depth images must be aligned with RGB images for accurate size estimation.

**Author**
Developed as a sample project to demonstrate computer vision integration with FastAPI for inference.

##TODO
- Secure API (add API key authentication, enable HTTPS with TLS certificate, request rate limit to avoid overload).
- Add support for images location on cloud
- Pass csv to client once completed processing images
