
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url

# Check GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Download sample image
download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "small-vehicles1.jpeg"
)

# Load pretrained YOLOv8 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    device=device
)

# Run SAHI sliced inference
result = get_sliced_prediction(
    "small-vehicles1.jpeg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Save results
result.export_visuals(export_dir="results/")

print("SAHI inference completed")
print("Objects detected:", len(result.object_prediction_list))

