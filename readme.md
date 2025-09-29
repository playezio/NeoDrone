# NEODrone Tools

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
**Official utility scripts for the NEODrone dataset** — a large-scale, multi-modal, low-altitude drone vision dataset for aerial object detection, scene understanding, and cross-domain perception.


> 🌐 **Data Access**: [Science Data Bank - NEODrone](https://www.scidb.cn) (10.57760/sciencedb.28912)(application required for raw data)

---

## 📦 Overview

This repository provides a suite of Python utilities to **process, validate, visualize, and convert** the NEODrone dataset. The tools support:

- Adaptive keyframe extraction based on flight dynamics  
- Multi-modal (visible + infrared) consistency verification  
- Automated image quality assessment  
- Pascal VOC → COCO annotation conversion  
- Metadata-based dataset filtering  
- Annotation visualization  

All scripts are designed to **reproduce the data pipeline** described in Sections 5.2–5.5 of the NEODrone paper.

---

## 🗂️ Repository Structure

```bash
NEODrone-Tools/
├── img_cut.py              # Adaptive keyframe extraction using speed & height
├── light-red.py            # Visible-infrared pair consistency validation
├── quality_check.py        # Image quality screening (sharpness, exposure, blur, noise)
├── voc2coco.py             # Convert VOC XML annotations to COCO JSON
├── visualize.py            # Visualize bounding boxes and class labels
├── metadata_parser.py      # Filter dataset subsets by metadata (height, weather, scene, etc.)
├── class_max_choose.py        # Select images from the most frequent class
├── class_number_sum.py        # Count total instances per class across all images
├── draw_box_new.py            # Visualize bounding boxes and class labels
├── object_number_sum.py       # Count total number of annotated objects
├── xml2txt-nochange.py        # Convert VOC XML annotations to YOLO-compatible .txt format
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Prerequisites
- Python ≥ 3.8
- OpenCV (with SIFT support)
- pandas, numpy, lxml, openpyxl, tqdm

### Install Dependencies
```bash
pip install opencv-contrib-python pandas numpy lxml openpyxl tqdm
```

> 💡 **Note**: `opencv-contrib-python` is required for SIFT (used in `light-red.py`). Do **not** install `opencv-python` simultaneously.

---

## 🛠️ Usage Examples

### 1. **Extract Keyframes Based on Flight Logs**
```bash
python img_cut.py \
  --video ./raw/mission_01.mp4 \
  --log ./logs/mission_01.json \
  --output_dir ./keyframes/mission_01
```
> Flight log must contain `timestamp`, `height` (m), and `speed` (m/s).

---

### 2. **Validate Visible-Infrared Pair Consistency**
```bash
python light-red.py \
  --light_dir ./Images/light \
  --red_dir ./Images/red \
  --ann_dir ./Annotations \
  --keep_consistent_only
```
- Enforces **<10ms time sync** and **≥150 SIFT matches**
- Removes inconsistent pairs to `rejected/` subdirs

---

### 3. **Batch Quality Check (Reproduce 5.2.2 Cleaning)**
```bash
python quality_check.py \
  --input_dir ./Images/light \
  --output_csv quality_report.csv \
  --laplacian_threshold 85 \
  --brightness_low 30 \
  --brightness_high 220 \
  --blur_threshold 0.25
```

---

### 4. **Convert VOC → COCO for MMDetection / Detectron2**
```bash
python voc2coco.py \
  --annotation_dir ./Annotations \
  --image_dir ./Images/light \
  --output_json ./coco/train.json
```

---

### 5. **Filter Subsets by Metadata**
```bash
# List unique metadata values
python metadata_parser.py --metadata_dir ./Metadata --list_unique

# Extract "urban + sunny + height 80–120m" subset
python metadata_parser.py \
  --metadata_dir ./Metadata \
  --scene 城市 \
  --weather 晴天 \
  --min_height 80 \
  --max_height 120 \
  --output_csv urban_sunny_80_120.csv
```

---

### 6. **Visualize Annotations**
```bash
# Preview 5 random images
python visualize.py \
  --image_dir ./Images/light \
  --annotation_dir ./Annotations \
  --num_samples 5

# Save visualization of a specific image
python visualize.py \
  --image_dir ./Images/light \
  --annotation_dir ./Annotations \
  --image_name Hebei_Urban_Sunny_AM_01_00045.jpg \
  --output_dir ./vis/
```

---

## 📚 Dataset Organization (Expected Input)

Your NEODrone dataset should follow this structure:
```
NEODrone/
├── Images/
│   ├── light/          # Visible images (.jpg)
│   └── red/            # Infrared images (.jpg)
├── Annotations/        # Pascal VOC XML files
└── Metadata/           # .xlsx files (one per video segment)
```

> File naming: `Hebei_Urban_Sunny_AM_01_00045.jpg` ↔ `Hebei_Urban_Sunny_AM_01_00045.xml`

---

## 📝 Citation

If you use NEODrone or these tools in your research, please cite:

coming soon

---

## 📜 License

This code is released under the [MIT License](LICENSE).  
The NEODrone dataset is available via [Science Data Bank](https://www.scidb.cn) under a restricted-access policy.

---

## 🙏 Acknowledgements

- Built on DJI Mavic 3T platform with hardware-level visible-infrared synchronization  
- Inspired by best practices from VisDrone, DroneVehicle, and UAVDT  
- Tools designed to support **reproducible, transparent, and responsible** aerial vision research

--- 

> ✨ **Contribution Welcome!** If you extend these tools (e.g., add YOLO format support, integrate with Hugging Face Datasets), feel free to open a PR!