# NEODrone Dataset Processing Pipeline

## Reproducibility Statement

**Code Version:** v1.0.0  
**Dataset Version:** NEODrone v2.1

### Input Specifications
- **Original Video Path Format:** `{source_dir}/{flight_id}/{sensor_type}_{timestamp}.mp4`
  - Example: `raw_videos/flight_001/visible_20240101_120000.mp4`
- **SDK Log Structure:** JSON format with `/telemetry/*` and `/camera/*` endpoints
- **Supported Sensor Types:** visible (RGB), thermal (infrared), multispectral

### Output Specifications
- **Directory Structure:** Matches publicly released NEODrone dataset exactly
- **File Naming Convention:** `{flight_id}_{frame_index:06d}.jpg` (RGB), `{flight_id}_{frame_index:06d}_thermal.png` (IR)
- **Metadata Format:** Structured `.xlsx` with 15 standardized fields
- **Annotation Format:** YOLO format `.txt` files with 28-class taxonomy

### Validation
- **Verification Script:** `scripts/verify_output.py` validates:
  - Output frame count matches extraction protocol
  - Metadata field completeness (15/15 fields)
  - RGB-IR pair alignment (≤5 pixels deviation)
  - Annotation consistency (α coefficient ≥0.8)

---

## Project Structure

```
NEODrone/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements*.txt         # Module-specific dependencies
├── configs/
│   ├── extraction_config.yaml
│   └── cleaning_thresholds.yaml
├── src/
│   ├── extraction/           # Data extraction scripts
│   ├── cleaning/             # Cleaning and quality control scripts
│   ├── annotation/           # Annotation assistance and consistency
│   └── metadata/             # Metadata parsing and validation
├── weights/                  # Pre-trained/fine-tuned model weights
├── docs/
│   ├── API_endpoints.md      # DJI SDK endpoint documentation
│   └── annotation_guidelines.pdf  # Annotation specifications
└── scripts/
    ├── run_full_pipeline.sh  # One-click reproduction script
    └── generate_dataset_card.py  # Dataset card generator
```

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- FFmpeg (for video processing)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/NEODrone.git
cd NEODrone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_extraction.txt
pip install -r requirements_annotation.txt
pip install -r requirements_metadata.txt
```

---

## Usage

### Quick Start

```bash
# Run full pipeline
bash scripts/run_full_pipeline.sh

# Or run individual modules
python src/extraction/extract_frames.py --input raw_videos/ --output frames/
python src/cleaning/clean_pipeline.py --input frames/ --output cleaned/
python src/annotation/train_rtdetr.py --data data.yaml
python src/metadata/validate_schema.py --metadata metadata.xlsx
```

### Module-Specific Usage

#### 1. Data Extraction
```bash
python src/extraction/extract_frames.py \
    --input raw_videos/ \
    --output frames/ \
    --config configs/extraction_config.yaml
```

#### 2. Multi-modal Synchronization
```bash
python src/extraction/sync_extract.py \
    --visible frames/visible/ \
    --thermal frames/thermal/ \
    --output synced/
```

#### 3. Metadata Parsing
```bash
python src/extraction/parse_metadata.py \
    --sdk_logs logs/ \
    --exif_data frames/ \
    --output metadata.xlsx
```

#### 4. Data Cleaning
```bash
python src/cleaning/clean_pipeline.py \
    --input frames/ \
    --output cleaned/ \
    --config configs/cleaning_thresholds.yaml
```

#### 5. Annotation Training
```bash
python src/annotation/train_rtdetr.py \
    --data data.yaml \
    --epochs 100 \
    --batch 16 \
    --output weights/rtdetr_neodrone.pt
```

#### 6. Annotation Inference
```bash
python src/annotation/inference_rtdetr.py \
    --weights weights/rtdetr_neodrone.pt \
    --input cleaned/ \
    --output annotations/
```

---

## Configuration

### Extraction Parameters (`configs/extraction_config.yaml`)
```yaml
sampling:
  strategy: "adaptive"  # adaptive, fixed, scene_change
  interval: 1.0  # seconds
  overlap_threshold: 0.7
  
quality:
  min_resolution: [1920, 1080]
  min_sharpness: 0.5
  max_blur_score: 0.3
```

### Cleaning Thresholds (`configs/cleaning_thresholds.yaml`)
```yaml
detection:
  model: "yolov8x"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  
thermal:
  anomaly_threshold: 0.05  # 5% abnormal pixels
  dead_pixel_threshold: 100
  
alignment:
  max_deviation: 5  # pixels
  min_overlap: 0.8
```

---

## Dataset Card

Generate a dataset card compatible with Hugging Face / ScienceDB:

```bash
python scripts/generate_dataset_card.py \
    --metadata metadata.xlsx \
    --output dataset_card.md
```

---

## Verification

Validate output quality and consistency:

```bash
python scripts/verify_output.py \
    --expected_frames 50000 \
    --metadata metadata.xlsx \
    --annotations annotations/
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{neodrone2024,
  title={NEODrone: A Multi-modal Drone Dataset for Object Detection},
  author={[Author Names]},
  journal={[Journal]},
  year={2024},
  volume={[Volume]},
  number={[Number]},
  pages={[Pages]}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- YOLOv8x model weights from [Ultralytics](https://github.com/ultralytics/ultralytics)
- RT-DETR implementation from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- Krippendorff's alpha calculation from [krippendorff package](https://github.com/pln-fing-udelar/krippendorff)

---

## Contact

For questions or issues, please open an issue on GitHub or contact [maintainer@email.com].

---

## Third-Party Dependencies

- **YOLOv8x:** Pre-trained weights from Ultralytics (AGPL-3.0 License)
- **RT-DETR:** Official implementation from PaddleDetection (Apache 2.0 License)
- **OpenCV:** BSD-3 License
- **PyTorch:** BSD-style License

See individual `requirements_*.txt` files for complete dependency lists with version specifications.