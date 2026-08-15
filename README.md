# 🔍 DefectVision — AI-Powered Industrial Defect Segmentation

DefectVision is a deep-learning-based industrial inspection system for **pixel-level defect segmentation** in RGB inspection images.

The project uses a custom **MS-Residual U-Net** architecture trained and evaluated on the **MVTec AD-SEG** dataset. The final system generates defect masks, probability maps, visual overlays, and defect-area estimates through an interactive Streamlit application.

**Live Demo:** YOUR_STREAMLIT_URL

---

## 📌 Overview

Traditional defect classification answers:

> Is this image defective?

DefectVision goes further and answers:

> **Which pixels correspond to the defect, and what is the spatial extent of the defect?**

### Pipeline

```text
RGB Inspection Image
        │
        ▼
Image Preprocessing
        │
        ▼
MS-Residual U-Net
        │
        ▼
Defect Probability Map
        │
        ▼
Thresholding
        │
        ▼
Pixel-Level Defect Mask
        │
        ├──► Defect Area
        ├──► Visual Overlay
        └──► Probability Map
```

---

## 🎯 Objectives

- Perform pixel-level industrial defect segmentation.
- Handle severe foreground/background class imbalance.
- Develop a stronger segmentation architecture than a basic baseline.
- Compare different training objectives through controlled experiments.
- Evaluate performance on a held-out test set.
- Analyze performance across different defect sizes.
- Build an interactive inference application.
- Deploy the final model as a web application.

---

## 📊 Dataset

The project uses the **MVTec AD-SEG** industrial anomaly segmentation dataset.

The dataset contains industrial inspection images with corresponding pixel-level defect annotations.

The dataset itself is **not included in this repository**.

---

# 🧠 Model Architecture

## MS-Residual U-Net

The final model is a custom U-Net-style architecture combining:

- Residual convolutional blocks
- Encoder-decoder architecture
- Skip connections
- Multi-scale feature extraction
- Multi-scale bottleneck processing
- Pixel-level prediction

### Architecture Overview

```text
                    Encoder
                       │
Input ──► Residual ──► Residual ──► Residual ──► Residual
          Block          Block          Block          Block
             │             │              │              │
             └─────────────┴──────────────┴──────────────┘
                              │
                     Multi-Scale Bottleneck
                              │
                              ▼
                           Decoder
                              │
                              ▼
                    Residual Refinement
                              │
                              ▼
                     Defect Probability
                              │
                              ▼
                       Binary Mask
```

---

# ⚖️ Training Strategy

The initial training objective used:

**Weighted Binary Cross Entropy + Dice Loss**

The weighting was motivated by the strong foreground/background class imbalance in the segmentation task.

### Dataset Pixel Statistics

| Statistic | Value |
|---|---:|
| Total pixels | 57,671,680 |
| Defect pixels | 2,469,565 |
| Defect fraction | 4.28% |
| Positive class weight | 20.0 |

---

# 🧪 Experimental Development

The project was developed through multiple controlled experiments rather than relying on a single training run.

## Baseline

The initial baseline achieved:

| Metric | Score |
|---|---:|
| Dice | 0.2070 |
| IoU | 0.1434 |
| Precision | 0.3136 |
| Recall | 0.5400 |

---

## 🚀 Improved Model

The MS-Residual U-Net substantially improved segmentation performance.

### Model 06 — Full Loss

| Metric | Score |
|---|---:|
| Dice | 0.3533 |
| IoU | 0.2528 |
| Precision | 0.3253 |
| Recall | 0.6822 |

### Improvement over baseline

- Dice: **+70.69%**
- IoU: **+76.28%**

---

# 🔬 Controlled Ablation Study

A controlled ablation was performed to investigate the contribution of the boundary-related loss component.

Two variants were compared:

1. Full loss
2. No-boundary loss

### Validation Results

| Variant | Best Validation Dice | Best Validation IoU |
|---|---:|---:|
| Full loss | 0.4237 | 0.3197 |
| **No boundary** | **0.4270** | **0.3178** |

The **no-boundary variant** was selected based on validation performance.

The held-out test set was not used for model selection.

---

# 🏆 Final Model

### Selected Configuration

```text
Architecture:          MS-Residual U-Net
Selected variant:      No-boundary loss
Best validation epoch: 8
```

## Held-Out Test Performance

| Model | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline | 0.2070 | 0.1434 | 0.3136 | 0.5400 |
| Model 06 — Full Loss | 0.3533 | 0.2528 | 0.3253 | 0.6822 |
| **Final — No Boundary** | **0.3753** | **0.2758** | **0.3484** | **0.6988** |

### Improvement over Baseline

- Dice improvement: **81.31%**
- IoU improvement: **92.29%**

The final model substantially improves segmentation overlap and defect recall compared with the baseline.

---

# 📐 Defect-Size Analysis

Performance was further analyzed according to defect size.

| Size Group | Images | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Small | 95 | 0.2109 | 0.1368 | 0.1548 | 0.6385 |
| Medium | 77 | 0.3717 | 0.2641 | 0.3083 | 0.7428 |
| Large | 81 | **0.5715** | **0.4500** | **0.6136** | 0.7277 |

### Observation

The model performs substantially better on medium and large defects than on small defects.

Small defects occupy fewer pixels and are therefore more difficult to localize accurately.

---

# 🔍 Qualitative Evaluation

The project includes qualitative analysis of:

- Best predictions
- Typical predictions
- Hardest examples
- Small defects
- Medium defects
- Large defects
- High false-positive examples
- High false-negative examples

This analysis helps identify failure modes that are not captured by aggregate metrics alone.

---

# 🖥️ Streamlit Application

DefectVision includes an interactive Streamlit application for model inference.

### Features

- Upload PNG, JPG, JPEG, BMP, or WEBP images
- Adjustable segmentation threshold
- Defect / no-defect indication
- Predicted defect area
- Mean defect probability
- Maximum probability
- Pixel-level defect mask
- Probability map
- Visual defect overlay
- Downloadable predicted mask

### Application Workflow

```text
Upload Image
     │
     ▼
Preprocessing
     │
     ▼
MS-Residual U-Net
     │
     ▼
Probability Map
     │
     ▼
Thresholding
     │
     ├──► Defect Status
     ├──► Defect Area
     ├──► Confidence
     ├──► Defect Mask
     └──► Visual Overlay
```

---

# 📈 Example Inference

For a representative test image, the deployed application produced:

```text
Defect Status       : DETECTED
Defect Area         : 12.39%
Mean Confidence     : 86.1%
Maximum Probability : 99.0%
```

The application generates:

```text
Input Image
      │
      ├──► Predicted Defect Overlay
      ├──► Binary Defect Mask
      └──► Defect Probability Map
```

---

# 📁 Project Structure

```text
DefectVision/
│
├── app/
│   └── app.py
│
├── notebooks/
│
├── models/
│   └── checkpoints/
│       └── model08_no_boundary_best.pth
│
├── src/
│
├── data/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset and unnecessary generated files are excluded from version control.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/UtkarshRode/DefectVision.git
cd DefectVision
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🌐 Live Demo

**Streamlit Cloud:**

YOUR_STREAMLIT_URL

---

# 🧩 Technologies Used

### Deep Learning

- Python
- PyTorch
- NumPy
- PIL
- Matplotlib

### Computer Vision

- Image segmentation
- Pixel-level classification
- Binary masks
- Probability maps
- Defect localization

### Deployment

- Streamlit
- Git
- GitHub
- Streamlit Community Cloud

---

# ⚠️ Limitations

The model was trained and evaluated on MVTec AD-SEG.

Performance may decrease when applied to images with substantial distribution differences, including:

- Different object categories
- Different lighting conditions
- Different camera systems
- Different image resolutions
- Different defect characteristics

The application is a **research and portfolio prototype**, not a safety-critical production inspection system.

---

# 🔮 Future Improvements

Potential improvements include:

- Better handling of very small defects
- Training on additional industrial datasets
- Test-time augmentation
- More advanced threshold calibration
- Domain adaptation
- Instance-level defect analysis
- Model quantization
- ONNX/TensorRT inference
- GPU acceleration
- Production monitoring
- Cross-domain evaluation

---

# 👨‍💻 Author

**Utkarsh Rode**

IIT Kharagpur

GitHub:  
https://github.com/UtkarshRode/DefectVision

---

# 📜 License

This project is intended for educational, research, and portfolio purposes.
