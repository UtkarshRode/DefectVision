# DefectVision — Streamlit App

This folder contains the user-facing Streamlit application for DefectVision.

## Expected project structure

```text
DefectVision/
├── app/
│   └── app.py
├── models/
│   └── checkpoints/
│       └── model08_no_boundary_best.pth
├── data/
├── notebooks/
└── requirements.txt
```

The trained checkpoint is intentionally not included in this package.

## Install

From the DefectVision project root:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/app.py
```

The browser should open the DefectVision dashboard.

## Final model

- Architecture: MS-Residual U-Net
- Loss: Weighted BCE + Dice
- Dataset: MVTec AD-SEG
- Test Dice: 0.3753
- Test IoU: 0.2758
- Test Precision: 0.3484
- Test Recall: 0.6988

## Important limitation

The model was trained and evaluated on MVTec AD-SEG. The app should therefore be treated as a research/portfolio prototype and used on images from a compatible visual domain. Performance on unrelated industrial imagery is not guaranteed.
