import json
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DefectVision | Industrial Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 0.50

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "model08_no_boundary_best.pth"
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1.15rem;
        opacity: 0.78;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .status-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 110px;
    }

    .status-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
    }

    .status-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }

    .disclaimer {
        padding: 0.9rem 1rem;
        border-radius: 0.65rem;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 0.9rem;
        opacity: 0.82;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()

        groups = min(8, cout)

        self.c1 = nn.Conv2d(
            cin, cout, 3, padding=1, bias=False
        )
        self.n1 = nn.GroupNorm(groups, cout)

        self.c2 = nn.Conv2d(
            cout, cout, 3, padding=1, bias=False
        )
        self.n2 = nn.GroupNorm(groups, cout)

        self.skip = (
            nn.Conv2d(cin, cout, 1, bias=False)
            if cin != cout
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        x = F.relu(
            self.n1(self.c1(x))
        )

        x = self.n2(self.c2(x))

        return F.relu(x + identity)


class MultiScaleBottleneck(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        b = channels // 4

        def branch(k, d):
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    b,
                    k,
                    padding=d if k == 3 else 0,
                    dilation=d,
                    bias=False,
                ),
                nn.GroupNorm(4, b),
                nn.ReLU(),
            )

        self.b1 = branch(1, 1)
        self.b2 = branch(3, 1)
        self.b3 = branch(3, 2)
        self.b4 = branch(3, 4)

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
        )

    def forward(self, x):
        y = torch.cat(
            [
                self.b1(x),
                self.b2(x),
                self.b3(x),
                self.b4(x),
            ],
            dim=1,
        )

        return self.fuse(y) + x


class DefectVisionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ResidualBlock(3, 16)
        self.d1 = nn.Conv2d(16, 16, 3, 2, 1)

        self.e2 = ResidualBlock(16, 32)
        self.d2 = nn.Conv2d(32, 32, 3, 2, 1)

        self.e3 = ResidualBlock(32, 48)
        self.d3 = nn.Conv2d(48, 48, 3, 2, 1)

        self.e4 = ResidualBlock(48, 64)
        self.d4 = nn.Conv2d(64, 64, 3, 2, 1)

        self.b = MultiScaleBottleneck(64)

        self.x4 = ResidualBlock(128, 48)
        self.x3 = ResidualBlock(96, 32)
        self.x2 = ResidualBlock(64, 16)
        self.x1 = ResidualBlock(32, 16)

        self.ref = ResidualBlock(16, 8)
        self.out = nn.Conv2d(8, 1, 1)

    def up(self, x, target):
        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        e4 = self.e4(self.d3(e3))

        x = self.b(self.d4(e4))

        x = self.x4(
            torch.cat(
                [self.up(x, e4), e4],
                dim=1,
            )
        )

        x = self.x3(
            torch.cat(
                [self.up(x, e3), e3],
                dim=1,
            )
        )

        x = self.x2(
            torch.cat(
                [self.up(x, e2), e2],
                dim=1,
            )
        )

        x = self.x1(
            torch.cat(
                [self.up(x, e1), e1],
                dim=1,
            )
        )

        return self.out(self.ref(x))


@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Final model checkpoint was not found at:\n"
            f"{CHECKPOINT}"
        )

    model = DefectVisionModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# INFERENCE
# ============================================================

def preprocess(image):
    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    rgb = (
        np.asarray(
            resized,
            dtype=np.float32,
        )
        / 255.0
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    normalized = (rgb - mean) / std

    tensor = torch.from_numpy(
        normalized
    ).permute(
        2, 0, 1
    ).unsqueeze(0).float()

    return rgb, tensor


@torch.no_grad()
def predict(model, image, threshold):
    rgb, tensor = preprocess(image)

    logits = model(
        tensor.to(DEVICE)
    )

    probability = torch.sigmoid(
        logits
    )[0, 0].cpu().numpy()

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    defect_pixels = int(mask.sum())
    total_pixels = int(mask.size)

    defect_percentage = (
        defect_pixels
        / total_pixels
        * 100.0
    )

    if defect_pixels > 0:
        mean_probability = float(
            probability[mask == 1].mean()
        )
    else:
        mean_probability = 0.0

    return {
        "rgb": rgb,
        "probability": probability,
        "mask": mask,
        "defect_detected": defect_pixels > 0,
        "defect_pixels": defect_pixels,
        "defect_percentage": defect_percentage,
        "mean_probability": mean_probability,
        "max_probability": float(
            probability.max()
        ),
    }


def create_overlay(rgb, mask):
    overlay = rgb.copy()
    defect = mask.astype(bool)

    overlay[defect] = (
        0.55 * overlay[defect]
        + 0.45 * np.array(
            [1.0, 0.0, 0.0]
        )
    )

    return overlay


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 DefectVision</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Industrial Defect Segmentation'
    '</div>',
    unsafe_allow_html=True,
)

st.write(
    "Upload an inspection image to obtain a pixel-level "
    "defect segmentation, probability map, and estimated "
    "defect area."
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Configuration")

    st.write(
        "**Architecture:**  MS-Residual U-Net"
    )
    st.write(
        "**Training objective:**  Weighted BCE + Dice"
    )
    st.write(
        "**Dataset:**  MVTec AD-SEG"
    )

    threshold = st.slider(
        "Segmentation threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help=(
            "Higher thresholds produce more conservative "
            "defect masks."
        ),
    )

    st.divider()

    st.subheader("Held-Out Test Performance")

    c1, c2 = st.columns(2)
    c1.metric("Dice", "0.3753")
    c2.metric("IoU", "0.2758")

    c3, c4 = st.columns(2)
    c3.metric("Precision", "0.3484")
    c4.metric("Recall", "0.6988")

    st.divider()

    st.caption(
        "Inference device: "
        + str(DEVICE).upper()
    )

# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload inspection image",
    type=[
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
    ],
)

if uploaded is None:

    st.info(
        "Upload an RGB inspection image to begin analysis."
    )

    st.markdown(
        """
        ### How it works

        **Image → Preprocessing → MS-Residual U-Net → Defect probability → Mask → Overlay**

        The model performs pixel-level segmentation rather than
        simply classifying the entire image as defective/non-defective.
        """
    )

else:

    try:

        model, checkpoint = load_model()

        image = Image.open(
            uploaded
        ).convert("RGB")

        with st.spinner(
            "Running DefectVision inference..."
        ):
            result = predict(
                model,
                image,
                threshold,
            )

        overlay = create_overlay(
            result["rgb"],
            result["mask"],
        )

        st.success(
            "Inspection completed successfully."
        )

        # ====================================================
        # KPI CARDS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Inspection Summary'
            '</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Defect Status",
            (
                "DETECTED"
                if result["defect_detected"]
                else "NOT DETECTED"
            ),
        )

        m2.metric(
            "Defect Area",
            f"{result['defect_percentage']:.2f}%",
        )

        m3.metric(
            "Mean Confidence",
            f"{result['mean_probability']:.1%}",
        )

        m4.metric(
            "Max Probability",
            f"{result['max_probability']:.1%}",
        )

        st.divider()

        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Segmentation Results'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Input image",
                use_container_width=True,
            )

        with col2:
            st.image(
                overlay,
                caption="Predicted defect overlay",
                use_container_width=True,
            )

        col3, col4 = st.columns(2)

        with col3:
            st.image(
                result["mask"] * 255,
                caption="Predicted defect mask",
                use_container_width=True,
            )

        with col4:
            st.image(
                result["probability"],
                caption="Defect probability map",
                use_container_width=True,
                clamp=True,
            )

        st.divider()

        # ====================================================
        # DETAILS + DOWNLOAD
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Prediction Details'
            '</div>',
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Defect Pixels",
            f"{result['defect_pixels']:,}",
        )

        d2.metric(
            "Model Input",
            f"{IMAGE_SIZE} × {IMAGE_SIZE}",
        )

        d3.metric(
            "Threshold",
            f"{threshold:.2f}",
        )

        mask_image = Image.fromarray(
            (
                result["mask"] * 255
            ).astype(np.uint8)
        )

        buffer = BytesIO()

        mask_image.save(
            buffer,
            format="PNG",
        )

        st.download_button(
            label="⬇ Download predicted mask",
            data=buffer.getvalue(),
            file_name="defectvision_predicted_mask.png",
            mime="image/png",
        )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div class="disclaimer">
            <strong>Research Prototype:</strong>
            DefectVision was trained and evaluated on MVTec AD-SEG.
            Results on images outside the training distribution may differ.
            This application is intended for research and portfolio
            demonstration, not safety-critical industrial deployment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:

        st.error(
            "Inference could not be completed."
        )

        st.exception(exc)

st.divider()

st.caption(
    "DefectVision • Deep Learning Industrial Inspection Prototype"
)
