"""
NeuroGraphia
────────────
Handwritten manuscript OCR tool.
Pipeline: deskew (OpenCV) -> binarization (Kraken) -> segmentation (Kraken) -> OCR (TrOCR)

Installation:
  pip install opencv-python-headless numpy Pillow "kraken>=4.3" transformers torch torchvision

Usage:
  python neurographia.py image1.jpg
  python neurographia.py --folder ./manuscripts --model large --device gpu --batch 8 --quantize
  python neurographia.py --model-info
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.quantization
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# kraken binarization replaced by OpenCV Otsu — no kraken imports needed here


TOOL_NAME    = "NeuroGraphia"
TOOL_VERSION = "1.0.0"

# ── Available models ──────────────────────────────────────────────────

MODELS = {
    "small": "microsoft/trocr-small-handwritten",
    "base" : "microsoft/trocr-base-handwritten",
    "large": "microsoft/trocr-large-handwritten",
}

MODEL_INFO = """
Available models (--model):

  small   microsoft/trocr-small-handwritten
          ~300 MB  |  fastest  |  lower accuracy
          Best for: quick tests, slow machines, legible handwriting

  base    microsoft/trocr-base-handwritten
          ~600 MB  |  balanced speed and quality
          Best for: general use, regular handwriting

  large   microsoft/trocr-large-handwritten   <- default
          ~1.3 GB  |  slowest  |  highest accuracy
          Best for: difficult manuscripts, irregular handwriting

Tip: start with --model small to test the pipeline quickly.
"""


# ── Deskew ────────────────────────────────────────────────────────────

def deskew(img_cv):
    """Corrects image orientation in two steps: cardinal rotation + fine adjustment."""

    def count_horizontal_lines(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Scale threshold with image width so it works on small images too
        thresh = max(50, img.shape[1] // 6)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
        count = 0
        if lines is not None:
            for line in lines:
                angle = np.degrees(line[0][1]) - 90
                if abs(angle) <= 15:
                    count += 1
        return count

    # Step A: pick the best cardinal rotation (0 / 90 / 180 / 270 deg)
    candidates = [
        img_cv,
        cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_cv, cv2.ROTATE_180),
        cv2.rotate(img_cv, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    img = max(candidates, key=count_horizontal_lines)

    # Step B: fine adjustment via Hough
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    thresh = max(50, img.shape[1] // 6)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=thresh)
    angles = []
    if lines is not None:
        for line in lines:
            a = np.degrees(line[0][1]) - 90
            if abs(a) <= 15:
                angles.append(a)
    fine_angle = float(np.median(angles)) if angles else 0.0

    if abs(fine_angle) > 0.3:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), fine_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return img


# ── Load model ────────────────────────────────────────────────────────

def load_model(model_name, device, quantize):
    repo = MODELS[model_name]
    print(f"[{TOOL_NAME}] Loading model: {repo}")
    processor = TrOCRProcessor.from_pretrained(repo)
    model     = VisionEncoderDecoderModel.from_pretrained(repo)

    if quantize:
        if device == "cuda":
            print(f"[{TOOL_NAME}] [warning] INT8 quantization is not supported on GPU -- skipped.")
        else:
            print(f"[{TOOL_NAME}] Applying INT8 quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

    model.to(device).eval()
    print(f"[{TOOL_NAME}] Model ready  |  device: {device}\n")
    return processor, model


# ── Line recognition ──────────────────────────────────────────────────

def recognize_lines(crops, processor, model, device, batch_size):
    """Runs TrOCR on a list of PIL crops in batches of `batch_size`."""
    results = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        inputs = processor(
            batch, return_tensors="pt", padding=True
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(inputs)
        results += processor.batch_decode(ids, skip_special_tokens=True)
    return results


# ── Process a single image ────────────────────────────────────────────

def process_image(path, processor, model, device, batch_size):
    print(f"[{TOOL_NAME}] -> {Path(path).name}")

    img_cv = cv2.imread(str(path))
    if img_cv is None:
        print(f"[{TOOL_NAME}] [ERROR] Could not open image.")
        return ""

    img_cv  = deskew(img_cv)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # ── Binarization (adaptive threshold) ────────────────────────────
    # Adaptive threshold handles uneven lighting better than global Otsu
    # (important for photos with shadows, curved pages, patterned backgrounds).
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bin_cv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )

    # ── Mask out borders (patterned/dark backgrounds) ─────────────────
    # Trim 5% from each edge to avoid background noise contaminating the
    # projection profile (e.g. tablecloth, spiral binding).
    h_img, w_img = bin_cv.shape
    mx, my = int(w_img * 0.05), int(h_img * 0.03)
    mask = np.zeros_like(bin_cv)
    mask[my:h_img - my, mx:w_img - mx] = 255
    bin_cv = cv2.bitwise_and(bin_cv, mask)

    # ── Line segmentation via horizontal projection profile ───────────
    # Wide kernel (40px) closes gaps between characters within a line;
    # height=2 avoids merging adjacent lines together.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    closed = cv2.morphologyEx(bin_cv, cv2.MORPH_CLOSE, kernel)

    # Sum pixel values per row (each white pixel = 255)
    proj = closed.sum(axis=1).astype(np.int64)
    h, w = closed.shape
    # A row counts as "text" if at least 0.5% of its width has ink
    min_row_fill = w * 255 * 0.005

    in_line, y_start = False, 0
    line_bboxes = []
    for y, val in enumerate(proj):
        if not in_line and val >= min_row_fill:
            in_line, y_start = True, y
        elif in_line and val < min_row_fill:
            in_line = False
            line_bboxes.append((y_start, y))
    if in_line:
        line_bboxes.append((y_start, h))

    # Merge segments whose gap is < 8px (broken ascenders/descenders)
    merged = []
    for (y0, y1) in line_bboxes:
        if merged and (y0 - merged[-1][1]) < 8:
            merged[-1] = (merged[-1][0], y1)
        else:
            merged.append((y0, y1))
    # Drop very thin segments (< 10px) — they are noise, not text lines
    line_bboxes = [(y0, y1) for y0, y1 in merged if (y1 - y0) >= 10]

    print(f"[{TOOL_NAME}]    {len(line_bboxes)} lines detected")

    # ── Debug: save annotated binarization image if --debug passed ────
    if getattr(process_image, "_debug", False):
        dbg = cv2.cvtColor(bin_cv, cv2.COLOR_GRAY2BGR)
        for (dy0, dy1) in line_bboxes:
            cv2.rectangle(dbg, (0, dy0), (w - 1, dy1), (0, 255, 0), 2)
        dbg_path = Path(path).stem + "_debug.png"
        cv2.imwrite(str(dbg_path), dbg)
        print(f"[{TOOL_NAME}]    debug image: {dbg_path}")

    # ── Build crops from original colour image ────────────────────────
    PAD = 8
    crops = []
    for (y0, y1) in line_bboxes:
        crop = img_pil.crop((
            0,
            max(0, y0 - PAD),
            img_pil.width,
            min(img_pil.height, y1 + PAD),
        )).convert("RGB")
        if crop.width >= 100 and crop.height >= 12:
            crops.append(crop)

    if not crops:
        return ""

    texts = recognize_lines(crops, processor, model, device, batch_size)
    return "\n".join(t.strip() for t in texts if t.strip())


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=f"{TOOL_NAME} {TOOL_VERSION} -- Handwritten manuscript OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("images", nargs="*", help="Image files to process")
    parser.add_argument("--folder",  "-p", help="Folder containing images")
    parser.add_argument("--output",  "-o", default="ocr_result.txt",
                        help="Output TXT file (default: ocr_result.txt)")
    parser.add_argument(
        "--model", "-m",
        choices=["small", "base", "large"],
        default="large",
        help="TrOCR model: small | base | large  (default: large)",
    )
    parser.add_argument(
        "--device", "-d",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Processing device: cpu | gpu  (default: cpu)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int, default=1,
        metavar="N",
        help="Lines processed per batch (default: 1). "
             "Increase to 4-8 if you have enough RAM.",
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Apply INT8 quantization on CPU (~30%% faster, CPU only)",
    )
    parser.add_argument(
        "--model-info",
        action="store_true",
        help="Show details about available models and exit",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{TOOL_NAME} {TOOL_VERSION}",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save <name>_debug.png with binarization + detected line boxes",
    )

    args = parser.parse_args()

    if args.model_info:
        print(MODEL_INFO)
        sys.exit(0)

    files = list(args.images)
    if args.folder:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif", "*.bmp"):
            files.extend(sorted(Path(args.folder).glob(ext)))

    if not files:
        parser.print_help()
        sys.exit(1)

    # Resolve device
    if args.device == "gpu":
        if not torch.cuda.is_available():
            print(f"[{TOOL_NAME}] [warning] GPU not found -- falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cpu"

    processor, model = load_model(args.model, device, args.quantize)
    process_image._debug = args.debug

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for file in files:
            text = process_image(file, processor, model, device, args.batch)
            if len(files) > 1:
                f.write(f"=== {Path(file).name} ===\n")
            f.write(text)
            f.write("\n\n")

    print(f"\n[{TOOL_NAME}] Saved to: {output}")


if __name__ == "__main__":
    main()