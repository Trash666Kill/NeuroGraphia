"""
ocr_manuscript.py
─────────────────
Extracts text from handwritten manuscripts and saves to TXT.
Pipeline: deskew (OpenCV) -> binarization (Kraken) -> segmentation (Kraken) -> OCR (TrOCR)

Installation:
  pip install opencv-python-headless numpy Pillow kraken transformers torch torchvision

Usage:
  python ocr_manuscript.py image1.jpg
  python ocr_manuscript.py --folder ./manuscripts --model large --device gpu --batch 8 --quantize
  python ocr_manuscript.py --model-info
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
from kraken import binarization, pageseg


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
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
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
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
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
    print(f"Loading model: {repo}")
    processor = TrOCRProcessor.from_pretrained(repo)
    model     = VisionEncoderDecoderModel.from_pretrained(repo)

    if quantize:
        if device == "cuda":
            print("  [warning] INT8 quantization is not supported on GPU -- skipped.")
        else:
            print("  Applying INT8 quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

    model.to(device).eval()
    print(f"  Ready  |  device: {device}\n")
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
    print(f"-> {Path(path).name}")

    img_cv = cv2.imread(str(path))
    if img_cv is None:
        print("  [ERROR] Could not open image.")
        return ""

    img_cv  = deskew(img_cv)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    img_bin     = binarization.nlbin(img_pil)
    segmentation = pageseg.segment(img_bin)
    print(f"  {len(segmentation.lines)} lines detected")

    # Build list of valid line crops
    crops = []
    for line in segmentation.lines:
        x0, y0, x1, y1 = line.bbox
        crop = img_pil.crop((
            max(0, x0 - 6), max(0, y0 - 6),
            min(img_pil.width, x1 + 6), min(img_pil.height, y1 + 6)
        )).convert("RGB")
        if crop.width >= 50:
            crops.append(crop)

    if not crops:
        return ""

    texts = recognize_lines(crops, processor, model, device, batch_size)
    return "\n".join(t.strip() for t in texts if t.strip())


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Handwritten manuscript OCR -> TXT",
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
            print("[warning] GPU not found -- falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cpu"

    processor, model = load_model(args.model, device, args.quantize)

    output = Path(args.output)
    with output.open("w", encoding="utf-8") as f:
        for file in files:
            text = process_image(file, processor, model, device, args.batch)
            if len(files) > 1:
                f.write(f"=== {Path(file).name} ===\n")
            f.write(text)
            f.write("\n\n")

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
