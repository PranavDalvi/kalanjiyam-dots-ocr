"""
Diagnostic script for inspecting and validating bounding-box transformations
across the Dots.OCR pipeline for known test images.
"""
import io
import json
from PIL import Image
from server_app import format_kalanjiyam_v2_response, scale_bbox_to_pixels


def run_diagnostic_case(name: str, width: int, height: int, raw_layout: list):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC TEST CASE: {name}")
    print(f"{'='*70}")
    
    # Create test image
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    # Step A: Input image dimensions
    print(f"A. Input Image Dimensions: {width} x {height} (Aspect ratio: {round(width/height, 3)})")

    # Step B: Raw Dots.OCR model response
    print(f"B. Raw Dots.OCR Response: {json.dumps(raw_layout)}")

    # Step C: Raw bbox coordinates
    raw_bboxes = [item.get("bbox") for item in raw_layout]
    print(f"C. Raw Bbox Coordinates ([0, 1000] scale): {raw_bboxes}")

    # Step D: Parsed / scaled bbox coordinates
    scaled_bboxes = [scale_bbox_to_pixels(b, width, height) for b in raw_bboxes]
    print(f"D. Parsed / Scaled Pixel Bboxes ([0..W, 0..H]): {scaled_bboxes}")

    # Step E: Final Kalanjiyam response
    response = format_kalanjiyam_v2_response(
        parsed_layout=raw_layout,
        image_bytes=image_bytes,
        filename=f"{name}.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=20,
        throughput=100.0,
    )
    final_bboxes = [b["bbox"] for b in response["blocks"]]
    print(f"E. Final Kalanjiyam Response Bboxes: {final_bboxes}")

    # Sanity checks
    for idx, (raw, final) in enumerate(zip(raw_bboxes, final_bboxes)):
        expected_x1 = round((raw[0] / 1000.0) * width, 2)
        expected_y1 = round((raw[1] / 1000.0) * height, 2)
        expected_x2 = round((raw[2] / 1000.0) * width, 2)
        expected_y2 = round((raw[3] / 1000.0) * height, 2)
        assert final == [expected_x1, expected_y1, expected_x2, expected_y2], (
            f"Mismatch at index {idx}: expected {[expected_x1, expected_y1, expected_x2, expected_y2]}, got {final}"
        )
    print("✓ All bounding boxes match exact normalized pixel projections.")


def test_diagnostic_suite():
    # Case 1: Standard A4 scanned page at 200 DPI (approx 1654 x 2338)
    run_diagnostic_case(
        name="A4_200DPI",
        width=1654,
        height=2338,
        raw_layout=[
            {"category": "title", "bbox": [100, 50, 900, 100], "text": "Main Document Heading"},
            {"category": "text", "bbox": [100, 120, 900, 600], "text": "Paragraph body text."}
        ]
    )

    # Case 2: Small Image (800 x 600)
    run_diagnostic_case(
        name="Small_Image_800x600",
        width=800,
        height=600,
        raw_layout=[
            {"category": "header", "bbox": [50, 50, 950, 150], "text": "Small Page Header"},
            {"category": "table", "bbox": [50, 200, 950, 800], "text": "<table>...</table>"}
        ]
    )

    # Case 3: Square Image (1000 x 1000)
    run_diagnostic_case(
        name="Square_1000x1000",
        width=1000,
        height=1000,
        raw_layout=[
            {"category": "text", "bbox": [200, 300, 800, 700], "text": "Centered Content"}
        ]
    )


if __name__ == "__main__":
    test_diagnostic_suite()
