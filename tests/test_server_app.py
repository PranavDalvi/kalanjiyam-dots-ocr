import pytest
import io
import math
from PIL import Image
from fastapi.testclient import TestClient
from server_app import (
    app,
    format_kalanjiyam_v2_response,
    _generate_word_spans,
    scale_bbox_to_pixels,
    resolve_engine,
    extract_json_from_model_output,
    ENGINE_CONFIGS,
)


def create_dummy_image(width=1240, height=1754) -> bytes:
    """Helper to generate dummy RGB JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_format_kalanjiyam_v2_1_response_spec():
    """Verify response complies with OCR API Response & Metrics Specification (v2.1)."""
    image_bytes = create_dummy_image(1240, 1754)

    parsed_layout = [
        {
            "category": "title",
            "bbox": [100, 40, 900, 88],  # normalized 0-1000 scale
            "text": "Chapter Title Text",
            "confidence": 0.985,
        },
        {
            "category": "text",
            "bbox": [100, 100, 900, 280],
            "text": "First line of body text.\nSecond line of body text.",
            "confidence": 0.912,
        },
    ]

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout,
        image_bytes=image_bytes,
        filename="1.jpg",
        active_gpu=0,
        language="sa",
        duration_seconds=0.3425,
        prompt_tokens=100,
        completion_tokens=50,
        throughput=146.0,
        engine="surya",
    )

    # 1. Top-level required fields
    assert resp["contract_version"] == "2.1"
    assert resp["engine"] == "surya"
    assert resp["model"] == {"name": "dots-ocr", "version": "4.0.0"}
    assert isinstance(resp["page_confidence"], float)
    assert 0.0 <= resp["page_confidence"] <= 1.0
    assert resp["engine_latency_ms"] == 342.5
    assert resp["page_width"] == 1240
    assert resp["page_height"] == 1754
    assert isinstance(resp["blocks"], list)
    assert len(resp["blocks"]) == 2

    # 2. Block 1 verification
    b1 = resp["blocks"][0]
    assert b1["id"] == "b1"
    assert b1["type"] == "heading"
    assert len(b1["bbox"]) == 4
    assert b1["reading_order"] == 1
    assert b1["content"] == "Chapter Title Text"
    assert b1["confidence"] == 0.985
    assert isinstance(b1["words"], list)
    assert len(b1["words"]) > 0

    # 3. Block 2 verification
    b2 = resp["blocks"][1]
    assert b2["id"] == "b2"
    assert b2["type"] == "paragraph"
    assert b2["reading_order"] == 2
    assert "First line" in b2["content"]
    assert b2["confidence"] == 0.912
    assert isinstance(b2["words"], list)
    assert len(b2["words"]) > 0

    # 4. Word fields verification
    for word in b2["words"]:
        assert "text" in word
        assert "bbox" in word
        assert len(word["bbox"]) == 4
        assert "confidence" in word
        assert isinstance(word["confidence"], float)


def test_core_metrics_extraction():
    """Test that all 6 core metrics can be extracted directly from response payload."""
    image_bytes = create_dummy_image(1000, 1000)
    parsed_layout = [
        {
            "category": "section-header",
            "bbox": [100, 50, 900, 150],
            "text": "Header Section",
            "confidence": 0.96,
        },
        {
            "category": "text",
            "bbox": [100, 200, 900, 600],
            "text": "Body paragraph content goes here.",
            "confidence": 0.88,
        },
    ]

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout,
        image_bytes=image_bytes,
        filename="test.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.5,
        prompt_tokens=10,
        completion_tokens=20,
        throughput=40.0,
        engine="tesseract",
    )

    # Core Metric 1: Engine
    engine_metric = resp["engine"]
    assert engine_metric == "tesseract"

    # Core Metric 2: Confidence
    page_confidence_metric = resp["page_confidence"]
    assert isinstance(page_confidence_metric, float)

    # Core Metric 3: p05 (5th percentile confidence)
    word_confidences = [
        w["confidence"] for b in resp["blocks"] for w in b.get("words", [])
    ]
    assert len(word_confidences) > 0
    # Calculate 5th percentile cutoff
    sorted_confs = sorted(word_confidences)
    p05_index = math.floor(0.05 * len(sorted_confs))
    p05_metric = sorted_confs[p05_index]
    assert 0.0 <= p05_metric <= 1.0

    # Core Metric 4: Blocks (Count of items in blocks array)
    blocks_metric = len(resp["blocks"])
    assert blocks_metric == 2

    # Core Metric 5: Chars (Character length sum of block contents)
    chars_metric = sum(len(b["content"]) for b in resp["blocks"])
    assert chars_metric == len("Header Section") + len("Body paragraph content goes here.")

    # Core Metric 6: Latency (Pure OCR model processing latency in ms)
    latency_metric = resp["engine_latency_ms"]
    assert latency_metric == 500.0


def test_dict_layout_wrapper():
    """Verify handling when parsed_layout is encapsulated in a dict."""
    image_bytes = create_dummy_image(800, 600)
    parsed_layout_dict = {
        "blocks": [
            {
                "category": "title",
                "bbox": [50, 50, 750, 100],
                "text": "Wrapped Layout Title",
                "confidence": 0.99,
            }
        ]
    }

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout_dict,
        image_bytes=image_bytes,
        filename="wrapped.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.1,
        prompt_tokens=5,
        completion_tokens=10,
        throughput=100.0,
    )

    assert resp["contract_version"] == "2.1"
    assert len(resp["blocks"]) == 1
    assert resp["blocks"][0]["content"] == "Wrapped Layout Title"
    assert resp["blocks"][0]["type"] == "heading"


def test_string_parsed_layout_fallback():
    """Verify fallback when parsed_layout is raw string plain text."""
    image_bytes = create_dummy_image(800, 600)
    raw_text = "Raw unparsed OCR output text string."

    resp = format_kalanjiyam_v2_response(
        parsed_layout=raw_text,
        image_bytes=image_bytes,
        filename="raw.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.2,
        prompt_tokens=5,
        completion_tokens=10,
        throughput=50.0,
    )

    assert resp["contract_version"] == "2.1"
    assert len(resp["blocks"]) == 1
    assert resp["blocks"][0]["type"] == "paragraph"
    assert resp["blocks"][0]["content"] == raw_text
    assert len(resp["blocks"][0]["words"]) > 0


def test_api_v1_ocr_endpoint_schema():
    """Test FastAPI endpoint discovery and schema."""
    client = TestClient(app)

    # GET /v1/engines
    resp = client.get("/v1/engines")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "dots-ocr" in data["engines"]
    assert "gemma-4" in data["engines"]
    assert "kalanjiyam-archival" in data["engines"]
    assert data["default_engine"] in data["engines"]

    # Also test /engines alias
    resp_alias = client.get("/engines")
    assert resp_alias.status_code == 200
    assert resp_alias.json().get("status") == "ok"
    assert "dots-ocr" in resp_alias.json()["engines"]
    assert "gemma-4" in resp_alias.json()["engines"]
    assert "kalanjiyam-archival" in resp_alias.json()["engines"]


def test_engine_resolution():
    """Verify that various aliases resolve to canonical engine IDs."""
    assert resolve_engine("dots-ocr") == "dots-ocr"
    assert resolve_engine("dots_ocr") == "dots-ocr"
    assert resolve_engine("dotsocr") == "dots-ocr"
    assert resolve_engine("dots") == "dots-ocr"
    assert resolve_engine("rednote-hilab/dots.ocr") == "dots-ocr"

    assert resolve_engine("gemma-4") == "gemma-4"
    assert resolve_engine("gemma-4-26b") == "gemma-4"
    assert resolve_engine("gemma-4-26b-a4b-it") == "gemma-4"
    assert resolve_engine("google/gemma-4-26B-A4B-it") == "gemma-4"
    assert resolve_engine("gemma4") == "gemma-4"
    assert resolve_engine("gemma") == "gemma-4"

    # Default fallback when None or empty
    assert resolve_engine(None) == "dots-ocr"
    assert resolve_engine("") == "dots-ocr"


def test_extract_json_from_model_output():
    """Verify robust extraction of JSON from raw model output strings."""
    # Direct JSON string
    raw_json = '{"blocks": [{"category": "text", "bbox": [0, 0, 100, 100], "text": "Hello"}]}'
    extracted = extract_json_from_model_output(raw_json)
    assert isinstance(extracted, dict)
    assert len(extracted["blocks"]) == 1

    # JSON inside markdown code fence
    markdown_fence = 'Here is the result:\n```json\n{"blocks": [{"category": "title", "bbox": [10, 10, 500, 50], "text": "Title"}]}\n```\nDone.'
    extracted_fence = extract_json_from_model_output(markdown_fence)
    assert isinstance(extracted_fence, dict)
    assert extracted_fence["blocks"][0]["category"] == "title"

    # JSON array inside text
    array_text = 'OCR output:\n[{"category": "formula", "bbox": [50, 50, 200, 100], "text": "E = mc^2"}]'
    extracted_arr = extract_json_from_model_output(array_text)
    assert isinstance(extracted_arr, list)
    assert extracted_arr[0]["category"] == "formula"

    # Plain text fallback
    plain = "This is just plain text with no JSON."
    assert extract_json_from_model_output(plain) == plain


def test_gemma4_response_format():
    """Verify format_kalanjiyam_v2_response formatting for Gemma 4 engine."""
    image_bytes = create_dummy_image(1200, 1600)
    parsed_layout = {
        "blocks": [
            {
                "category": "title",
                "bbox": [100, 50, 900, 120],
                "text": "Gemma 4 Document Title",
                "confidence": 0.99
            },
            {
                "category": "formula",
                "bbox": [100, 150, 600, 250],
                "text": "\\sum_{i=1}^n x_i",
                "confidence": 0.97
            }
        ]
    }

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout,
        image_bytes=image_bytes,
        filename="gemma_doc.jpg",
        active_gpu=1,
        language="en",
        duration_seconds=0.45,
        prompt_tokens=150,
        completion_tokens=60,
        throughput=133.3,
        engine="gemma-4",
    )

    assert resp["contract_version"] == "2.1"
    assert resp["engine"] == "gemma-4"
    assert resp["model"]["name"] == "gemma-4-26b-a4b-it"
    assert resp["page_width"] == 1200
    assert resp["page_height"] == 1600
    assert len(resp["blocks"]) == 2
    assert resp["blocks"][0]["type"] == "heading"
    assert resp["blocks"][1]["type"] == "equation"
    assert resp["blocks"][1]["content"] == "\\sum_{i=1}^n x_i"


def test_server_entrypoint_import():
    """Verify server.py imports and exposes application components."""
    import server
    assert hasattr(server, "app")
    assert hasattr(server, "gpu_manager")
    assert hasattr(server, "ENGINE_CONFIGS")
    assert "gemma-4" in server.ENGINE_CONFIGS



def test_document_and_project_level_aggregation_simulation():
    """Simulate Document & Project Level Aggregated Metrics (Section 4)."""
    image_bytes = create_dummy_image(1000, 1000)

    page1 = format_kalanjiyam_v2_response(
        parsed_layout=[{"category": "text", "bbox": [10, 10, 900, 500], "text": "Page one text content.", "confidence": 0.95}],
        image_bytes=image_bytes,
        filename="page1.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.35,
        prompt_tokens=10,
        completion_tokens=20,
        throughput=50.0,
        engine="surya",
    )

    page2 = format_kalanjiyam_v2_response(
        parsed_layout=[{"category": "text", "bbox": [10, 10, 900, 500], "text": "Low quality page content.", "confidence": 0.62}],
        image_bytes=image_bytes,
        filename="page2.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.45,
        prompt_tokens=10,
        completion_tokens=20,
        throughput=40.0,
        engine="surya",
    )

    pages = [page1, page2]

    # Section 4 Metric Rollups
    total_pages = len(pages)
    avg_conf = sum(p["page_confidence"] for p in pages) / total_pages
    min_conf = min(p["page_confidence"] for p in pages)
    low_conf_page_count = sum(1 for p in pages if p["page_confidence"] < 0.7)
    avg_engine_latency_sec = sum(p["engine_latency_ms"] / 1000.0 for p in pages) / total_pages
    total_chars = sum(sum(len(b["content"]) for b in p["blocks"]) for p in pages)

    assert total_pages == 2
    assert round(avg_conf, 3) == 0.785
    assert min_conf == 0.62
    assert low_conf_page_count == 1
    assert round(avg_engine_latency_sec, 2) == 0.40
    assert total_chars == len("Page one text content.") + len("Low quality page content.")


def test_scale_bbox_to_pixels_cases():
    """Verify scale_bbox_to_pixels across all required edge cases (Cases A-F)."""
    # Case A & B: Normalized [0, 1000] scale on standard page (1240 x 1754)
    res = scale_bbox_to_pixels([100, 40, 900, 88], 1240, 1754)
    assert res == [124.0, 70.16, 1116.0, 154.35]

    # Case C: Large image (2000 x 3000)
    res_large = scale_bbox_to_pixels([100, 200, 900, 800], 2000, 3000)
    assert res_large == [200.0, 600.0, 1800.0, 2400.0]

    # Case D: Small image (800 x 600) - specifically checks fix for width/height <= 1000
    res_small = scale_bbox_to_pixels([100, 200, 900, 800], 800, 600)
    assert res_small == [80.0, 120.0, 720.0, 480.0]

    # Case E: Bbox touching image boundaries [0, 0, 1000, 1000]
    res_bound = scale_bbox_to_pixels([0, 0, 1000, 1000], 1200, 1600)
    assert res_bound == [0.0, 0.0, 1200.0, 1600.0]

    # Case E2: Slight model float overshoot / negative values
    res_overshoot = scale_bbox_to_pixels([-5.0, -10.0, 1005.0, 1008.0], 1000, 1000)
    assert res_overshoot == [0.0, 0.0, 1000.0, 1000.0]

    # Case F: Malformed bboxes
    assert scale_bbox_to_pixels(None, 1000, 1000) == [0.0, 0.0, 0.0, 0.0]
    assert scale_bbox_to_pixels([100, 200], 1000, 1000) == [0.0, 0.0, 0.0, 0.0]
    assert scale_bbox_to_pixels(["invalid", "data", "bad", "type"], 1000, 1000) == [0.0, 0.0, 0.0, 0.0]
    assert scale_bbox_to_pixels("not a list", 1000, 1000) == [0.0, 0.0, 0.0, 0.0]

    # Inverted coordinates [x2, y2, x1, y1] should be sorted properly
    res_inverted = scale_bbox_to_pixels([900, 800, 100, 200], 1000, 1000)
    assert res_inverted == [100.0, 200.0, 900.0, 800.0]

    # Explicit pixel coordinates on large page (>1050 values within dimensions)
    res_already_px = scale_bbox_to_pixels([1200, 1500, 1800, 2500], 2000, 3000)
    assert res_already_px == [1200.0, 1500.0, 1800.0, 2500.0]


def test_small_image_end_to_end_scaling():
    """Verify end-to-end response formatting for small image (800 x 600)."""
    image_bytes = create_dummy_image(800, 600)
    parsed_layout = [
        {
            "category": "title",
            "bbox": [100, 100, 900, 200],  # 10%-90% width, 10%-20% height
            "text": "Small Image Title",
            "confidence": 0.98,
        }
    ]

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout,
        image_bytes=image_bytes,
        filename="small.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.15,
        prompt_tokens=20,
        completion_tokens=30,
        throughput=200.0,
    )

    block = resp["blocks"][0]
    # Expected: x1 = 80.0, y1 = 60.0, x2 = 720.0, y2 = 120.0
    assert block["bbox"] == [80.0, 60.0, 720.0, 120.0]
    assert block["words"][0]["bbox"][0] >= 80.0
    assert block["words"][-1]["bbox"][2] <= 720.0


def test_debug_bbox_flag(monkeypatch):
    """Verify DEBUG_BBOX exposes raw model response and bbox coordinates."""
    import server_app
    monkeypatch.setattr(server_app, "DEBUG_BBOX", True)

    image_bytes = create_dummy_image(1000, 1000)
    parsed_layout = [
        {"category": "text", "bbox": [50, 50, 950, 950], "text": "Debug text", "confidence": 0.95}
    ]

    resp = format_kalanjiyam_v2_response(
        parsed_layout=parsed_layout,
        image_bytes=image_bytes,
        filename="debug_test.jpg",
        active_gpu=0,
        language="en",
        duration_seconds=0.1,
        prompt_tokens=10,
        completion_tokens=20,
        throughput=200.0,
    )

    assert "debug_bbox" in resp
    assert resp["debug_bbox"]["image_width"] == 1000
    assert resp["debug_bbox"]["image_height"] == 1000
    assert resp["debug_bbox"]["raw_model_response"] == parsed_layout
    assert len(resp["debug_bbox"]["final_bboxes"]) == 1
    assert resp["debug_bbox"]["final_bboxes"][0] == [50.0, 50.0, 950.0, 950.0]


