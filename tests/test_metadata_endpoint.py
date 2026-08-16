import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from server_app import (
    app,
    MetadataRequest,
    WindowInput,
    PageInput,
    BlockInput,
    build_metadata_extraction_prompt,
    format_kalanjiyam_metadata_response,
    verify_evidence_spans,
    compute_window_derived_metrics,
    aggregate_document_metrics,
    normalize_whitespace,
    resolve_engine,
    ENGINE_CONFIGS,
)


@pytest.fixture
def sample_metadata_request_dict():
    return {
        "contract_version": "1.0",
        "unit_id": "kalanjiyam:project/kalat-1932-17",
        "window": {
            "index": 3,
            "total": 24,
            "page_slugs": ["61", "62", "63", "64", "65"]
        },
        "taxonomy_version": "client-2026-08",
        "tags": ["TITLE", "DATE", "CREATOR", "SCOPE CONTENT", "PERSON NAME", "PLACE"],
        "language_hint": ["fa", "ur", "en"],
        "pages": [
            {
                "page_slug": "61",
                "ocr_confidence": 0.94,
                "blocks": [
                    {"id": "b1", "type": "heading", "reading_order": 1, "text": "Grant of an honorary commission to Lt. Shahzada Ahmad Yar Khan"},
                    {"id": "b2", "type": "paragraph", "reading_order": 2, "text": "First line of body text concerning the proposal to confer an honorary commission."},
                    {"id": "b3", "type": "paragraph", "reading_order": 3, "text": "Dated 11th March 1932 at Kalat."}
                ]
            },
            {
                "page_slug": "62",
                "ocr_confidence": 0.91,
                "blocks": [
                    {"id": "b7", "type": "paragraph", "reading_order": 1, "text": "Lt. Shahzada Ahmad Yar Khan of Kalat was born in 1904 and died in 1979."}
                ]
            }
        ]
    }


def test_metadata_request_pydantic_validation(sample_metadata_request_dict):
    """Verify MetadataRequest schema validation with Pydantic."""
    req = MetadataRequest(**sample_metadata_request_dict)
    assert req.contract_version == "1.0"
    assert req.unit_id == "kalanjiyam:project/kalat-1932-17"
    assert req.window.index == 3
    assert req.window.total == 24
    assert len(req.pages) == 2
    assert req.pages[0].page_slug == "61"
    assert req.pages[0].ocr_confidence == 0.94
    assert len(req.pages[0].blocks) == 3


def test_build_metadata_extraction_prompt(sample_metadata_request_dict):
    """Test prompt construction and character count."""
    req = MetadataRequest(**sample_metadata_request_dict)
    system_prompt, user_prompt, chars_in = build_metadata_extraction_prompt(req)

    # Check system prompt requirements
    assert "Kalanjiyam Metadata Extraction Specification (v1.0)" in system_prompt
    assert "TITLE" in system_prompt
    assert "PERSON NAME" in system_prompt

    # Check user prompt formatting
    assert "Unit ID: kalanjiyam:project/kalat-1932-17" in user_prompt
    assert "=== Page Slug: 61 (OCR confidence: 0.94) ===" in user_prompt
    assert "[Block id=b1 type=heading] Grant of an honorary commission" in user_prompt
    assert "[Block id=b7 type=paragraph]" in user_prompt

    # Check character counting
    expected_chars = sum(len(b["text"]) for p in sample_metadata_request_dict["pages"] for b in p["blocks"])
    assert chars_in == expected_chars
    assert chars_in > 0


def test_format_kalanjiyam_metadata_response_spec_compliance(sample_metadata_request_dict):
    """Verify response complies strictly with Specification (v1.0) Section 2B and Section 6."""
    req = MetadataRequest(**sample_metadata_request_dict)

    raw_model_json = {
        "fields": {
            "TITLE": {
                "value": "Grant of an honorary commission to Lt. Shahzada Ahmad Yar Khan",
                "confidence": 0.91,
                "source": "record",
                "evidence": [
                    {"page_slug": "61", "block_id": "b1", "quote": "Grant of an honorary commission"}
                ]
            },
            "DATE": {
                "value": "1932-03-11",
                "confidence": 0.88,
                "source": "record",
                "evidence": [
                    {"page_slug": "61", "block_id": "b3", "quote": "11th March 1932"}
                ]
            },
            "PERSON NAME": {
                "confidence": 0.77,
                "value": [
                    {
                        "label": "Ahmad Yar Khan, Shahzada",
                        "variants": ["Lt. Shahzada Ahmad Yar Khan"],
                        "dates": "1904-1979",
                        "source": "record",
                        "evidence": [
                            {"page_slug": "62", "block_id": "b7", "quote": "Lt. Shahzada Ahmad Yar Khan"}
                        ]
                    }
                ]
            },
            "SCOPE CONTENT": {
                "value": "Correspondence concerning the proposal to confer an honorary commission...",
                "confidence": 0.80,
                "source": "derived",
                "evidence": [
                    {"page_slug": "61"}, {"page_slug": "62"}
                ]
            }
        }
    }

    resp = format_kalanjiyam_metadata_response(
        raw_output=json.dumps(raw_model_json),
        request=req,
        chars_in=250,
        engine_latency_ms=3120.4,
        prompt_tokens=14320,
        completion_tokens=2870,
        engine="gemma-4",
        model_name="gemma-4-26b-a4b-it",
        model_version="1.0.0",
    )

    # 1. Top-Level Required Fields (Section 6)
    assert resp["contract_version"] == "1.0"
    assert resp["status"] == "success"
    assert resp["engine"] == "gemma-4"
    assert resp["model"] == {"name": "gemma-4-26b-a4b-it", "version": "1.0.0"}
    assert resp["taxonomy_version"] == "client-2026-08"
    assert resp["unit_id"] == "kalanjiyam:project/kalat-1932-17"
    assert resp["window_index"] == 3
    assert resp["chars_in"] == 250
    assert resp["engine_latency_ms"] == 3120.4
    assert resp["usage"] == {
        "prompt_tokens": 14320,
        "completion_tokens": 2870,
        "total_tokens": 17190,
    }

    # 2. Field Count Accounting (Section 3)
    assert resp["fields_attempted"] == 6  # 6 tags in request
    assert resp["fields_returned"] == 4   # 4 filled
    assert resp["fields_declined"] == 2   # 2 declined (CREATOR, PLACE)

    # 3. Field details
    assert "TITLE" in resp["fields"]
    assert resp["fields"]["TITLE"]["value"] == "Grant of an honorary commission to Lt. Shahzada Ahmad Yar Khan"
    assert resp["fields"]["TITLE"]["confidence"] == 0.91
    assert resp["fields"]["TITLE"]["source"] == "record"
    assert len(resp["fields"]["TITLE"]["evidence"]) == 1

    assert "PERSON NAME" in resp["fields"]
    assert resp["fields"]["PERSON NAME"]["confidence"] == 0.77
    assert isinstance(resp["fields"]["PERSON NAME"]["value"], list)
    entity = resp["fields"]["PERSON NAME"]["value"][0]
    assert entity["label"] == "Ahmad Yar Khan, Shahzada"
    assert entity["dates"] == "1904-1979"
    assert entity["source"] == "record"


def test_withholding_unrequested_tags(sample_metadata_request_dict):
    """Section 7: Honour tags. Tags absent from request.tags must NEVER appear in response."""
    # Request only asks for TITLE and DATE
    sample_metadata_request_dict["tags"] = ["TITLE", "DATE"]
    req = MetadataRequest(**sample_metadata_request_dict)

    # Model incorrectly generated unrequested UNAPPROVED_TAG and PERSON NAME
    raw_model_json = {
        "fields": {
            "TITLE": {"value": "Test Title", "confidence": 0.95, "source": "record"},
            "DATE": {"value": "1932", "confidence": 0.90, "source": "record"},
            "PERSON NAME": {"value": [{"label": "Unrequested Person", "source": "record"}], "confidence": 0.8},
            "UNAPPROVED_TAG": {"value": "Secret metadata", "confidence": 0.99, "source": "derived"}
        }
    }

    resp = format_kalanjiyam_metadata_response(
        raw_output=raw_model_json,
        request=req,
        chars_in=100,
        engine_latency_ms=500.0,
        prompt_tokens=100,
        completion_tokens=50,
        engine="gemma-4",
        model_name="gemma-4-26b-a4b-it",
        model_version="1.0.0",
    )

    assert "TITLE" in resp["fields"]
    assert "DATE" in resp["fields"]
    assert "PERSON NAME" not in resp["fields"]
    assert "UNAPPROVED_TAG" not in resp["fields"]
    assert resp["fields_attempted"] == 2
    assert resp["fields_returned"] == 2
    assert resp["fields_declined"] == 0


def test_evidence_quote_verification_verbatim_matching(sample_metadata_request_dict):
    """Section 4.3: Check that quote appears verbatim (whitespace-normalised) in source block."""
    req = MetadataRequest(**sample_metadata_request_dict)

    fields = {
        "TITLE": {
            "value": "Grant of an honorary commission",
            "source": "record",
            "evidence": [
                {"page_slug": "61", "block_id": "b1", "quote": "Grant of an honorary commission"}
            ]
        },
        "DATE": {
            "value": "1932-03-11",
            "source": "record",
            "evidence": [
                # Mismatched quote that does not exist in block b3
                {"page_slug": "61", "block_id": "b3", "quote": "Invented fake quote not in text"}
            ]
        },
        "SCOPE CONTENT": {
            "value": "Synthesized summary",
            "source": "derived",
            "evidence": [{"page_slug": "61"}]
        }
    }

    verification = verify_evidence_spans(fields=fields, pages=sample_metadata_request_dict["pages"])

    assert verification["total_record_values_count"] == 2  # TITLE and DATE are record sources
    assert verification["verified_spans_count"] == 1       # TITLE is verified, DATE is not
    assert verification["evidence_verified_rate"] == 0.5   # 1 / 2 = 50%

    details = verification["span_details"]
    assert details[0]["tag"] == "TITLE"
    assert details[0]["verified"] is True
    assert details[1]["tag"] == "DATE"
    assert details[1]["verified"] is False


def test_compute_window_derived_metrics(sample_metadata_request_dict):
    """Section 3: Test per-window derived metrics computation."""
    response_payload = {
        "unit_id": "kalanjiyam:project/kalat-1932-17",
        "window_index": 3,
        "engine": "kalanjiyam-archival",
        "model": {"name": "gemma-3-27b-it", "version": "1.0"},
        "taxonomy_version": "client-2026-08",
        "chars_in": 350,
        "engine_latency_ms": 1200.0,
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
        "fields_attempted": 6,
        "fields_returned": 3,
        "fields_declined": 3,
        "fields": {
            "TITLE": {
                "value": "Grant of an honorary commission",
                "confidence": 0.92,
                "source": "record",
                "evidence": [{"page_slug": "61", "block_id": "b1", "quote": "Grant of an honorary commission"}]
            },
            "DATE": {
                "value": "1932-03-11",
                "confidence": 0.65,  # < 0.7 low confidence field
                "source": "record",
                "evidence": [{"page_slug": "61", "block_id": "b3", "quote": "11th March 1932"}]
            },
            "SCOPE CONTENT": {
                "value": "Derived summary",
                "confidence": 0.80,
                "source": "derived",
                "evidence": [{"page_slug": "61"}]
            }
        }
    }

    derived = compute_window_derived_metrics(
        response_payload=response_payload,
        request_payload=sample_metadata_request_dict,
        extraction_latency_ms=1350.0
    )

    assert derived["mean_field_confidence"] == round((0.92 + 0.65 + 0.80) / 3, 3)
    assert derived["min_field_confidence"] == 0.65
    assert derived["low_confidence_fields_count"] == 1  # DATE at 0.65
    assert derived["evidence_spans_count"] == 3
    assert derived["evidence_verified_rate"] == 1.0     # Both TITLE and DATE quotes exist
    assert derived["source_ocr_confidence"] == round((0.94 + 0.91) / 2, 3)
    assert derived["engine_latency_ms"] == 1200.0
    assert derived["extraction_latency_ms"] == 1350.0


def test_nullable_ocr_confidence_stays_null(sample_metadata_request_dict):
    """Section 4.5: Nulls must stay null when all pages have null OCR confidence."""
    for p in sample_metadata_request_dict["pages"]:
        p["ocr_confidence"] = None

    response_payload = {
        "unit_id": "kalanjiyam:project/kalat-1932-17",
        "window_index": 3,
        "engine": "kalanjiyam-archival",
        "model": {"name": "gemma-3-27b-it", "version": "1.0"},
        "taxonomy_version": "client-2026-08",
        "chars_in": 350,
        "engine_latency_ms": 1000.0,
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        "fields_attempted": 2,
        "fields_returned": 1,
        "fields_declined": 1,
        "fields": {"TITLE": {"value": "Title", "confidence": 0.9, "source": "derived"}}
    }

    derived = compute_window_derived_metrics(response_payload, sample_metadata_request_dict)
    assert derived["source_ocr_confidence"] is None


def test_aggregate_document_metrics():
    """Section 5: Test Full Document Aggregated Metrics rollup."""
    win1 = {
        "unit_id": "doc-123",
        "window": {"index": 1, "total": 2},
        "engine_latency_ms": 2000.0,
        "usage": {"prompt_tokens": 5000, "completion_tokens": 1000},
        "fields": {
            "TITLE": {"value": "Document Title", "confidence": 0.95},
            "DATE": {"value": "1932", "confidence": 0.85},
        }
    }
    win2 = {
        "unit_id": "doc-123",
        "window": {"index": 2, "total": 2},
        "engine_latency_ms": 3000.0,
        "usage": {"prompt_tokens": 6000, "completion_tokens": 1500},
        "fields": {
            "PERSON NAME": {"value": [{"label": "Khan"}], "confidence": 0.65},
            "SCOPE CONTENT": {"value": "Summary content", "confidence": 0.80},
        }
    }

    schema_tags = ["TITLE", "DATE", "CREATOR", "PERSON NAME", "SCOPE CONTENT"]
    rollup = aggregate_document_metrics(
        window_responses=[win1, win2],
        total_pages=10,
        taxonomy_tags=schema_tags,
        unit_id="doc-123"
    )

    assert rollup["unit_id"] == "doc-123"
    assert rollup["windows_completed"] == 2
    assert rollup["windows_total"] == 2
    assert rollup["pages_read"] == 10
    assert rollup["extraction_coverage"] == 1.0
    assert rollup["field_coverage"] == 4 / 5  # 4 fields filled out of 5 tags
    assert rollup["avg_confidence"] == round((0.95 + 0.85 + 0.65 + 0.80) / 4, 3)
    assert rollup["min_confidence"] == 0.65
    assert rollup["fields_below_0_7"] == 1
    assert rollup["total_prompt_tokens"] == 11000
    assert rollup["total_completion_tokens"] == 2500
    assert rollup["total_tokens"] == 13500
    assert rollup["avg_engine_latency_ms"] == 2500.0


def test_api_v1_metadata_endpoint_execution(sample_metadata_request_dict):
    """Test FastAPI /v1/metadata endpoint with mocked backend."""
    client = TestClient(app)

    mock_llm_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({
                        "fields": {
                            "TITLE": {
                                "value": "Grant of an honorary commission",
                                "confidence": 0.95,
                                "source": "record",
                                "evidence": [{"page_slug": "61", "block_id": "b1", "quote": "Grant of an honorary commission"}]
                            }
                        }
                    })
                }
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160
        }
    }

    with patch("server_app.gpu_manager.start_backend") as mock_start:
        mock_start.return_value = {"gpu_idx": 0, "port": 8000, "engine": "gemma-4"}
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_llm_response
            mock_post.return_value = mock_resp

            response = client.post("/v1/metadata", json=sample_metadata_request_dict)
            assert response.status_code == 200
            data = response.json()
            assert data["contract_version"] == "1.0"
            assert data["status"] == "success"
            assert data["engine"] == "gemma-4"
            assert data["model"]["name"] == "gemma-4-26b-a4b-it"
            assert data["unit_id"] == "kalanjiyam:project/kalat-1932-17"
            assert data["window_index"] == 3
            assert "TITLE" in data["fields"]
            assert data["fields"]["TITLE"]["value"] == "Grant of an honorary commission"


def test_api_v1_metadata_error_handling():
    """Section 8: Verify HTTP error returns {"status": "error", "detail": "..."}"""
    client = TestClient(app)

    # Empty pages list
    bad_request = {
        "contract_version": "1.0",
        "unit_id": "doc-1",
        "window": {"index": 1, "total": 1, "page_slugs": []},
        "taxonomy_version": "v1",
        "tags": ["TITLE"],
        "pages": []
    }
    response = client.post("/v1/metadata", json=bad_request)
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "at least one page" in data["detail"]


def test_engine_resolution_archival():
    """Verify archival / metadata engine aliases resolve to gemma-4."""
    assert resolve_engine("kalanjiyam-archival") in ("gemma-4", "kalanjiyam-archival")
    assert resolve_engine("kalanjiyam_archival") in ("gemma-4", "kalanjiyam-archival")
    assert resolve_engine("archival") == "gemma-4"
    assert resolve_engine("metadata") == "gemma-4"
    assert resolve_engine("gemma-4") == "gemma-4"
    assert resolve_engine("gemma-4-26b-a4b-it") == "gemma-4"
    assert resolve_engine("gemma-4-26b-a4b-it") == "gemma-4"
