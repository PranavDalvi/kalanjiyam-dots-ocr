#!/usr/bin/env bash

JOBS=(
  "/projects/data/downloads/nauman/archive_scraper/downloads_urdu|/fsxnew/shyam.pawar/OCR_stuff/02_OCRd/Archive_Multilingual/dedups/urdu.jsonl"
)

for job in "${JOBS[@]}"; do
    IFS="|" read -r INPUT_DIR OUTPUT_FILE <<< "$job"

    LABEL="$(basename "$OUTPUT_FILE" .jsonl)"

    echo "========================================"
    echo "Starting job: $LABEL"
    echo "Input : $INPUT_DIR"
    echo "Output: $OUTPUT_FILE"
    echo "========================================"

    echo "----- [$LABEL] Running remove_invalid_json (pre-run) -----"
    echo ""

    python3 /fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/utils/remove_invalid_json.py \
        -i "$OUTPUT_FILE" \
        || echo "[WARN] [$LABEL] remove_invalid_json failed on run $i"

    echo ""

    OUT_DIR="$(dirname "$OUTPUT_FILE")"
    OUT_BASE="$(basename "$OUTPUT_FILE" .jsonl)"
    LOG_FILE="$OUT_DIR/.${OUT_BASE}.line_counts_utc.log"

    for i in $(seq 1 10); do
        TS="$(date '+%d-%m-%Y %H:%M:%S %Z')"
        if [[ -f "$OUTPUT_FILE" ]]; then
            STARTING_LINES=$(wc -l < "$OUTPUT_FILE")
        else
            STARTING_LINES=0
        fi
        echo "----- [$LABEL] Run $i (starting from $STARTING_LINES) ($TS) -----"
        echo "----- [$LABEL] Run $i (starting from $STARTING_LINES) ($TS) -----" >> "$LOG_FILE"

        python3 /fsxnew/shyam.pawar/inference_scripts/async_infer-lazy-buffer_newer_w_s3.py \
            --input-path "$INPUT_DIR" \
            --output-file "$OUTPUT_FILE" \
            --instruction-path /fsxnew/shyam.pawar/inference_scripts/instruction_prompts.yml \
            --task dotsocr_w_layout \
            --backend vllm-chat \
            --extra-request-body '{"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2, "min_p": 0.01, "max_tokens": 8192}' \
            --max-concurrency $((50 * 8 * 150)) \
            --buffer-size $((50 * 8 * 150)) \
            --host 10.67.18.72 \
            --port 20100 \
            || echo "[WARN] [$LABEL] async_infer failed on run $i"

        echo ""

        # wc -l BEFORE post-run
        TS="$(date '+%d-%m-%Y %H:%M:%S %Z')"
        if [[ -f "$OUTPUT_FILE" ]]; then
            BEFORE_LINES=$(wc -l < "$OUTPUT_FILE")
        else
            BEFORE_LINES=0
        fi

        echo "[$TS] [$LABEL] Run=$i BEFORE post-run lines=$BEFORE_LINES"
        echo "[$TS] [$LABEL] Run=$i BEFORE post-run lines=$BEFORE_LINES" >> "$LOG_FILE"

        echo ""
        echo "----- [$LABEL] Post-run $i -----"

        python3 /fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/utils/remove_invalid_json.py \
            -i "$OUTPUT_FILE" \
            || echo "[WARN] [$LABEL] remove_invalid_json failed on run $i"

        echo ""

        # wc -l AFTER post-run
        TS="$(date '+%d-%m-%Y %H:%M:%S %Z')"
        if [[ -f "$OUTPUT_FILE" ]]; then
            AFTER_LINES=$(wc -l < "$OUTPUT_FILE")
        else
            AFTER_LINES=0
        fi

        echo "[$TS] [$LABEL] Run=$i AFTER post-run lines=$AFTER_LINES"
        echo "[$TS] [$LABEL] Run=$i AFTER post-run lines=$AFTER_LINES" >> "$LOG_FILE"

        TS="$(date '+%d-%m-%Y %H:%M:%S %Z')"
        echo ""
        echo "----- [$LABEL] Finished run $i ($TS) -----"
        echo ""
    done
done

echo "ALL JOBS COMPLETED"
