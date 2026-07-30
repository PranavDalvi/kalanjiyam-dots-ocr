#!/usr/bin/env bash
set -eo pipefail

# Ensure model config.json has auto_map entries for DotsOCR custom architecture.
# This is the standard mechanism transformers + vLLM use to resolve custom models.
# Without auto_map, vLLM cannot find DotsOCRForCausalLM from a local model path.

MODEL_DIR="${MODEL_DIR:-/root/.cache/weights/DotsOCR}"
CONFIG_FILE="$MODEL_DIR/config.json"

if [ -f "$CONFIG_FILE" ]; then
  if ! python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
am = cfg.get('auto_map', {})
if 'AutoModelForCausalLM' not in am:
    sys.exit(1)
" 2>/dev/null; then
    echo "[patch.sh] Fixing config.json auto_map for DotsOCR..."
    python3 -c "
import json
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)
if 'auto_map' not in cfg:
    cfg['auto_map'] = {}
cfg['auto_map'].setdefault('AutoConfig', 'configuration_dots.DotsOCRConfig')
cfg['auto_map'].setdefault('AutoModelForCausalLM', 'modeling_dots_ocr.DotsOCRForCausalLM')
with open('$CONFIG_FILE', 'w') as f:
    json.dump(cfg, f, indent=2)
print('[patch.sh] config.json auto_map updated successfully.')
"
  else
    echo "[patch.sh] config.json auto_map already correct."
  fi
else
  echo "[patch.sh] No config.json found at $CONFIG_FILE (model not yet downloaded)."
fi
