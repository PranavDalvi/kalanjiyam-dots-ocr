#!/usr/bin/env python3
from fabric_utils import Deployment

# ------------------------------
# CONFIG
# ------------------------------
nodes_file = "/fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/newer/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]

user = "shyam_pawar"
ssh_key = "/home/shyam_pawar/.ssh/id_ecdsa"
force = True

deploy = Deployment(nodes, user, ssh_key)

# PHASES
deploy.run_command_parallel("echo ping", phase=1)
deploy.run_command_parallel("command -v tmux", phase=2)

docker_cmd = (
    "docker image inspect rednotehilab/dots.ocr:vllm-openai-v0.9.1 >/dev/null 2>&1 || "
    "docker load -i /fsxnew/shyam.pawar/docker/images/rednotehilab_dots_ocr__vllm-openai-v091.tar"
)
deploy.run_command_parallel(docker_cmd, phase=3, hide=False)

# TMUX session check / create
tmux_cmd = (
    f"tmux has-session -t ocr_w_dots 2>/dev/null || "
    f"{'tmux kill-session -t ocr_w_dots && ' if force else ''}"
    "tmux new-session -d -s ocr_w_dots -n server && tmux new-window -t ocr_w_dots -n client"
)
deploy.run_command_per_host(tmux_cmd, phase=4)

# Create panes
pane_cmd = (
    "for i in $(seq 1 7); do "
    "tmux split-window -t ocr_w_dots:server -d || exit 2; "
    "tmux select-layout -t ocr_w_dots:server tiled; "
    "done; tmux list-panes -t ocr_w_dots:server -F '#D'"
)
deploy.run_command_parallel(pane_cmd, phase=5)

docker_cmd_template = (
    "docker run -it --name DotsOCR_on_gpu{gpu_idx} --rm --pid host --network host --gpus all "
    "-v /fsxnew/shyam.pawar:/fsxnew/shyam.pawar "
    "-v /fsxnew/opensource-models/weights/DotsOCR:/root/.cache/weights/DotsOCR "
    "-e hf_model_path=/root/.cache/weights/DotsOCR/ "
    "--entrypoint /bin/bash rednotehilab/dots.ocr:vllm-openai-v0.9.1 "
    "-lc 'source /fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/patch.sh && "
    "CUDA_VISIBLE_DEVICES={gpu_idx} vllm serve \${hf_model_path} "
    "--tensor-parallel-size 1 "
    "--gpu-memory-utilization 0.8 "
    "--chat-template-content-format string "
    "--served-model-name model "
    "--trust-remote-code "
    "--host 0.0.0.0 "
    "--port 3010{gpu_idx}'"
)

container_cmd = f"""
panes=$(tmux list-panes -t ocr_w_dots:server -F '#D')
i=0
for p in $panes; do
    tmux send-keys -t $p "{docker_cmd_template.replace('{gpu_idx}', '$i')}" C-m
    i=$((i+1))
done
"""

# Deploy containers to pane
deploy.run_command_per_host(container_cmd, phase=6)

deploy.print_final_summary()
