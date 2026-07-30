#!/usr/bin/env python3
from fabric import ThreadingGroup as Group

# --- CONFIG ---
nodes_file = "/fsx/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]
user = "shyam.pawar"
ssh_key = "/home/shyam.pawar/.ssh/id_rsa"


group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

docker_stop_cmd = (
    'docker ps -q --filter "ancestor=rednotehilab/dots.ocr:vllm-openai-v0.9.1" | '
    'xargs -r docker stop'
)
# tmux_kill_cmd="tmux kill-window -t ocr_w_dots:server"
tmux_kill_cmd="tmux kill-session -t ocr_w_dots"

print("Stopping Docker containers and killing tmux window on all nodes...\n")

docker_results = group.run(docker_stop_cmd, hide=True, warn=True)
tmux_results = group.run(tmux_kill_cmd, hide=True, warn=True)

# --- display results ---
for connection, docker_res in docker_results.items():
    host = connection.host

    if docker_res.ok:
        output = docker_res.stdout.strip()
        status = "success" if output else "no matching containers"
        print(f"[{host}] Docker stop: {status}: {output}")
    else:
        print(f"[{host}] Docker stop failed: {docker_res.stderr.strip()}")

    tmux_res = tmux_results[connection]
    if tmux_res.ok:
        output = tmux_res.stdout.strip()
        status = "success" if output else "no tmux window"
        print(f"[{host}] TMUX kill: {status}: {output}")
    else:
        print(f"[{host}] TMUX kill failed: {tmux_res.stderr.strip()}")
