#!/usr/bin/env python3
from fabric import ThreadingGroup as Group

# --- CONFIG ---
nodes_file = "/fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]

user = "shyam_pawar"
ssh_key = "/home/shyam_pawar/.ssh/id_ecdsa"

group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

docker_stop_cmd = (
    'docker ps -q --filter "ancestor=vllm/vllm-openai:nightly" | '
    'xargs -r docker stop'
)

print("stopping docker containers on all nodes...\n")
results = group.run(docker_stop_cmd, hide=True, warn=True)

# --- display results ---
for host, result in results.items():
    if result.ok:
        output = result.stdout.strip()
        status = "success" if output else "no matching containers or session"
        print(f"[{host}] {status}: {output}")
    else:
        print(f"[{host}] failed: {result.stderr.strip()}")
