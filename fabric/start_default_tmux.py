#!/usr/bin/env python3
from fabric import ThreadingGroup as Group

# --- CONFIG ---
nodes_file = "/fsx/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]
user = "shyam.pawar"
ssh_key = "/home/shyam.pawar/.ssh/id_rsa"

group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

# --- Run echo ping on all nodes in parallel ---
results = group.run("tmux new -s default -d", hide=True, warn=True)

for host, result in results.items():
    if result.ok:
        print(f"[{host}]: {result.stdout.strip()}")
    else:
        print(f"[{host}]: {result.stderr.strip()}")

