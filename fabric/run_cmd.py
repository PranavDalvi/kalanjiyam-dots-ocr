#!/usr/bin/env python3
from fabric import ThreadingGroup as Group

# --- CONFIG ---
nodes_file = "/fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]
user = "shyam_pawar"
ssh_key = "/home/shyam_pawar/.ssh/id_ecdsa"

group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

# --- Run echo ping on all nodes in parallel ---

results = group.run("docker ps -a --filter \"name=language_classifier_w_ray\"", hide=True, warn=True)


for host, result in results.items():
    if result.ok:
        print(f"[{host}] reachable: {result.stdout.strip()}")
    else:
        print(f"[{host}] unreachable: {result.stderr.strip()}")
