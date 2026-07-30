# #!/usr/bin/env python3

# import os
# from pathlib import Path
# from fabric import ThreadingGroup as Group

# # --- CONFIG ---
# nodes_file = "/fsxnew/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
# with open(nodes_file, "r") as f:
#     nodes = [line.strip() for line in f if line.strip()]
# user = "shyam.pawar"
# ssh_key = "/home/shyam.pawar/.ssh/id_rsa"

# nodes_file = SHYAM_MONOREPO_ROOT / "OCR_stuff/scripts/w_dots/fabric/nodes.txt"

# if not nodes_file.exists():
#     raise FileNotFoundError(f"Nodes file not found: {nodes_file}")

# nodes = [
#     line.strip()
#     for line in nodes_file.read_text().splitlines()
#     if line.strip()
# ]

# user = "shyam.pawar"
# ssh_key = Path.home() / ".ssh/id_rsa"

# group = Group(
#     *nodes,
#     user=user,
#     connect_kwargs={"key_filename": str(ssh_key)},
# )

# results = group.run("echo ping", hide=True, warn=True)

# ok_count = fail_count = 0

# for conn, result in results.items():
#     hostname = conn.host

#     success = result.ok
#     ok_count += success
#     fail_count += not success

#     status = "OK" if success else "FAIL"
#     output = (result.stdout or result.stderr or "").strip() or "no output"

#     print(f"{hostname:<25} | {status:<4} | {output}")

# print(f"\nSummary: {ok_count} OK, {fail_count} FAIL, {len(results)} total")


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
results = group.run("echo ping", hide=True, warn=True)

for host, result in results.items():
    if result.ok:
        print(f"[{host}] reachable: {result.stdout.strip()}")
    else:
        print(f"[{host}] unreachable: {result.stderr.strip()}")
