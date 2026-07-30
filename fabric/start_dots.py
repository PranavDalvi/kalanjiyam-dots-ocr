#!/usr/bin/env python3
import argparse
import sys
from fabric import Connection
from fabric import ThreadingGroup as Group
from concurrent.futures import ThreadPoolExecutor, as_completed

# check by docker container name
# if alr have it, just start it
# stop by name not image


# --- HELPER FUNCTIONS ---
def print_phase_header(phase_num: int, title: str):
    """Print a formatted phase header."""
    print("\n" + "=" * 60)
    print(f"  PHASE {phase_num}: {title}")
    print("=" * 60)

def print_phase_results(results: dict):
    """Print phase results in a formatted way."""
    print("\n📋 Phase Results:")
    print("-" * 40)
    for host, status in results.items():
        icon = "✅" if status["success"] else "❌"
        print(f"  {icon} [{host}] {status['message']}")
    print("-" * 40)

def print_ping_summary(successful: list, failed: list):
    """Print a detailed ping summary."""
    total = len(successful) + len(failed)
    
    print("\n" + "─" * 50)
    print("📡 PING SUMMARY")
    print("─" * 50)
    print(f"  Total Nodes:    {total}")
    print(f"  ✅ Reachable:   {len(successful)}")
    print(f"  ❌ Unreachable: {len(failed)}")
    print("─" * 50)
    
    if successful:
        print("\n  🟢 REACHABLE NODES (will continue):")
        for node in successful:
            print(f"      • {node}")
    
    if failed:
        print("\n  🔴 UNREACHABLE NODES (will be skipped):")
        for node, reason in failed:
            print(f"      • {node} - {reason}")
    
    print("─" * 50)


# --- ARGS ---
parser = argparse.ArgumentParser(description="Deploy OCR tmux environment on nodes (Group-based)")
parser.add_argument("--force", action="store_true", help="Kill existing 'ocr_w_dots' tmux session before recreating it")
args = parser.parse_args()
force = args.force

# --- CONFIG ---
nodes_file = "/fsx/shyam.pawar/OCR_stuff/scripts/w_dots/fabric/nodes.txt"
with open(nodes_file, "r") as f:
    nodes = [line.strip() for line in f if line.strip()]
user = "shyam.pawar"
ssh_key = "/home/shyam.pawar/.ssh/id_rsa"

windows = ["server", "client"]

# Track active hosts (hostnames only)
active_hosts = {}

# ============================================================
# PHASE 1: CONNECTIVITY CHECK (parallel with Group)
# ============================================================
print_phase_header(1, "CONNECTIVITY CHECK")

print(f"\n🔍 Attempting to ping {len(nodes)} node(s)...")
print("─" * 50)

phase1_results = {}
successful_nodes = []
failed_nodes = []

# Show all nodes that will be pinged
print("\n📋 Nodes to ping:")
for node in nodes:
    print(f"   • {node}")

print("\n🔄 Pinging nodes in parallel...\n")

# Use ThreadingGroup to run echo ping in parallel
group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

try:
    group_result = group.run("echo ping", hide=True, warn=True, timeout=10)
except Exception as e:
    # In practice Group.run may raise for severe problems; mark all failed
    err = str(e).splitlines()[0]
    for node in nodes:
        phase1_results[node] = {"success": False, "message": f"Connection error: {err}"}
        failed_nodes.append((node, err))
else:
    # iterate per-host results
    for conn, result in group_result.items():
        host = conn.host
        if result is None:
            # no result (rare)
            phase1_results[host] = {"success": False, "message": "No response"}
            failed_nodes.append((host, "No response"))
        elif result.ok:
            phase1_results[host] = {"success": True, "message": f"Reachable (response: {result.stdout.strip()})"}
            successful_nodes.append(host)
            active_hosts[host] = True  # mark reachable
        else:
            # warn or failed command
            err = (result.stderr or "").strip() or "Command failed"
            phase1_results[host] = {"success": False, "message": f"Command failed: {err}"}
            failed_nodes.append((host, err))

print_ping_summary(successful_nodes, failed_nodes)

# Check if we have any hosts to continue with
if not active_hosts:
    print("\n" + "=" * 50)
    print("🛑 ERROR: No reachable nodes found!")
    print("   All nodes failed connectivity check.")
    print("   Please verify:")
    print("     • Network connectivity")
    print("     • SSH key configuration")
    print("     • Node availability")
    print("=" * 50)
    sys.exit(1)

print(f"\n✨ Continuing deployment with {len(active_hosts)} reachable node(s):")
for node in active_hosts.keys():
    print(f"   → {node}")


# ============================================================
# PHASE 2: PREREQUISITES CHECK (tmux) - parallel
# ============================================================
print_phase_header(2, "PREREQUISITES CHECK (tmux)")

phase2_results = {}
hosts_with_tmux = []

reachable_list = list(active_hosts.keys())
group = Group(*reachable_list, user=user, connect_kwargs={"key_filename": ssh_key})

check = group.run("command -v tmux", hide=True, warn=True)
for conn, result in check.items():
    host = conn.host
    if result and result.ok and result.stdout.strip():
        path = result.stdout.strip()
        phase2_results[host] = {"success": True, "message": f"tmux found at: {path}"}
        hosts_with_tmux.append(host)
    else:
        msg = (result.stderr or "").strip() if result is not None else "tmux not found"
        phase2_results[host] = {"success": False, "message": f"ERROR: tmux not installed ({msg})"}

# Update active_hosts to only those with tmux
active_hosts = {h: True for h in hosts_with_tmux}

print_phase_results(phase2_results)

if not active_hosts:
    print("\n🛑 No hosts with tmux available. Exiting.")
    sys.exit(1)

print(f"\n✨ Continuing with {len(active_hosts)} node(s) that have tmux installed.")


# ============================================================
# PHASE 3: DOCKER IMAGE SETUP - parallel
# ============================================================
print_phase_header(3, "DOCKER IMAGE SETUP")

phase3_results = {}
hosts_with_image = []

group = Group(*active_hosts.keys(), user=user, connect_kwargs={"key_filename": ssh_key})

# The command: if image exists, OK; otherwise load from tar (adjust path if needed)
docker_cmd = (
    "docker image inspect rednotehilab/dots.ocr:vllm-openai-v0.9.1 >/dev/null 2>&1 "
    "|| (docker load -i /fsx/shyam.pawar/docker/images/rednotehilab_dots_ocr__vllm-openai-v091.tar "
    "|| docker pull rednotehilab/dots.ocr:vllm-openai-v0.9.1)"
)

res = group.run(docker_cmd, hide=True, warn=True)
for conn, result in res.items():
    host = conn.host
    if result and result.ok:
        phase3_results[host] = {"success": True, "message": "Docker image ready"}
        hosts_with_image.append(host)
    else:
        err = (result.stderr or "").strip() if result is not None else "Docker image load failed"
        phase3_results[host] = {"success": False, "message": f"ERROR: Docker image load failed ({err})"}

active_hosts = {h: True for h in hosts_with_image}
print_phase_results(phase3_results)

if not active_hosts:
    print("\n🛑 No hosts with Docker image available. Exiting.")
    sys.exit(1)

print(f"\n✨ Continuing with {len(active_hosts)} node(s) with Docker image ready.")


# ============================================================
# PHASE 4: TMUX SESSION SETUP - parallel-ish
# ============================================================
print_phase_header(4, "TMUX SESSION SETUP")

phase4_results = {}
hosts_to_create = []
hosts_already = []
hosts_to_kill = []

group = Group(*active_hosts.keys(), user=user, connect_kwargs={"key_filename": ssh_key})
check_sessions = group.run("tmux has-session -t ocr_w_dots 2>/dev/null", hide=True, warn=True)

for conn, result in check_sessions.items():
    host = conn.host
    if result is not None and result.ok:
        if force:
            hosts_to_kill.append(host)
        else:
            hosts_already.append(host)
    else:
        hosts_to_create.append(host)

# Kill sessions in parallel if requested
if hosts_to_kill:
    kill_group = Group(*hosts_to_kill, user=user, connect_kwargs={"key_filename": ssh_key})
    kill_group.run("tmux kill-session -t ocr_w_dots", hide=True, warn=True)
    # after killing, we will recreate
    hosts_to_create.extend(hosts_to_kill)

# Create sessions (create both session and window)
if hosts_to_create:
    create_group = Group(*hosts_to_create, user=user, connect_kwargs={"key_filename": ssh_key})
    # create a detached session and a client window. chain with && so both happen in one run
    create_cmd = "tmux new-session -d -s ocr_w_dots -n server && tmux new-window -t ocr_w_dots -n client"
    create_res = create_group.run(create_cmd, hide=True, warn=True)
    for conn, result in create_res.items():
        host = conn.host
        if result and result.ok:
            phase4_results[host] = {"success": True, "message": f"Created session 'ocr_w_dots' with windows: {windows}"}
        else:
            err = (result.stderr or "").strip() if result is not None else "Failed to create session"
            phase4_results[host] = {"success": False, "message": f"ERROR: Failed to create session - {err}"}

# Mark hosts that already had sessions (and weren't killed) as success
for host in hosts_already:
    phase4_results[host] = {"success": True, "message": "Session 'ocr_w_dots' already exists (use --force to recreate)"}

# Remove failures from active_hosts
failed_hosts = [h for h, r in phase4_results.items() if not r["success"]]
for h in failed_hosts:
    active_hosts.pop(h, None)

print_phase_results(phase4_results)

if not active_hosts:
    print("\n🛑 No hosts with tmux session available. Exiting.")
    sys.exit(1)

print(f"\n✨ Continuing with {len(active_hosts)} node(s) with tmux session ready.")


# ============================================================
# PHASE 5: PANE CREATION (8 panes for 8 GPUs) - run pane loop on each host in parallel using Group
# ============================================================
print_phase_header(5, "PANE CREATION (8 panes for 8 GPUs)")

phase5_results = {}

# We'll run a small shell loop on each host: create 7 splits and then list panes
pane_cmd = (
    "for i in $(seq 1 7); do "
    "tmux split-window -t ocr_w_dots:server -d || exit 2; "
    "tmux select-layout -t ocr_w_dots:server tiled; "
    "done; "
    "tmux list-panes -t ocr_w_dots:server -F '#D'"
)

group = Group(*active_hosts.keys(), user=user, connect_kwargs={"key_filename": ssh_key})
pane_res = group.run(pane_cmd, hide=True, warn=True)

for conn, result in pane_res.items():
    host = conn.host
    if result and result.ok:
        # count lines in stdout
        panes = [l for l in (result.stdout or "").splitlines() if l.strip()]
        actual_panes = len(panes)
        phase5_results[host] = {"success": True, "message": f"Created {actual_panes} panes in 'server' window"}
    else:
        err = (result.stderr or "").strip() if result is not None else "Pane creation failed"
        phase5_results[host] = {"success": False, "message": f"ERROR: Pane creation failed ({err})"}

# Remove failed hosts
for h, r in list(phase5_results.items()):
    if not r["success"]:
        active_hosts.pop(h, None)

print_phase_results(phase5_results)

if not active_hosts:
    print("\n🛑 No hosts with panes available. Exiting.")
    sys.exit(1)

print(f"\n✨ Continuing with {len(active_hosts)} node(s) with panes ready.")


# ============================================================
# PHASE 6: CONTAINER DEPLOYMENT
# We use a ThreadPoolExecutor to parallelize across hosts, because each host
# needs unique per-pane commands (pane ids differ).
# ============================================================
print_phase_header(6, "CONTAINER DEPLOYMENT (Docker containers on each GPU)")

phase6_results = {}

def deploy_on_host(hostname: str):
    """Deploy containers on a single host (returns (host, result_dict))."""
    conn = Connection(host=hostname, user=user, connect_kwargs={"key_filename": ssh_key})
    try:
        panes_result = conn.run("tmux list-panes -t ocr_w_dots:server -F '#D'", hide=True, warn=True)
        if not panes_result.ok:
            return hostname, {"success": False, "message": "ERROR: Could not list panes"}
        
        panes = [p.strip() for p in panes_result.stdout.strip().splitlines() if p.strip()]
        deployed_count = 0

        for idx, pane in enumerate(panes, start=0):
            cmd = (
                f"""tmux send-keys -t {pane} """
                f""""docker run -it --name DotsOCR_on_gpu{idx}_by_shyam_pawar --rm --pid host --network host --gpus all \
        -v /fsx/shyam.pawar:/fsx/shyam.pawar \
        -v /fsx/opensource-models/weights/DotsOCR:/root/.cache/weights/DotsOCR \
        -e hf_model_path=/root/.cache/weights/DotsOCR/ \
        --entrypoint /bin/bash rednotehilab/dots.ocr:vllm-openai-v0.9.1 \
        -lc 'source /fsx/shyam.pawar/OCR_stuff/scripts/w_dots/patch.sh && \
        CUDA_VISIBLE_DEVICES={idx} vllm serve \${{hf_model_path}} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.8 \
        --chat-template-content-format string \
        --served-model-name model \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port 3010{idx}'" C-m"""
            )

            send = conn.run(cmd, hide=True, warn=False)

            if not send.ok:
                return hostname, {"success": False, "message": f"ERROR: send-keys failed on pane {pane} (GPU {idx})"}
            deployed_count += 1

        return hostname, {"success": True, "message": f"Deployed {deployed_count} containers (ports 3010x)"}
    except Exception as e:
        return hostname, {"success": False, "message": f"Exception: {str(e).splitlines()[0]}"}
    finally:
        conn.close()

# Run per-host deployments concurrently
with ThreadPoolExecutor(max_workers=min(16, len(active_hosts))) as executor:
    futures = {executor.submit(deploy_on_host, host): host for host in active_hosts.keys()}
    for fut in as_completed(futures):
        host = futures[fut]
        try:
            hostname, result = fut.result()
        except Exception as e:
            phase6_results[host] = {"success": False, "message": f"Unhandled exception: {str(e)}"}
        else:
            phase6_results[hostname] = result

print_phase_results(phase6_results)


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  🎉 DEPLOYMENT COMPLETE - FINAL SUMMARY")
print("=" * 60)

# Calculate stats
initial_nodes = len(nodes)
reachable_nodes = len(successful_nodes)
unreachable_nodes = len(failed_nodes)
success_count = sum(1 for r in phase6_results.values() if r["success"])
failed_deployment = reachable_nodes - success_count

print(f"""
┌───────────────────────────────────────────────────┐
│  📊 DEPLOYMENT STATISTICS                         │
├───────────────────────────────────────────────────┤
│  Initial Nodes:           {initial_nodes:<20}     │
│  ├─ Reachable:            {reachable_nodes:<20}   │
│  └─ Unreachable:          {unreachable_nodes:<20} │
│                                                   │
│  Deployment Results:                              │
│  ├─ Successful:           {success_count:<20}     │
│  └─ Failed:               {failed_deployment:<20} │
└───────────────────────────────────────────────────┘
""")

# Show unreachable nodes if any
if failed_nodes:
    print("⚠️  UNREACHABLE NODES (skipped):")
    for node, reason in failed_nodes:
        print(f"   • {node}: {reason}")
    print()

# Show successful deployments
if success_count > 0:
    print("✅ SUCCESSFULLY DEPLOYED NODES:")
    for hostname, result in phase6_results.items():
        if result["success"]:
            print(f"   • {hostname}")
    print()

# Show failed deployments
failed_deployments = [(h, r) for h, r in phase6_results.items() if not r["success"]]
if failed_deployments:
    print("❌ FAILED DEPLOYMENTS:")
    for hostname, result in failed_deployments:
        print(f"   • {hostname}: {result['message']}")
    print()

print("─" * 60)
print("📝 USEFUL COMMANDS:")
print("─" * 60)
print("   Attach to session:    tmux attach -t ocr_w_dots")
print("   View server panes:    tmux select-window -t ocr_w_dots:server")
print("   List all sessions:    tmux list-sessions")
print("   Kill session:         tmux kill-session -t ocr_w_dots")
print("=" * 60)
