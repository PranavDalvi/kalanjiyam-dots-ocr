from fabric import ThreadingGroup as Group

# --- CONFIG ---
nodes = [
    "bharatgpt072",
    "bharatgpt073",
    "bharatgpt074",
    "bharatgpt075",
    "bharatgpt076",
    "bharatgpt077",
    "bharatgpt078",
    "bharatgpt079",
    "bharatgpt121",
    "bharatgpt122",
    "bharatgpt123",
    "bharatgpt125",
    "bharatgpt127",
    "bharatgpt130",
    "bharatgpt131",
    "bharatgpt132",
    "bharatgpt005",
    "bharatgpt006",
    "bharatgpt007",
    "bharatgpt008",
    "bharatgpt009",
    "bharatgpt012",
    "bharatgpt013",
    "bharatgpt014",
    "bharatgpt016",
    "bharatgpt017",
    "bharatgpt018",
    "bharatgpt020",
    "bharatgpt024",
    "bharatgpt025",
    "bharatgpt026",
    "bharatgpt032",
    "bharatgpt034",
    "bharatgpt039",
    "bharatgpt040",
    "bharatgpt045",
    "bharatgpt046",
    "bharatgpt047",
    "bharatgpt048",
    "bharatgpt049",
    "bharatgpt053",
    "bharatgpt057",
    "bharatgpt059",
    "bharatgpt062",
    "bharatgpt068",
    "bharatgpt081",
    "bharatgpt082",
    "bharatgpt083",
    "bharatgpt085",
    "bharatgpt086",
    "bharatgpt087",
    "bharatgpt088",
    "bharatgpt089",
    "bharatgpt090"
]

user = "shyam_pawar"
ssh_key = "/home/shyam_pawar/.ssh/id_ecdsa"

group = Group(*nodes, user=user, connect_kwargs={"key_filename": ssh_key})

# --- Check for container named 'fabric' ---
cmd = "docker ps --format '{{.Names}}' | grep -w fabric || echo 'NOT_FOUND'"

results = group.run(cmd, hide=True, warn=True)

for host, result in results.items():
    output = result.stdout.strip()

    if "fabric" in output:
        print(f"[{host}] ✅ container 'fabric' is RUNNING")
    else:
        print(f"[{host}] ❌ container 'fabric' NOT found (running)")
