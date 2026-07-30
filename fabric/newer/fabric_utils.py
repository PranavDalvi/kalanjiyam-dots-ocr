#!/usr/bin/env python3
from fabric import ThreadingGroup as Group
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

class Deployment:
    def __init__(self, nodes: List[str], user: str, ssh_key: str):
        self.all_nodes = nodes
        self.user = user
        self.ssh_key = ssh_key

        self.successful_nodes: List[str] = nodes.copy()  # start with all nodes
        self.failed_nodes: List[Tuple[str, str]] = []
        self.phase_results: Dict[str, Dict[str, Dict]] = {}  # phase -> host -> {success,msg}

    # ----------------------------
    # Printing helpers
    # ----------------------------
    @staticmethod
    def print_header(phase_num: int, title: str):
        print("\n" + "=" * 60)
        print(f"  PHASE {phase_num}: {title}")
        print("=" * 60)

    @staticmethod
    def print_results(results: Dict[str, Dict]):
        print("\n📋 PHASE RESULTS")
        print("-" * 40)
        for host, res in results.items():
            icon = "✅" if res["success"] else "❌"
            print(f"  {icon} [{host}] {res['message']}")
        print("-" * 40)

    @staticmethod
    def print_summary(successful: List[str], failed: List[Tuple[str, str]]):
        total = len(successful) + len(failed)
        print("\n" + "─" * 50)
        print("📡 SUMMARY")
        print("─" * 50)
        print(f"  Total Nodes:    {total}")
        print(f"  ✅ Successful:  {len(successful)}")
        print(f"  ❌ Failed:      {len(failed)}")
        print("─" * 50)
        if successful:
            print("\n  🟢 Successful Nodes:")
            for n in successful:
                print(f"      • {n}")
        if failed:
            print("\n  🔴 Failed Nodes:")
            for n, reason in failed:
                print(f"      • {n} - {reason}")
        print("─" * 50)

    # ----------------------------
    # Run a command on all working nodes in parallel
    # ----------------------------
    def run_command_parallel(self, command: str, phase: int, hide: bool = True):
        self.print_header(phase, command)

        if not self.successful_nodes:
            print(f"🛑 No working nodes left for phase {phase}. Skipping.")
            return {}

        phase_result: Dict[str, Dict] = {}
        group = Group(*self.successful_nodes, user=self.user, connect_kwargs={"key_filename": self.ssh_key})

        try:
            res = group.run(command, hide=hide, warn=True)
        except Exception as e:
            err = str(e).splitlines()[0]
            for node in self.successful_nodes:
                phase_result[node] = {"success": False, "message": f"Connection error: {err}"}
        else:
            for conn, result in res.items():
                host = conn.host
                if result and result.ok:
                    phase_result[host] = {"success": True, "message": result.stdout.strip() or "OK"}
                else:
                    err = (result.stderr or "Command failed").strip() if result else "Command failed"
                    phase_result[host] = {"success": False, "message": err}

        # Update internal states
        self.successful_nodes = [h for h, r in phase_result.items() if r["success"]]
        self.failed_nodes.extend([(h, r["message"]) for h, r in phase_result.items() if not r["success"]])
        self.phase_results[f"Phase {phase}"] = phase_result

        self.print_results(phase_result)
        self.print_summary(self.successful_nodes, self.failed_nodes)
        return phase_result

    # ----------------------------
    # Run a command per host concurrently
    # ----------------------------
    def run_command_per_host(self, command_template: str, phase: int, max_workers: int = 16):
        """
        command_template: shell command string, can use {host} placeholder
        """
        self.print_header(phase, command_template)

        if not self.successful_nodes:
            print(f"🛑 No working nodes left for phase {phase}. Skipping.")
            return {}

        phase_result: Dict[str, Dict] = {}

        def run_on_host(host: str):
            from fabric import Connection
            conn = Connection(host=host, user=self.user, connect_kwargs={"key_filename": self.ssh_key})
            try:
                cmd = command_template.format(host=host)
                res = conn.run(cmd, hide=True, warn=True)
                if res.ok:
                    return {"success": True, "message": res.stdout.strip() or "OK"}
                else:
                    return {"success": False, "message": (res.stderr or "Command failed").strip()}
            except Exception as e:
                return {"success": False, "message": str(e).splitlines()[0]}
            finally:
                conn.close()

        with ThreadPoolExecutor(max_workers=min(max_workers, len(self.successful_nodes))) as executor:
            futures = {executor.submit(run_on_host, host): host for host in self.successful_nodes}
            for fut in futures:
                host = futures[fut]
                phase_result[host] = futures[fut].result() if fut.done() else {"success": False, "message": "Unknown error"}

        # Update internal states
        self.successful_nodes = [h for h, r in phase_result.items() if r["success"]]
        self.failed_nodes.extend([(h, r["message"]) for h, r in phase_result.items() if not r["success"]])
        self.phase_results[f"Phase {phase}"] = phase_result

        self.print_results(phase_result)
        self.print_summary(self.successful_nodes, self.failed_nodes)
        return phase_result

    # ----------------------------
    # Final summary
    # ----------------------------
    def print_final_summary(self):
        print("\n" + "=" * 60)
        print("  🎉 DEPLOYMENT COMPLETE - FINAL SUMMARY")
        print("=" * 60)

        total_nodes = len(self.all_nodes)
        successful_count = len(self.successful_nodes)
        failed_count = len(self.failed_nodes)

        print(f"""
┌─────────────────────────────────────────────────┐
│  📊 DEPLOYMENT STATISTICS                       │
├─────────────────────────────────────────────────┤
│  Initial Nodes:           {total_nodes:<20} │
│  ✅ Successful:           {successful_count:<20} │
│  ❌ Failed:               {failed_count:<20} │
└─────────────────────────────────────────────────┘
""")

        if self.successful_nodes:
            print("✅ SUCCESSFUL NODES:")
            for host in self.successful_nodes:
                print(f"   • {host}")
            print()

        if self.failed_nodes:
            print("❌ FAILED NODES:")
            for host, reason in self.failed_nodes:
                print(f"   • {host}: {reason}")
            print()

        print("─" * 60)
        print("📝 USEFUL COMMANDS:")
        print("   tmux attach -t ocr_w_dots")
        print("   tmux list-sessions")
        print("   tmux kill-session -t ocr_w_dots")
        print("=" * 60)
