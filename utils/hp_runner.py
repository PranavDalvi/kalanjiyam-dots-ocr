"""
High-performance parallel runner for massive line-based workloads.

Features:
- NUMA-aware workers
- Streaming input
- Backpressure
- Lock-free writing
- Ordered output
- tqdm progress bar
"""

import os
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Iterable, Tuple, Any

# ---------------- NUMA ----------------

NUMA_CPU_MAP = {
    0: list(range(0, 56)) + list(range(112, 168)),
    1: list(range(56, 112)) + list(range(168, 224)),
}

def pin_to_numa_node(node_id: int):
    try:
        os.sched_setaffinity(0, set(NUMA_CPU_MAP[node_id]))
    except Exception:
        pass


# ---------------- WORKER ----------------

def _worker_batch(
    batch,
    worker_id: int,
    numa_node: int,
    user_fn: Callable[[str], Any],
):
    pin_to_numa_node(numa_node)

    out = []
    for idx, line in batch:
        try:
            result = user_fn(line)
        except Exception as e:
            result = e
        out.append((idx, result))
    return out


# ---------------- RUNNER ----------------

def run_parallel(
    input_path: Path,
    output_path: Path,
    user_fn: Callable[[str], Any],
    *,
    batch_size: int = 50_000,
    workers: int | None = None,
    max_inflight: int | None = None,
    show_progress: bool = True,
):
    """
    Generic high-performance runner.

    user_fn: function(line: str) -> Any
    """

    if workers is None:
        workers = max(1, mp.cpu_count() - 24)
    if max_inflight is None:
        max_inflight = workers

    worker_numa = [i % 2 for i in range(workers)]

    # Optional progress bar
    progress = None
    if show_progress:
        try:
            from tqdm import tqdm
            total = sum(1 for _ in input_path.open("rb"))
            progress = tqdm(total=total, unit="lines", desc="Processing")
        except Exception:
            progress = None

    with (
        mp.Pool(workers) as pool,
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
    ):
        pending = []
        batch = []
        buffer = {}
        next_write_idx = 0
        line_idx = 0
        worker_id = 0

        def submit(b):
            return pool.apply_async(
                _worker_batch,
                (b, worker_id, worker_numa[worker_id], user_fn),
            )

        for line in fin:
            batch.append((line_idx, line))
            line_idx += 1

            if progress:
                progress.update(1)

            if len(batch) >= batch_size:
                pending.append(submit(batch))
                worker_id = (worker_id + 1) % workers
                batch = []

            while len(pending) >= max_inflight:
                fut = pending.pop(0)
                for idx, res in fut.get():
                    buffer[idx] = res

                while next_write_idx in buffer:
                    fout.write(str(buffer.pop(next_write_idx)) + "\n")
                    next_write_idx += 1

        if batch:
            pending.append(submit(batch))

        for fut in pending:
            for idx, res in fut.get():
                buffer[idx] = res

            while next_write_idx in buffer:
                fout.write(str(buffer.pop(next_write_idx)) + "\n")
                next_write_idx += 1

        if progress:
            progress.close()
