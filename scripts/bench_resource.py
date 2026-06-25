#!/usr/bin/env python3
"""bench_resource.py — resource-aware benchmark harness for OpenSplat.

Runs an OpenSplat binary on a dataset under an optional RAM budget and records
ground-truth measurements: peak host RSS, peak GPU VRAM, wall time, throughput,
final gaussian count and loss. No estimates — every number comes from /proc and
nvidia-smi sampled while the process runs.

RAM-budget enforcement modes (auto-selected, override with --enforce):
  cgroup   : real kernel limit via cgroup v2 memory.max (needs a writable,
             memory-delegated cgroup; the strongest emulation of a low-RAM box).
  watchdog : sample peak RSS of the process tree; if it exceeds the budget, kill
             the tree and flag oom=true. Emulates an OOM event by measurement.
             Used automatically when cgroup creation is not permitted (e.g. the
             Google Colab sandbox, where /sys/fs/cgroup is read-only).
  none     : measure only, never kill.

Usage:
  python3 scripts/bench_resource.py --bin ./output/opensplat \
      --dataset /path/to/scene --budget-gb 4 --extra-args "-n 200" \
      --out linux_research/measurements/run.json

Stdlib only (plus nvidia-smi if present). Python 3.8+.
"""
import argparse, json, os, re, shlex, signal, subprocess, sys, time
from pathlib import Path

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def list_descendants(root_pid):
    """Return [root_pid, ...all descendant pids] by scanning /proc ppid links."""
    children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            ppid = int(fields[1])  # field 4 overall; after "comm)" it's index 1
            children.setdefault(ppid, []).append(int(entry))
        except Exception:
            continue
    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def tree_rss_bytes(root_pid):
    """Sum VmRSS (resident physical memory) over the process tree, in bytes."""
    total = 0
    for pid in list_descendants(root_pid):
        kb = None
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        break
        except Exception:
            continue
        if kb is not None:
            total += kb * 1024
    return total


def gpu_stats(gpu_index=0):
    """(used VRAM MiB, GPU utilization %) in one nvidia-smi call, or (None, None).
    The benchmark assumes the GPU is otherwise idle, which holds on a dedicated
    bench box / fresh Colab VM. GPU-util is the % of the last sample interval the
    GPU had >=1 kernel running — evidence the workload is actually GPU-bound."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}",
             "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        used, util = out.splitlines()[0].split(",")
        return int(used), int(util)
    except Exception:
        return None, None


# --- optional real cgroup enforcement -------------------------------------
def try_make_cgroup(budget_bytes):
    """Create a cgroup v2 with memory.max=budget. Returns path or None."""
    base = Path("/sys/fs/cgroup")
    if not (base / "cgroup.controllers").exists():
        return None
    if "memory" not in (base / "cgroup.controllers").read_text().split():
        return None
    cg = base / f"osplat_bench_{os.getpid()}"
    try:
        cg.mkdir()
        (cg / "memory.max").write_text(str(budget_bytes))
        # swap off so a breach is a hard fail, not silent swapping
        try:
            (cg / "memory.swap.max").write_text("0")
        except Exception:
            pass
        return cg
    except Exception:
        return None


def parse_stdout(text):
    """Extract ground-truth signals from OpenSplat stdout."""
    steps = [int(m) for m in re.findall(r"^Step (\d+):", text, re.M)]
    losses = [float(m) for m in re.findall(r"^Step \d+: ([0-9.eE+-]+)", text, re.M)]
    # Final gaussian count = the last densification ("new count N") or cull
    # ("remaining M") line in CHRONOLOGICAL order (a single regex over the whole
    # stream preserves ordering; concatenating two separate lists would not).
    gaussians = None
    for m in re.finditer(r"(?:new count|remaining) (\d+)", text):
        gaussians = int(m.group(1))
    return {
        "steps_completed": max(steps) if steps else 0,
        "final_loss": losses[-1] if losses else None,
        "gaussians_final": gaussians,
        "device": "cuda" if "Using CUDA" in text else
                  ("mps" if "Using MPS" in text else "cpu"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--budget-gb", type=float, default=0.0,
                    help="RAM budget in GiB; 0 = unlimited (measure only)")
    ap.add_argument("--enforce", choices=["auto", "cgroup", "watchdog", "none"],
                    default="auto")
    ap.add_argument("--poll-ms", type=int, default=50)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--workdir", default="")
    args = ap.parse_args()

    budget_bytes = int(args.budget_gb * (1024 ** 3)) if args.budget_gb > 0 else 0
    cmd = [args.bin, args.dataset] + shlex.split(args.extra_args)

    # choose enforcement mode
    cg = None
    mode = args.enforce
    if budget_bytes and mode in ("auto", "cgroup"):
        cg = try_make_cgroup(budget_bytes)
        if cg:
            mode = "cgroup"
        elif args.enforce == "cgroup":
            print("WARN: cgroup requested but unavailable; falling back to watchdog",
                  file=sys.stderr)
            mode = "watchdog"
        else:
            mode = "watchdog"
    elif not budget_bytes:
        mode = "none"

    env = dict(os.environ)
    workdir = args.workdir or None

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=workdir, env=env, text=True, bufsize=1,
                            start_new_session=True)  # own pgid -> safe killpg
    if cg:
        try:
            (cg / "cgroup.procs").write_text(str(proc.pid))
        except Exception as e:
            print(f"WARN: could not attach pid to cgroup: {e}", file=sys.stderr)

    peak_rss = 0
    peak_vram = 0
    peak_util = 0
    util_sum = 0
    util_n = 0
    vram_polled = False   # distinguish "never sampled" from "sampled, was ~0"
    oom = False
    timed_out = False
    out_lines = []
    last_vram_poll = 0.0
    # Drain stdout in a thread so the pipe never blocks the monitor.
    import threading, queue
    q = queue.Queue()

    def _reader():
        for line in proc.stdout:
            q.put(line)
        q.put(None)
    threading.Thread(target=_reader, daemon=True).start()

    poll = args.poll_ms / 1000.0
    while True:
        if proc.poll() is not None:
            break
        if time.monotonic() - t0 > args.timeout_s:
            timed_out = True
            print("TIMEOUT", file=sys.stderr)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            break
        rss = tree_rss_bytes(proc.pid)
        if rss > peak_rss:
            peak_rss = rss
        now = time.monotonic()
        if now - last_vram_poll > 0.25:  # GPU polling is heavier; 4 Hz
            v, u = gpu_stats(args.gpu_index)
            if v is not None:
                vram_polled = True
                if v > peak_vram:
                    peak_vram = v
            if u is not None:
                peak_util = max(peak_util, u)
                util_sum += u
                util_n += 1
            last_vram_poll = now
        if mode == "watchdog" and budget_bytes and rss > budget_bytes:
            oom = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            break
        while not q.empty():
            ln = q.get_nowait()
            if ln is not None:
                out_lines.append(ln)
        time.sleep(poll)

    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    # drain remaining stdout
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            ln = q.get(timeout=0.1)
        except Exception:
            continue
        if ln is None:
            break
        out_lines.append(ln)
    wall = time.monotonic() - t0
    exit_code = proc.returncode

    # cgroup may have recorded an OOM kill
    if cg:
        events = (cg / "memory.events")
        if events.exists() and re.search(r"oom_kill (\d+)", events.read_text()):
            n = int(re.search(r"oom_kill (\d+)", events.read_text()).group(1))
            if n > 0:
                oom = True
        peak_from_cg = _read_int(cg / "memory.peak")
        if peak_from_cg:
            peak_rss = max(peak_rss, peak_from_cg)
        try:
            (cg).rmdir()
        except Exception:
            pass

    text = "".join(out_lines)
    parsed = parse_stdout(text)
    steps = parsed["steps_completed"]
    iters_per_s = (steps / wall) if (steps and wall > 0) else None

    result = {
        "label": args.label,
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "budget_gb": args.budget_gb,
        "enforce_mode": mode,
        "wall_s": round(wall, 3),
        "exit_code": exit_code,
        "oom": oom,
        "timed_out": timed_out,
        "peak_rss_mb": round(peak_rss / (1024 ** 2), 1),
        "peak_vram_mb": peak_vram if vram_polled else None,
        "gpu_util_peak_pct": peak_util if util_n else None,
        "gpu_util_mean_pct": round(util_sum / util_n, 1) if util_n else None,
        "iters_per_s": round(iters_per_s, 3) if iters_per_s else None,
        **parsed,
        "completed": (exit_code == 0 and not oom and not timed_out),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        # also append tail of log next to it for debugging
        Path(args.out).with_suffix(".log").write_text(text[-20000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
