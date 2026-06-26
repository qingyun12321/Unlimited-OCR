#!/usr/bin/env python3
"""Run a command while sampling RAM and NVIDIA GPU metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

try:
    import pynvml
except Exception:  # pragma: no cover - optional on non-NVIDIA hosts
    pynvml = None


def mib(value: int | float) -> float:
    return round(float(value) / (1024 * 1024), 2)


def tail_text(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def process_tree_memory(root: psutil.Process) -> dict[str, Any]:
    rss = 0
    vms = 0
    pids: list[int] = []
    for proc in [root, *root.children(recursive=True)]:
        try:
            mem = proc.memory_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        pids.append(proc.pid)
        rss += mem.rss
        vms += mem.vms
    return {"rss_bytes": rss, "vms_bytes": vms, "pids": pids}


def init_nvml() -> Any | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def sample_gpu(handle: Any | None) -> dict[str, Any] | None:
    if handle is None:
        return None
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            power_w = None
        return {
            "memory_used_bytes": int(mem.used),
            "memory_total_bytes": int(mem.total),
            "gpu_util_percent": int(util.gpu),
            "mem_util_percent": int(util.memory),
            "temperature_c": int(temp),
            "power_w": power_w,
        }
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="JSON metrics output path")
    parser.add_argument("--log", required=True, help="Child stdout/stderr log path")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    metrics_path = Path(args.metrics)
    log_path = Path(args.log)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handle = init_nvml()
    baseline_gpu = sample_gpu(handle)
    baseline_ram = psutil.virtual_memory()

    started = time.time()
    samples = 0
    peak_tree_rss = 0
    peak_tree_vms = 0
    peak_system_ram_used = baseline_ram.used
    peak_gpu_mem_used = baseline_gpu["memory_used_bytes"] if baseline_gpu else None
    peak_gpu_util = 0
    peak_gpu_mem_util = 0
    peak_gpu_temp = 0
    peak_gpu_power = 0.0
    last_pids: list[int] = []

    print("Running:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        root = psutil.Process(proc.pid)
        try:
            while proc.poll() is None:
                samples += 1
                try:
                    tree = process_tree_memory(root)
                    peak_tree_rss = max(peak_tree_rss, tree["rss_bytes"])
                    peak_tree_vms = max(peak_tree_vms, tree["vms_bytes"])
                    last_pids = tree["pids"]
                except psutil.NoSuchProcess:
                    pass

                vm = psutil.virtual_memory()
                peak_system_ram_used = max(peak_system_ram_used, vm.used)

                gpu = sample_gpu(handle)
                if gpu:
                    peak_gpu_mem_used = max(peak_gpu_mem_used or 0, gpu["memory_used_bytes"])
                    peak_gpu_util = max(peak_gpu_util, gpu["gpu_util_percent"])
                    peak_gpu_mem_util = max(peak_gpu_mem_util, gpu["mem_util_percent"])
                    peak_gpu_temp = max(peak_gpu_temp, gpu["temperature_c"])
                    if gpu["power_w"] is not None:
                        peak_gpu_power = max(peak_gpu_power, float(gpu["power_w"]))
                time.sleep(args.interval)
        finally:
            returncode = proc.wait()

    ended = time.time()
    final_gpu = sample_gpu(handle)
    final_ram = psutil.virtual_memory()
    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    metrics = {
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": round(ended - started, 3),
        "samples": samples,
        "sampling_interval_seconds": args.interval,
        "process_tree_peak_rss_mib": mib(peak_tree_rss),
        "process_tree_peak_vms_mib": mib(peak_tree_vms),
        "system_ram_baseline_used_mib": mib(baseline_ram.used),
        "system_ram_peak_used_mib": mib(peak_system_ram_used),
        "system_ram_final_used_mib": mib(final_ram.used),
        "gpu_baseline_used_mib": mib(baseline_gpu["memory_used_bytes"]) if baseline_gpu else None,
        "gpu_peak_used_mib": mib(peak_gpu_mem_used) if peak_gpu_mem_used is not None else None,
        "gpu_final_used_mib": mib(final_gpu["memory_used_bytes"]) if final_gpu else None,
        "gpu_total_mib": mib(final_gpu["memory_total_bytes"]) if final_gpu else (mib(baseline_gpu["memory_total_bytes"]) if baseline_gpu else None),
        "gpu_peak_increment_mib": mib((peak_gpu_mem_used or 0) - baseline_gpu["memory_used_bytes"]) if baseline_gpu and peak_gpu_mem_used is not None else None,
        "gpu_peak_util_percent": peak_gpu_util,
        "gpu_peak_memory_util_percent": peak_gpu_mem_util,
        "gpu_peak_temperature_c": peak_gpu_temp,
        "gpu_peak_power_w": round(peak_gpu_power, 2),
        "last_observed_pids": last_pids,
        "log_path": str(log_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("\n--- child log tail ---", flush=True)
    print(tail_text(log_path), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
