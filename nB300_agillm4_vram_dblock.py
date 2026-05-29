#!/usr/bin/env python3

# n.py - Joint AR+SAT+NAT Trainer with Expansion Ratio Testing
# Enhanced inference: checkpoint name, tok/s, UK time

from __future__ import annotations
import argparse, copy, json, math, pathlib, random, time, os, sys, threading, hashlib, re, subprocess
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

_ASCII_LOG_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
})


def _ascii_log_text(text: str) -> str:
    return str(text).translate(_ASCII_LOG_TRANSLATION).encode("ascii", "replace").decode("ascii")


class _AsciiLogStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        return self._wrapped.write(_ascii_log_text(text))

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def fileno(self):
        return self._wrapped.fileno()

    @property
    def encoding(self):
        return "ascii"

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


if (
    not sys.stdout.isatty()
    and os.environ.get("NB300_RAW_UNICODE_LOGS", "").lower() not in {"1", "true", "yes"}
):
    sys.stdout = _AsciiLogStream(sys.stdout)
    sys.stderr = _AsciiLogStream(sys.stderr)

STATUS_SCRIPT_PATH = Path(__file__).resolve()
STATUS_DEFAULT_LOG = STATUS_SCRIPT_PATH.parent / "train.log"
STATUS_DEFAULT_SAVE_DIR = STATUS_SCRIPT_PATH.parent / "ckpts_expansion"
_STATUS_PROGRESS_RE = re.compile(
    r"^\[(?P<percent>\d+(?:\.\d+)?)%\]\s+"
    r"(?P<seen>[\d,]+)/(?P<target>[\d,]+)\s+tok\s+\|\s+"
    r"(?P<tok_s>[\d.]+)\s+tok/s\s+\|\s+"
    r"loss=(?P<loss>-?[\d.]+)\s+B=(?P<batch>\d+)\s+L=(?P<block>\d+)"
    r"(?:\s+step=(?P<step>\d+))?"
    r"(?:\s+eta=(?P<eta>\S+))?"
    r"(?:\s+elapsed=(?P<elapsed>\S+))?"
    r"\s*$"
)
_STATUS_DELTA_RE = re.compile(r"\[delta\]\s+saved\s+(?P<name>\S+?\.pt)\s+\((?P<sha>[0-9a-f]+)\.\.\.\)")
_STATUS_STEP_RE = re.compile(r"step(?P<step>\d+)")


def _status_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _status_human_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _status_compact_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    try:
        if not math.isfinite(float(seconds)):
            return "unknown"
    except Exception:
        return "unknown"
    total = max(0, int(seconds))
    years, rem = divmod(total, 365 * 86400)
    days, rem = divmod(rem, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if years:
        return f"{years}y{days}d{hours}h"
    if days:
        return f"{days}d{hours}h{minutes}m"
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def _status_format_int(value: Optional[int]) -> str:
    return "?" if value is None else f"{value:,}"


def _status_parse_step(text: str) -> Optional[int]:
    match = _STATUS_STEP_RE.search(text)
    return int(match.group("step")) if match else None


def _status_resolve_ckpt_path(raw_path: str, base_dir: Path) -> Path:
    ckpt_path = Path(raw_path)
    return ckpt_path if ckpt_path.is_absolute() else (base_dir / ckpt_path).resolve()


def _status_read_cmdline(proc_dir: Path) -> Optional[List[str]]:
    try:
        data = (proc_dir / "cmdline").read_bytes().split(b"\0")
        return [item.decode("utf-8", errors="ignore") for item in data if item]
    except Exception:
        return None


def _status_resolve_proc_arg(proc_dir: Path, raw_arg: str) -> Optional[Path]:
    try:
        arg_path = Path(raw_arg)
        if arg_path.is_absolute():
            return arg_path.resolve()
        cwd = Path(os.readlink(proc_dir / "cwd"))
        return (cwd / arg_path).resolve()
    except Exception:
        return None


def _status_proc_uptime(proc_dir: Path) -> Optional[float]:
    try:
        proc_uptime = float((Path("/proc") / "uptime").read_text().split()[0])
        stat_text = (proc_dir / "stat").read_text()
        after = stat_text[stat_text.rfind(")") + 2:].split()
        start_ticks = float(after[19])
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return max(0.0, proc_uptime - (start_ticks / clock_ticks))
    except Exception:
        return None


def _status_find_trainers(script_path: Path) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        args = _status_read_cmdline(proc_dir)
        if not args or "train" not in args:
            continue
        resolved_script = None
        for arg in args:
            if Path(arg).name != script_path.name:
                continue
            candidate = _status_resolve_proc_arg(proc_dir, arg)
            if candidate == script_path:
                resolved_script = candidate
                break
        if resolved_script is None:
            continue
        uptime_seconds = _status_proc_uptime(proc_dir)
        try:
            cwd = str(Path(os.readlink(proc_dir / "cwd")))
        except Exception:
            cwd = None
        matches.append({
            "pid": int(proc_dir.name),
            "cmdline": " ".join(args),
            "args": args,
            "cwd": cwd,
            "uptime_seconds": round(uptime_seconds, 3) if uptime_seconds is not None else None,
            "uptime_human": _status_human_duration(uptime_seconds),
        })
    return sorted(matches, key=lambda item: item["pid"])


def _status_parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    match = _STATUS_PROGRESS_RE.match(line.strip())
    if not match:
        return None
    tok_per_sec = float(match.group("tok_s"))
    loss = float(match.group("loss"))
    return {
        "raw_line": line.strip(),
        "percent": float(match.group("percent")),
        "seen_tokens": int(match.group("seen").replace(",", "")),
        "target_tokens": int(match.group("target").replace(",", "")),
        "tok_per_sec": int(tok_per_sec) if tok_per_sec.is_integer() else tok_per_sec,
        "loss": loss,
        "batch": int(match.group("batch")),
        "block": int(match.group("block")),
        "step": int(match.group("step")) if match.group("step") else None,
        "eta": match.group("eta"),
        "elapsed": match.group("elapsed"),
    }


def _status_parse_delta_line(line: str) -> Optional[Dict[str, Any]]:
    match = _STATUS_DELTA_RE.search(line)
    if not match:
        return None
    name = match.group("name")
    return {
        "raw_line": line.strip(),
        "name": name,
        "step": _status_parse_step(name),
        "sha_prefix": match.group("sha"),
        "source": "log",
    }


def _status_scan_log(log_path: Path) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    now = time.time()
    info: Dict[str, Any] = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "mtime": None,
        "mtime_iso": None,
        "age_seconds": None,
        "age_human": None,
        "size_bytes": None,
    }
    warnings: List[str] = []
    if not log_path.exists():
        warnings.append(f"train log missing: {log_path}")
        return info, None, None, warnings
    try:
        st = log_path.stat()
        info["mtime"] = st.st_mtime
        info["mtime_iso"] = _status_iso(st.st_mtime)
        info["age_seconds"] = round(max(0.0, now - st.st_mtime), 3)
        info["age_human"] = _status_human_duration(info["age_seconds"])
        info["size_bytes"] = st.st_size
    except Exception as exc:
        warnings.append(f"failed to stat train log: {exc}")
    last_progress = None
    last_delta = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                progress = _status_parse_progress_line(line)
                if progress is not None:
                    last_progress = progress
                delta = _status_parse_delta_line(line)
                if delta is not None:
                    last_delta = delta
    except Exception as exc:
        warnings.append(f"failed to read train log: {exc}")
    return info, last_progress, last_delta, warnings


def _status_latest_full_checkpoint(save_dir: Path, base_dir: Path) -> tuple[Dict[str, Any], List[str]]:
    latest_path = save_dir / "latest.json"
    info: Dict[str, Any] = {
        "metadata_path": str(latest_path),
        "exists": latest_path.exists(),
        "raw_path": None,
        "checkpoint_path": None,
        "checkpoint_name": None,
        "checkpoint_exists": None,
        "step": None,
        "checkpoint_mtime": None,
        "checkpoint_mtime_iso": None,
    }
    warnings: List[str] = []
    if not latest_path.exists():
        warnings.append(f"latest.json missing: {latest_path}")
        return info, warnings
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to parse latest.json: {exc}")
        return info, warnings
    raw_path = payload.get("path")
    info["raw_path"] = raw_path
    info["step"] = payload.get("step")
    if raw_path:
        ckpt_path = _status_resolve_ckpt_path(raw_path, base_dir)
        info["checkpoint_path"] = str(ckpt_path)
        info["checkpoint_name"] = ckpt_path.name
        info["checkpoint_exists"] = ckpt_path.exists()
        if ckpt_path.exists():
            try:
                st = ckpt_path.stat()
                info["checkpoint_mtime"] = st.st_mtime
                info["checkpoint_mtime_iso"] = _status_iso(st.st_mtime)
            except Exception as exc:
                warnings.append(f"failed to stat full checkpoint: {exc}")
        else:
            warnings.append(f"latest.json points to missing checkpoint: {ckpt_path}")
    return info, warnings


def _status_newest_delta(save_dir: Path) -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not save_dir.exists():
        warnings.append(f"save dir missing: {save_dir}")
        return None, warnings
    try:
        candidates = [item for item in save_dir.glob("*_delta_step*.pt") if item.is_file()]
    except Exception as exc:
        warnings.append(f"failed to list delta checkpoints: {exc}")
        return None, warnings
    if not candidates:
        warnings.append(f"no delta checkpoints found in {save_dir}")
        return None, warnings
    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    st = newest.stat()
    return {
        "path": str(newest),
        "name": newest.name,
        "step": _status_parse_step(newest.name),
        "mtime": st.st_mtime,
        "mtime_iso": _status_iso(st.st_mtime),
        "size_bytes": st.st_size,
        "source": "disk",
    }, warnings


def _status_gpu_info() -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return None, warnings
    except Exception as exc:
        warnings.append(f"failed to query GPU status: {exc}")
        return None, warnings
    if result.returncode != 0:
        warnings.append(result.stderr.strip() or "nvidia-smi returned non-zero exit status")
        return None, warnings
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None, warnings
    if len(lines) > 1:
        warnings.append("multiple GPUs detected; reporting the first GPU only")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 6:
        warnings.append(f"unexpected nvidia-smi format: {lines[0]}")
        return None, warnings

    def _parse_int(raw: str) -> Optional[int]:
        try:
            return int(float(raw))
        except Exception:
            return None

    def _parse_float(raw: str) -> Optional[float]:
        try:
            return float(raw)
        except Exception:
            return None

    return {
        "name": parts[0],
        "utilization_gpu": _parse_int(parts[1]),
        "memory_used_mib": _parse_int(parts[2]),
        "memory_total_mib": _parse_int(parts[3]),
        "temperature_c": _parse_int(parts[4]),
        "power_draw_w": _parse_float(parts[5]),
    }, warnings


def _status_choose_delta(from_log: Optional[Dict[str, Any]], from_disk: Optional[Dict[str, Any]], warnings: List[str]) -> Optional[Dict[str, Any]]:
    if from_log and from_disk:
        log_step = from_log.get("step")
        disk_step = from_disk.get("step")
        if log_step is not None and disk_step is not None:
            if log_step != disk_step:
                warnings.append(
                    f"log delta step {log_step} and newest on-disk delta step {disk_step} differ; using the newer step"
                )
            if disk_step >= log_step:
                merged = dict(from_disk)
                merged["source"] = "disk+log" if disk_step == log_step else "disk"
                if disk_step == log_step:
                    merged["sha_prefix"] = from_log.get("sha_prefix")
                return merged
            return dict(from_log)
        return dict(from_disk)
    if from_disk:
        return dict(from_disk)
    if from_log:
        return dict(from_log)
    return None


def _collect_status(log_path: Path, save_dir: Path) -> tuple[Dict[str, Any], int]:
    checked_at = time.time()
    requested_save_dir = save_dir.expanduser()
    log_path = log_path.expanduser()
    status: Dict[str, Any] = {
        "checked_at": checked_at,
        "checked_at_iso": _status_iso(checked_at),
        "running": False,
        "process": None,
        "progress": None,
        "delta_checkpoint": None,
        "delta_from_log": None,
        "delta_on_disk": None,
        "latest_full_checkpoint": None,
        "log": None,
        "gpu": None,
        "save_dir": {
            "requested_path": str(requested_save_dir),
            "path": str(requested_save_dir),
            "exists": requested_save_dir.exists(),
            "source": "requested",
        },
        "warnings": [],
    }
    warnings = status["warnings"]

    matches = _status_find_trainers(STATUS_SCRIPT_PATH)
    if len(matches) > 1:
        status["error"] = "multiple active n.py train processes found"
        status["processes"] = matches
        return status, 1
    if matches:
        status["running"] = True
        status["process"] = matches[0]

    save_dir = requested_save_dir
    if status["process"] and status["process"].get("cwd"):
        proc_cwd = Path(status["process"]["cwd"])
        alt_save_dir = (proc_cwd / requested_save_dir.name).resolve()
        if alt_save_dir != requested_save_dir and alt_save_dir.exists():
            requested_delta, _ = _status_newest_delta(requested_save_dir)
            requested_full, _ = _status_latest_full_checkpoint(requested_save_dir, STATUS_SCRIPT_PATH.parent)
            alt_delta, _ = _status_newest_delta(alt_save_dir)
            alt_full, _ = _status_latest_full_checkpoint(alt_save_dir, proc_cwd)
            requested_score = int(requested_delta is not None) + int(bool(requested_full.get("checkpoint_exists")))
            alt_score = int(alt_delta is not None) + int(bool(alt_full.get("checkpoint_exists")))
            if alt_score > requested_score:
                save_dir = alt_save_dir
                status["save_dir"] = {
                    "requested_path": str(requested_save_dir),
                    "path": str(save_dir),
                    "exists": save_dir.exists(),
                    "source": "process_cwd_fallback",
                }
                warnings.append(
                    f"using process cwd save dir fallback: {save_dir} (requested {requested_save_dir})"
                )

    log_info, progress, delta_from_log, log_warnings = _status_scan_log(log_path)
    warnings.extend(log_warnings)
    status["log"] = log_info
    status["progress"] = progress
    status["delta_from_log"] = delta_from_log

    latest_base_dir = STATUS_SCRIPT_PATH.parent
    if status["save_dir"].get("source") == "process_cwd_fallback" and status["process"] and status["process"].get("cwd"):
        latest_base_dir = Path(status["process"]["cwd"])
    latest_full, latest_warnings = _status_latest_full_checkpoint(save_dir, latest_base_dir)
    warnings.extend(latest_warnings)
    status["latest_full_checkpoint"] = latest_full

    delta_on_disk, delta_warnings = _status_newest_delta(save_dir)
    warnings.extend(delta_warnings)
    status["delta_on_disk"] = delta_on_disk
    status["delta_checkpoint"] = _status_choose_delta(delta_from_log, delta_on_disk, warnings)

    gpu, gpu_warnings = _status_gpu_info()
    warnings.extend(gpu_warnings)
    status["gpu"] = gpu

    if status["running"] and log_info.get("age_seconds") is not None and log_info["age_seconds"] > 600:
        warnings.append(f"train log appears stale while trainer is running ({log_info['age_human']} old)")
    if log_info.get("exists") and progress is None:
        warnings.append("no parseable progress line found in train log")
    latest_step = latest_full.get("step") if latest_full else None
    delta_step = status["delta_checkpoint"].get("step") if status["delta_checkpoint"] else None
    if latest_step is not None and delta_step is not None and latest_step < delta_step:
        warnings.append(f"latest.json step {latest_step} lags newest delta step {delta_step}")
    if not status["running"] and progress is None:
        warnings.append("no active trainer process found")

    return status, 0


def _format_status_text(status: Dict[str, Any]) -> str:
    lines = [f"AGILLM status @ {status.get('checked_at_iso')}"]
    if status.get("error"):
        lines.append(f"Error: {status['error']}")
        for proc in status.get("processes", []):
            lines.append(f"- pid {proc.get('pid')}: {proc.get('cmdline')}")
        return "\n".join(lines)

    process = status.get("process")
    if status.get("running") and process:
        lines.append(f"Process: RUNNING | pid {process.get('pid')} | uptime {process.get('uptime_human') or 'unknown'}")
        lines.append(f"Cmd: {process.get('cmdline')}")
    else:
        lines.append("Process: NOT RUNNING")

    progress = status.get("progress")
    if progress:
        eta = progress.get("eta")
        if not eta and progress.get("tok_per_sec"):
            remaining = max(0, progress["target_tokens"] - progress["seen_tokens"])
            eta = _status_compact_duration(remaining / float(progress["tok_per_sec"]))
        lines.append(
            "Progress: "
            f"{progress['percent']:.1f}% | "
            f"{_status_format_int(progress['seen_tokens'])}/{_status_format_int(progress['target_tokens'])} tok | "
            f"{progress['tok_per_sec']} tok/s | loss {progress['loss']:.3f} | "
            f"B={progress['batch']} L={progress['block']}"
            + (f" | step {progress['step']}" if progress.get("step") else "")
            + (f" | ETA {eta}" if eta else "")
        )
    else:
        lines.append("Progress: unavailable")

    log_info = status.get("log") or {}
    if log_info.get("exists"):
        lines.append(
            f"Log: {log_info.get('path')} | updated {log_info.get('age_human') or 'unknown'} ago | "
            f"mtime {log_info.get('mtime_iso')}"
        )
    else:
        lines.append(f"Log: missing ({log_info.get('path')})")

    delta = status.get("delta_checkpoint")
    if delta:
        line = f"Delta: {delta.get('name')} | step {delta.get('step')} | source {delta.get('source')}"
        if delta.get("path"):
            line += f" | {delta['path']}"
        lines.append(line)
    else:
        lines.append("Delta: unavailable")

    latest_full = status.get("latest_full_checkpoint") or {}
    if latest_full.get("exists"):
        lines.append(
            f"Latest full: step {latest_full.get('step')} | {latest_full.get('checkpoint_path') or latest_full.get('raw_path')}"
        )
    else:
        lines.append(f"Latest full: unavailable ({latest_full.get('metadata_path')})")

    gpu = status.get("gpu")
    if gpu:
        lines.append(
            f"GPU: {gpu.get('name')} | {gpu.get('utilization_gpu')}% | "
            f"{gpu.get('memory_used_mib')}/{gpu.get('memory_total_mib')} MiB | "
            f"{gpu.get('temperature_c')}C | {gpu.get('power_draw_w')} W"
        )

    warnings = status.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _emit_status(log_path: Path, save_dir: Path, as_json: bool) -> int:
    status, exit_code = _collect_status(log_path, save_dir)
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(_format_status_text(status))
    return exit_code


def _run_status_command(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{STATUS_SCRIPT_PATH.name} status", description="Read-only training status")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--log", type=Path, default=STATUS_DEFAULT_LOG, help="Path to the training log")
    parser.add_argument("--save_dir", type=Path, default=STATUS_DEFAULT_SAVE_DIR, help="Checkpoint directory")
    args = parser.parse_args(argv)
    return _emit_status(args.log, args.save_dir, args.json_output)


def _maybe_handle_status_fastpath() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        raise SystemExit(_run_status_command(sys.argv[2:]))


_maybe_handle_status_fastpath()

import torch
import torch.utils.checkpoint as torch_checkpoint
from anchor_memory import AnchorMemoryConfig, AnchorMemoryLayer

# SafeProgress - Claude-safe progress (discrete lines, not single growing line)
class SafeProgress:
    def __init__(self, total, initial=0, unit="tok", print_every=100, print_every_sec=60):
        self.total, self.n, self.unit = total, initial, unit
        self.initial = initial
        self.last_print, self.postfix = initial, {}
        self.print_every = max(1, int(print_every))
        self.print_every_sec = max(1, int(print_every_sec))
        self.step = 0
        self.last_print_step = 0
        self.start_time = __import__('time').time()
        self.last_print_time = self.start_time
    def update(self, n=1):
        self.n += n
        self.step += 1
        now = __import__('time').time()
        if (
            self.step == 1
            or (self.step - self.last_print_step) >= self.print_every
            or (now - self.last_print_time) >= self.print_every_sec
        ):
            self._print(now)
            self.last_print = self.n
            self.last_print_step = self.step
            self.last_print_time = now
    def set_postfix(self, **kwargs): self.postfix = kwargs
    def _print(self, now=None):
        now = now or __import__('time').time()
        elapsed = now - self.start_time
        rate = (self.n - self.initial) / elapsed if elapsed > 0 else 0
        pct = 100 * self.n / self.total if self.total > 0 else 0
        pf = ' '.join(f"{k}={v}" for k,v in self.postfix.items())
        remaining = max(0, self.total - self.n)
        eta = _status_compact_duration(remaining / rate) if rate > 0 else "unknown"
        elapsed_s = _status_compact_duration(elapsed)
        print(
            f"[{pct:.4f}%] {self.n:,}/{self.total:,} {self.unit} | "
            f"{rate:.2f} tok/s | {pf} step={self.step} eta={eta} elapsed={elapsed_s}",
            flush=True,
        )
    def close(self): self._print(); print("Done.", flush=True)

import torch.nn as nn
import torch.nn.functional as F
from dblocks_train import _dblock_init, _dblock_step
import signal
import os
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, logging as hf_log
# from tqdm.auto import tqdm  # DISABLED - kills Claude context

# ─────────────────────────────── HOT DATASET LOADING ───────────────────────────────
HOT_CONFIG_PATH = Path("/workspace/hot_config.json")
_hot_config_cache = {"mtime": 0, "data": {}}

def get_hot_config() -> dict:
    """Load hot_config.json with caching, return empty dict if missing"""
    try:
        if HOT_CONFIG_PATH.exists():
            mtime = HOT_CONFIG_PATH.stat().st_mtime
            if mtime > _hot_config_cache["mtime"]:
                with open(HOT_CONFIG_PATH) as f:
                    _hot_config_cache["data"] = json.load(f)
                _hot_config_cache["mtime"] = mtime
        return _hot_config_cache["data"]
    except Exception as e:
        print(f"[hot_config] Error loading: {e}")
        return {}

def get_hot_datasets(default_sources: str) -> str:
    """Get datasets from hot_config if present, else use default"""
    cfg = get_hot_config()
    if "datasets" in cfg and cfg["datasets"]:
        hot_ds = cfg["datasets"]
        if isinstance(hot_ds, list):
            hot_ds = ",".join(hot_ds)
        print(f"[hot_config] Using hot datasets: {hot_ds}")
        return hot_ds
    return default_sources


# DISABLED: # Auto-rotating log to prevent context-window suicide
# DISABLED: try:
# DISABLED:     from rotating_log import install_rotating_log
# DISABLED:     install_rotating_log()
# DISABLED: except ImportError:
# pass  # Running without rotation

# ───────────────────────── ASCII Sanitizer ─────────────────────────
def _ascii_safe(s):
    if not isinstance(s, str):
        return s
    return (s
            .replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u201C', '"').replace('\u201D', '"')
            .replace('\u2014', '-').replace('\u2013', '-')
            .replace('\u2026', '...')
            .replace('\u00A0', ' '))

# ───────────────────────── ANSI Colors ─────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    PROMPT = "\033[36m"
    GEN = "\033[0m"
    INFO = "\033[90m"
    WARN = "\033[93m"

# ───────────────────────── Globals ─────────────────────────
hf_log.set_verbosity_error()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
SYNTHETIC_TOKENIZER = os.environ.get("AGILLM_SYNTHETIC_TOKENIZER", "").lower() in {"1", "true", "yes"}

class _SyntheticTokenizer:
    pad_token = "<|pad|>"
    pad_token_id = 0
    eos_token_id = 1
    sep_token_id = 1

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.backend_tokenizer = self

    def add_special_tokens(self, _tokens):
        return 0

    def get_vocab(self):
        return {f"tok_{i}": i for i in range(self.vocab_size)}

    def encode(self, text):
        return [2 + (ord(ch) % max(1, self.vocab_size - 2)) for ch in str(text)]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids if not skip_special_tokens or int(i) > 1)

    def to_str(self):
        return json.dumps({"type": "synthetic", "vocab_size": self.vocab_size})

if SYNTHETIC_TOKENIZER:
    tok = _SyntheticTokenizer(int(os.environ.get("AGILLM_SYNTHETIC_VOCAB", "8192")))
    print(f"[tokenizer] synthetic tokenizer enabled vocab={tok.vocab_size}")
else:
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})

# ─── Fix tokenizer Ġ/▁ mismatch ───
# The DeepSeek-V3.2 vocab uses Ġ (U+0120) for space-prefixed tokens,
# but some transformers versions set the Metaspace pre-tokenizer to use
# ▁ (U+2581) instead, causing encode/decode to lose all spaces.
def _fix_tokenizer_space_mismatch(tokenizer):
    try:
        import json as _json
        from tokenizers import Tokenizer as _Tokenizer
        bt = tokenizer.backend_tokenizer
        tj = _json.loads(bt.to_str())
        pre = tj.get("pre_tokenizer", {})
        needs_fix = (pre.get("type") == "Metaspace" and pre.get("replacement") == "\u2581")
        if not needs_fix:
            return
        # Check if vocab actually uses Ġ (U+0120) for spaces
        vocab = tj.get("model", {}).get("vocab", {})
        has_gpt2_space = any(k.startswith("\u0120") for k in list(vocab.keys())[:500])
        if not has_gpt2_space:
            return
        # Patch pre_tokenizer: ▁ -> Ġ
        tj["pre_tokenizer"]["replacement"] = "\u0120"
        # Patch decoder: ▁ -> Ġ in Replace step
        for step in tj.get("decoder", {}).get("decoders", []):
            if step.get("type") == "Replace":
                pat = step.get("pattern", {})
                if pat.get("String") == "\u2581":
                    pat["String"] = "\u0120"
        # Rebuild backend tokenizer
        fixed = _Tokenizer.from_str(_json.dumps(tj))
        tokenizer.backend_tokenizer = fixed
        # Verify fix
        test_ids = tokenizer.encode("hello world")
        test_dec = tokenizer.decode(test_ids, skip_special_tokens=True)
        if "hello world" in test_dec:
            print("[tokenizer] Fixed Ġ/▁ space mismatch")
        else:
            print(f"[tokenizer] WARNING: fix applied but decode test failed: {repr(test_dec)}")
    except Exception as e:
        print(f"[tokenizer] Could not fix space mismatch: {e}")

if not SYNTHETIC_TOKENIZER:
    _fix_tokenizer_space_mismatch(tok)

# ─── Tokenizer startup health check ───
# Abort early if tokenizer can't roundtrip spaces — prevents silent data corruption
def _tokenizer_health_check(tokenizer):
    import transformers as _tf
    ver = _tf.__version__
    print(f"[tokenizer] transformers={ver}, tokenizers={__import__('tokenizers').__version__}")
    # Warn on known-bad versions
    try:
        from packaging.version import Version
        if Version(ver) >= Version('5.0.0'):
            print(f'[tokenizer] WARNING: transformers {ver} may have Metaspace bug — verify carefully')
    except ImportError:
        pass
    # Roundtrip tests — must preserve spaces
    tests = [
        'Water boils at one hundred degrees',
        'The quick brown fox jumps over the lazy dog',
        'Hello world! This is a test sentence with spaces.',
    ]
    for text in tests:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if ' ' not in decoded:
            print(f'[tokenizer] FATAL: Roundtrip lost all spaces!')
            print(f'  Input:   {repr(text)}')
            print(f'  Encoded: {ids[:20]}...')
            print(f'  Decoded: {repr(decoded)}')
            print(f'[tokenizer] ABORTING — fix tokenizer before training!')
            sys.exit(1)
        # Check decoded is reasonably close to input
        if text.lower().split()[:3] != decoded.lower().split()[:3]:
            print(f'[tokenizer] WARNING: Roundtrip diverged:')
            print(f'  Input:   {repr(text[:60])}')
            print(f'  Decoded: {repr(decoded[:60])}')
    print(f'[tokenizer] Health check PASSED — spaces preserved in roundtrip')

if not SYNTHETIC_TOKENIZER:
    _tokenizer_health_check(tok)

VOCAB, BLANK, EOS = (
    max(tok.get_vocab().values()) + 1,
    int(getattr(tok, "pad_token_id", 0) or 0),
    tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
)

# ───────────────────────── PRESETS ─────────────────────────
PRESETS: Dict[str, Dict[str, int]] = {
    "femto_1x":  dict(d=16, layers=1, heads=1, rank=16),
    "femto_12x": dict(d=16, layers=1, heads=1, rank=192),
    "femto_24x": dict(d=16, layers=1, heads=1, rank=384),
    "pico_1x":   dict(d=32, layers=1, heads=2, rank=16),
    "pico_3x":   dict(d=32, layers=1, heads=2, rank=48),
    "pico_6x":   dict(d=32, layers=1, heads=2, rank=96),
    "pico_12x":  dict(d=32, layers=1, heads=2, rank=192),
    "pico_24x":  dict(d=32, layers=1, heads=2, rank=384),
    "pico_48x":  dict(d=32, layers=1, heads=2, rank=768),
    "nano_1x":   dict(d=64,  layers=2, heads=4, rank=16),
    "nano_3x":   dict(d=64,  layers=2, heads=4, rank=48),
    "nano_6x":   dict(d=64,  layers=2, heads=4, rank=96),
    "nano_12x":  dict(d=64,  layers=2, heads=4, rank=192),
    "nano_24x":  dict(d=64,  layers=2, heads=4, rank=384),
    "nano_48x":  dict(d=64,  layers=2, heads=4, rank=768),
    "nano_96x":  dict(d=64,  layers=2, heads=4, rank=1536),
    "micro_3x":  dict(d=128, layers=4, heads=8, rank=48),
    "micro_6x":  dict(d=128, layers=4, heads=8, rank=96),
    "micro_12x": dict(d=128, layers=4, heads=8, rank=192),
    "micro_24x": dict(d=128, layers=4, heads=8, rank=384),
    "small":     dict(d=512, layers=8,  heads=16, rank=64),
    "smallx2":   dict(d=512, layers=16, heads=16, rank=64),
    "base":      dict(d=768, layers=12, heads=24, rank=96),
    "base18":    dict(d=768, layers=18, heads=24, rank=96),
    "large":     dict(d=1024, layers=24, heads=16, rank=128),
    # AGILLM-4 tiers. These are intentionally above the ~700M AGILLM-3 size.
    # Approx dense parameter count with the current untied embedding+AR+SAT+NAT heads:
    # agillm4_floor ~= 1.21B, agillm4_main ~= 1.70B, agillm4_big ~= 2.40B.
    "agillm4_floor": dict(d=1280, layers=28, heads=20, rank=160),
    "agillm4_main":  dict(d=1536, layers=32, heads=24, rank=192),
    "agillm4_big":   dict(d=1792, layers=36, heads=28, rank=224),
}

DEFAULT_BLOCK = 1122
DEFAULT_BATCH = 4
SAT_BLOCK = 2
LR_CORE, LR_HEAD = 5e-5, 2e-4
EMIT_LAMBDA = 0.1
DEFAULT_SAVE_SEC = 24 * 3600
DEFAULT_DELTA_STEPS = 100000     # lightweight weight-only save every N steps
DEFAULT_MAX_DELTAS = 5         # keep last N deltas (older pruned after full save)
CKDIR = pathlib.Path("ckpts_expansion")

DEFAULT_PRETRAIN_SOURCES = "LLM360/TxT360,OpenTransformer/goddess-crawl,OpenTransformer/agillm-crawl-data,OpenTransformer/web-crawl-2026,OpenTransformer/web-crawl-clean-v2,OpenTransformer/scraped-web-data,OpenTransformer/turbo-crawl,OpenTransformer/sft-data-clean,OpenTransformer/web-crawl-v1,HuggingFaceFW/fineweb,wikimedia/wikipedia:20231101.en,allenai/c4:en,EleutherAI/proof-pile-2"
DEFAULT_AFTER_SFT_SOURCES = "mlabonne/opc-sft-stage2-chat,HuggingFaceH4/ultrachat_200k@train_sft"
DEFAULT_AFTER_SFT_BLOCK = 768
DEFAULT_ATTN_BACKEND = os.environ.get("AGILLM_ATTN_BACKEND", "manual")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

DEFAULT_SUBLINEAR_WINDOW = _env_int("AGILLM_SUBLINEAR_WINDOW", 256)
DEFAULT_SUBLINEAR_STRIDE = _env_int("AGILLM_SUBLINEAR_STRIDE", 64)
DEFAULT_SUBLINEAR_MAX_ANCHORS = _env_int("AGILLM_SUBLINEAR_MAX_ANCHORS", 256)
DEFAULT_SUBLINEAR_CHUNK = _env_int("AGILLM_SUBLINEAR_CHUNK", 128)
DEFAULT_SUBLINEAR_SINKS = _env_int("AGILLM_SUBLINEAR_SINKS", 4)
DEFAULT_SUBLINEAR_RECENT_ANCHORS = _env_int("AGILLM_SUBLINEAR_RECENT_ANCHORS", -1)  # -1 = half of max anchors
DEFAULT_SUBLINEAR_POOLED_LANDMARKS = bool(_env_int("AGILLM_SUBLINEAR_POOLED_LANDMARKS", 0))
DEFAULT_ANCHOR_MEMORY = bool(_env_int("AGILLM_ANCHOR_MEMORY", 0))
DEFAULT_ANCHOR_STRIDE = _env_int("AGILLM_ANCHOR_STRIDE", 256)
DEFAULT_ANCHOR_MAX = _env_int("AGILLM_ANCHOR_MAX", 2048)
DEFAULT_ANCHOR_POSITION = _env_int("AGILLM_ANCHOR_POSITION", -1)  # -1 = stack middle
DEFAULT_KV_BUFFER = bool(_env_int("AGILLM_KV_BUFFER", 0))
AGILLM4_TOKEN_PARAM_RATIO = 100.0

# ───────────────────────── UK Time Helper ─────────────────────────
def get_uk_time() -> str:
    utc_now = datetime.now(timezone.utc)
    year = utc_now.year
    march_last = datetime(year, 3, 31, 1, 0, tzinfo=timezone.utc)
    while march_last.weekday() != 6:
        march_last = march_last.replace(day=march_last.day - 1)
    oct_last = datetime(year, 10, 31, 1, 0, tzinfo=timezone.utc)
    while oct_last.weekday() != 6:
        oct_last = oct_last.replace(day=oct_last.day - 1)
    if march_last <= utc_now < oct_last:
        uk_offset = 1
        tz_name = "BST"
    else:
        uk_offset = 0
        tz_name = "GMT"
    from datetime import timedelta
    uk_time = utc_now + timedelta(hours=uk_offset)
    return uk_time.strftime(f'%Y-%m-%d %H:%M:%S {tz_name}')

# ───────────────────────── Utilities ─────────────────────────
def rng_state():
    if DEV.type == "cuda":
        try:
            return torch.cuda.get_rng_state(DEV)
        except TypeError:
            return torch.cuda.get_rng_state()
    return torch.get_rng_state()

def _is_probably_ckpt(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.suffix == ".pt" and not path.name.endswith(".pt.tmp") and path.stat().st_size > (1<<20)
    except Exception:
        return False

def _resolve_ckpt(path: pathlib.Path) -> pathlib.Path | None:
    try:
        if path.is_dir():
            cands = sorted([p for p in path.glob("*.pt") if _is_probably_ckpt(p)],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        if path.suffix == ".tmp":
            solid = path.with_suffix("")
            return solid if _is_probably_ckpt(solid) else _resolve_ckpt(path.parent)
        return path if _is_probably_ckpt(path) else _resolve_ckpt(path.parent)
    except Exception:
        return None

def _try_load(path: pathlib.Path, map_location="cpu"):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None

def _prune_checkpoints(save_dir: pathlib.Path, phase_name: str, max_ckpts: int):
    if max_ckpts is None or max_ckpts <= 0:
        return
    try:
        pattern = f"{phase_name}_step*.pt"
        ckpts = sorted(
            [p for p in save_dir.glob(pattern) if _is_probably_ckpt(p)],
            key=lambda p: p.stat().st_mtime
        )
        excess = len(ckpts) - max_ckpts
        if excess > 0:
            for p in ckpts[:excess]:
                try:
                    p.unlink()
                    print(f"  [prune] deleted old {p.name}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[ckpt-prune] error: {e}")

def print_expansion_info(cfg: dict, tie_weights: bool = False, plain: bool = False):
    d_k = cfg["d"] // cfg["heads"]
    rank = cfg["rank"]
    ratio = rank / d_k
    regime = "COMPRESSION" if ratio < 1 else ("IDENTITY" if ratio == 1 else "EXPANSION")
    tie_str = "YES" if tie_weights else "NO"
    if plain:
        print("[attention_config]")
        print(f"d_model={cfg['d']} heads={cfg['heads']} d_k={d_k}")
        print(f"layers={cfg['layers']} tie_weights={tie_str}")
        print(f"rank={rank} ratio={ratio:.1f}x regime={regime}")
        return
    print(f"┌─────────────────────────────────────────┐")
    print(f"│ TUNEABLE ATTENTION CONFIG               │")
    print(f"├─────────────────────────────────────────┤")
    print(f"│ d_model: {cfg['d']:4d}  heads: {cfg['heads']:2d}  d_k: {d_k:3d}     │")
    print(f"│ layers: {cfg['layers']:4d}  tie_weights: {tie_str:3s}          │")
    print(f"│ rank: {rank:4d}  ratio: {ratio:.1f}x  [{regime:11s}] │")
    print(f"└─────────────────────────────────────────┘")

# ───────────────────────── AMP helper ─────────────────────────
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _auto_amp_dtype():
    if DEV.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported(): return torch.bfloat16
            return torch.float16
        except Exception: return torch.float16
    return torch.float32

def amp(enabled: bool):
    return nullcontext() if not (enabled and DEV.type == "cuda") else _ac(device_type="cuda", dtype=_auto_amp_dtype())

def _needs_grad_scaler() -> bool:
    return bool(DEV.type == "cuda" and _auto_amp_dtype() == torch.float16)

# ───────────────────────── Chat & Data Stream ─────────────────────────
def _coerce_role(r: str) -> str:
    r = (r or "").lower()
    if r in {"user", "human", "customer"}: return "user"
    if r in {"assistant", "gpt", "bot"}: return "assistant"
    if r in {"system", "context"}: return "system"
    return r or "user"

def _chat_content(m: dict) -> str:
    content = m.get("content", m.get("text", m.get("value", "")))
    return content if isinstance(content, str) else ""

def _chat_role(m: dict) -> str:
    return _coerce_role(m.get("role", m.get("from", m.get("speaker", ""))))

def _fallback_chat_template(messages: list[dict], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = _chat_role(m)
        content = _chat_content(m).strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    if add_generation_prompt and (not parts or not parts[-1].startswith("Assistant:")):
        parts.append("Assistant:")
    return "\n".join(parts)

def _render_chat_text_from_ex(ex: dict, messages_key: str, add_generation_prompt: bool) -> Optional[str]:
    msgs = ex.get(messages_key)
    if msgs is None:
        for alt in ("conversations", "dialog", "turns"):
            if isinstance(ex.get(alt), list):
                msgs = ex[alt]; break
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        norm = []
        for m in msgs:
            content = _chat_content(m)
            if not isinstance(content, str) or not content:
                continue
            norm.append({"role": _chat_role(m), "content": content})
        if not norm: return None
        try:
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return _fallback_chat_template(norm, add_generation_prompt)
    for a, b in (("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if isinstance(ex.get(a), str) and isinstance(ex.get(b), str):
            return f"User: {ex[a]}\nAssistant: {ex[b]}"
    return None

def _parse_dataset_ref(ds_name: str):
    split = "train"
    ref = ds_name
    if "@" in ref:
        ref, split = ref.rsplit("@", 1)
        split = split or "train"
    if ":" in ref:
        base, config = ref.split(":", 1)
    else:
        base, config = ref, None
    return base, config, split

def _open_stream_one(ds_name: str, seed: int, streaming: bool = True):
    dc = DownloadConfig(max_retries=5, use_etag=True, resume_download=True)
    base, config, split = _parse_dataset_ref(ds_name)
    if not streaming:
        print(f"[download] Downloading {ds_name} (non-streaming)...")
    if base == "json":
        data_files = {"train": config}
        ds = load_dataset("json", data_files=data_files, split=split, streaming=streaming, download_config=dc)
    else:
        ds = load_dataset(base, config, split=split, streaming=streaming, download_config=dc) if config else \
             load_dataset(base, split=split, streaming=streaming, download_config=dc)
    if streaming:
        return iter(ds.shuffle(buffer_size=1000, seed=seed))
    else:
        print(f"[download] Got {len(ds):,} examples. Shuffling...")
        ds = ds.shuffle(seed=seed)
        return iter(ds)

def token_stream(ds_names: str, target: int, seed: int = 42,
                 chat: bool = False, chat_messages_key: str = "messages",
                 sft_add_generation_prompt: bool = False, dataset_field_text: str = "text",
                 streaming: bool = True):
    ds_names = get_hot_datasets(ds_names)  # HOT LOAD
    sources = [s.strip() for s in ds_names.split(",") if s.strip()]
    if not sources: return
    src_idx = 0; emitted = 0; it = None; attempts = 0; backoff_base = 2.0
    while emitted < target:
        try:
            if it is None: it = _open_stream_one(sources[src_idx], seed, streaming=streaming)
            ex = next(it)
            text = None
            if isinstance(ex, dict):
                if chat:
                    text = _render_chat_text_from_ex(ex, chat_messages_key, sft_add_generation_prompt)
                if text is None:
                    if dataset_field_text and isinstance(ex.get(dataset_field_text), str):
                        text = ex[dataset_field_text]
                    elif isinstance(ex.get("text"), str):
                        text = ex["text"]
            if not isinstance(text, str):
                attempts = 0; continue
            enc = tok.encode(text)
            if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
                enc = enc + [EOS]
            for t in enc:
                yield t
                emitted += 1
                if emitted >= target: return
            attempts = 0
        except StopIteration:
            it = None; src_idx = (src_idx + 1) % len(sources)
        except Exception as e:
            attempts += 1
            sleep_s = min(60.0, backoff_base ** min(attempts, 6))
            print(f"[stream-retry] {sources[src_idx]} error: {type(e).__name__}, sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s); it = None
            if attempts % 5 == 0 and len(sources) > 1:
                src_idx = (src_idx + 1) % len(sources)

# ───────────────────────── ALiBi ─────────────────────────
def _alibi_slopes(n_heads: int):
    def pow2slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer(): vals = pow2slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0) 
    return -_alibi_slopes(n_heads) * dist


class StructuredAttentionMask:
    """Symbolic attention rules for sublinear attention.

    Dense masks are O(T^2). This object carries the rule so sublinear attention can
    apply it only to the gathered local/anchor candidate keys: O(T * candidates).
    """

    __slots__ = ("kind", "q_len", "k_len", "query_base", "block")

    def __init__(self, kind: str, q_len: int, k_len: int = None, query_base: int = 0, block: int = 1):
        self.kind = (kind or "none").lower()
        self.q_len = int(q_len)
        self.k_len = int(k_len if k_len is not None else q_len)
        self.query_base = int(query_base)
        self.block = max(1, int(block))

    def to_dense(self, device=None, dtype=torch.float32):
        device = device or DEV
        if self.kind in {"none", "nat", "bidirectional", "unrestricted"}:
            return None
        q_pos = torch.arange(self.query_base, self.query_base + self.q_len, device=device, dtype=torch.long).view(self.q_len, 1)
        k_pos = torch.arange(self.k_len, device=device, dtype=torch.long).view(1, self.k_len)
        if self.kind == "causal":
            allow = k_pos <= q_pos
        elif self.kind in {"sat", "block_causal", "block-causal"}:
            allow = (k_pos // self.block) <= (q_pos // self.block)
        else:
            raise ValueError(f"unknown structured attention mask kind: {self.kind}")
        zeros = torch.zeros((self.q_len, self.k_len), device=device, dtype=dtype)
        neg = torch.full_like(zeros, float("-inf"))
        return torch.where(allow, zeros, neg).unsqueeze(0).unsqueeze(0)


def _is_structured_attention_mask(mask) -> bool:
    return isinstance(mask, StructuredAttentionMask)


def use_structured_masks(args=None, backend: str = None) -> bool:
    backend = (backend or getattr(args, "attn_backend", "") or "").lower()
    return backend == "sublinear" and not bool(getattr(args, "no_structured_masks", False))

# ───────────────────────── Model components ─────────────────────────
class KVBuffer:
    """Preallocated K/V cache for decode. Replaces torch.cat-based growth.

    Layout matches MHA-internal head-major shape [B, H, T, d_k]. Caller sizes
    once; each ``append`` writes ``length:length+n`` slots in place and grows
    ``length``. ``view()`` returns slices of the live region so attention sees
    only filled positions.
    """

    __slots__ = ("k", "v", "length", "capacity")

    def __init__(
        self,
        batch: int,
        heads: int,
        capacity: int,
        d_k: int,
        device,
        dtype,
    ):
        self.k = torch.empty(batch, heads, capacity, d_k, device=device, dtype=dtype)
        self.v = torch.empty(batch, heads, capacity, d_k, device=device, dtype=dtype)
        self.length = 0
        self.capacity = capacity

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        n = k_new.size(2)
        end = self.length + n
        if end > self.capacity:
            raise RuntimeError(
                f"KVBuffer overflow: length={self.length} + n={n} > capacity={self.capacity}"
            )
        self.k[:, :, self.length:end].copy_(k_new)
        self.v[:, :, self.length:end].copy_(v_new)
        self.length = end

    def view(self):
        return self.k[:, :, :self.length], self.v[:, :, :self.length]


class TuneableAttentionMHA(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        r: int,
        use_relpos: bool = True,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
    ):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.r = h, d // h, r
        self.use_relpos = use_relpos
        self.attn_backend = (attn_backend or "manual").lower()
        self.sublinear_window = max(1, int(sublinear_window))
        self.sublinear_stride = max(0, int(sublinear_stride))
        self.sublinear_max_anchors = max(0, int(sublinear_max_anchors))
        self.sublinear_chunk = max(1, int(sublinear_chunk))
        self.sublinear_sinks = max(0, int(sublinear_sinks))
        recent = int(sublinear_recent_anchors)
        if recent < 0:
            recent = self.sublinear_max_anchors // 2
        self.sublinear_recent_anchors = min(max(0, recent), self.sublinear_max_anchors)
        self.sublinear_pooled_landmarks = bool(sublinear_pooled_landmarks)
        # Exact n1 harvest: one fused QKV projection is mathematically the same
        # as three independent bias-free Linear(d, d) projections with their
        # weights stacked along out_features.
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.U = nn.Parameter(torch.randn(self.dk, r))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(h * self.dk, d, bias=False)
        self.drop = nn.Dropout(0.1)
        # Exact n1 harvest: for expansion ranks, (q @ U) @ (k @ U).T is
        # q @ (U @ U.T) @ k.T. This keeps score/cache width at d_k with no
        # quality change. Inference caches the metric and training recomputes
        # it so gradients through U are unchanged.
        self._metric_cache: Optional[torch.Tensor] = None
        self._metric_cache_ver: int = -1
        self._metric_cache_param_id: int = -1
        self._metric_cache_data_ptr: int = -1
        self._metric_cache_shape: Tuple[int, int] = (-1, -1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        qkv_key = prefix + "qkv.weight"
        if qkv_key not in state_dict:
            qk = prefix + "q.weight"
            kk = prefix + "k.weight"
            vk = prefix + "v.weight"
            if qk in state_dict and kk in state_dict and vk in state_dict:
                fused = _cat_legacy_weight_blocks([state_dict[qk], state_dict[kk], state_dict[vk]])
                if fused is not None:
                    state_dict[qkv_key] = fused
                    state_dict.pop(qk)
                    state_dict.pop(kk)
                    state_dict.pop(vk)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _proj_qk(self, x):
        B, N, _ = x.shape
        return (x.view(B, N, self.h, self.dk).transpose(1, 2) @ self.U)
    
    def _reshape_v(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)

    def _reshape_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)

    def _get_metric(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.U @ self.U.T
        cur_ver = self.U._version
        cur_param_id = id(self.U)
        cur_data_ptr = int(self.U.data_ptr())
        cur_shape = tuple(self.U.shape)
        cache = self._metric_cache
        if (
            cache is None
            or cache.dtype != self.U.dtype
            or cache.device != self.U.device
            or self._metric_cache_ver != cur_ver
            or self._metric_cache_param_id != cur_param_id
            or self._metric_cache_data_ptr != cur_data_ptr
            or self._metric_cache_shape != cur_shape
        ):
            cache = (self.U @ self.U.T).detach()
            self._metric_cache = cache
            self._metric_cache_ver = cur_ver
            self._metric_cache_param_id = cur_param_id
            self._metric_cache_data_ptr = cur_data_ptr
            self._metric_cache_shape = cur_shape
        return cache

    def train(self, mode: bool = True):
        if mode:
            self._metric_cache = None
            self._metric_cache_ver = -1
            self._metric_cache_param_id = -1
            self._metric_cache_data_ptr = -1
            self._metric_cache_shape = (-1, -1)
        return super().train(mode)

    def _structured_valid(self, attn_mask, q_pos, idx):
        if not _is_structured_attention_mask(attn_mask):
            return None
        kind = attn_mask.kind
        if kind in {"none", "nat", "bidirectional", "unrestricted"}:
            return torch.ones_like(idx, dtype=torch.bool)
        if kind == "causal":
            return idx <= q_pos[:, None]
        if kind in {"sat", "block_causal", "block-causal"}:
            block = max(1, int(attn_mask.block))
            return (idx // block) <= (q_pos[:, None] // block)
        raise ValueError(f"unknown structured attention mask kind: {kind}")

    def _sublinear_anchor_positions(self, k_len: int, device):
        anchor_start = self.sublinear_stride - 1
        if self.sublinear_stride <= 0 or self.sublinear_max_anchors <= 0 or anchor_start >= k_len:
            anchors = torch.empty(0, device=device, dtype=torch.long)
        else:
            all_anchors = torch.arange(anchor_start, k_len, self.sublinear_stride, device=device, dtype=torch.long)
            if all_anchors.numel() <= self.sublinear_max_anchors:
                anchors = all_anchors
            else:
                recent_budget = min(self.sublinear_recent_anchors, self.sublinear_max_anchors)
                span_budget = max(0, self.sublinear_max_anchors - recent_budget)
                parts = []
                if span_budget > 0:
                    span_sel = torch.linspace(0, all_anchors.numel() - 1, span_budget, device=device).round().long().unique()
                    parts.append(all_anchors[span_sel])
                if recent_budget > 0:
                    parts.append(all_anchors[-recent_budget:])
                anchors = torch.cat(parts).unique() if parts else torch.empty(0, device=device, dtype=torch.long)
        if self.sublinear_sinks > 0 and k_len > 0:
            sinks = torch.arange(min(self.sublinear_sinks, k_len), device=device, dtype=torch.long)
            anchors = torch.cat([sinks, anchors]).unique() if anchors.numel() else sinks
        return anchors

    def _sublinear_attention(self, q, k, v, attn_mask=None, rel_bias_tokens=None):
        """Local-window + landmark attention: O(N * (window + N/stride))."""
        bsz, heads, q_len, _ = q.shape
        k_len = k.size(2)
        device = q.device
        query_base = max(0, k_len - q_len)
        outputs = []
        scale = 1.0 / math.sqrt(self.dk)
        slopes = None
        if self.use_relpos and rel_bias_tokens is not None:
            slopes = _alibi_slopes(self.h).to(device=device, dtype=torch.float32)

        anchors = self._sublinear_anchor_positions(k_len, device)
        anchor_k = anchor_v = None
        if anchors.numel() and self.sublinear_pooled_landmarks and self.sublinear_stride > 1:
            # Optional pooled landmarks: each global anchor summarizes its stride segment.
            # This is off by default because it adds cumsum work; enable after benchmarking.
            ends = anchors + 1
            starts = (ends - self.sublinear_stride).clamp_min(0)
            zero_k = k.new_zeros(k.size(0), k.size(1), 1, k.size(3))
            zero_v = v.new_zeros(v.size(0), v.size(1), 1, v.size(3))
            prefix_k = torch.cat([zero_k, k.cumsum(dim=2)], dim=2)
            prefix_v = torch.cat([zero_v, v.cumsum(dim=2)], dim=2)
            denom = (ends - starts).to(dtype=k.dtype).view(1, 1, -1, 1).clamp_min(1)
            anchor_k = (prefix_k[:, :, ends, :] - prefix_k[:, :, starts, :]) / denom
            anchor_v = (prefix_v[:, :, ends, :] - prefix_v[:, :, starts, :]) / denom

        offsets = torch.arange(
            -self.sublinear_window,
            self.sublinear_window + 1,
            device=device,
            dtype=torch.long,
        )

        for q_start in range(0, q_len, self.sublinear_chunk):
            q_end = min(q_len, q_start + self.sublinear_chunk)
            cur = q_end - q_start
            q_pos = torch.arange(query_base + q_start, query_base + q_end, device=device, dtype=torch.long)

            local_raw = q_pos[:, None] + offsets[None, :]
            local_valid = (local_raw >= 0) & (local_raw < k_len)
            local_idx = local_raw.clamp(0, max(0, k_len - 1))

            k_local = k[:, :, local_idx, :]
            v_local = v[:, :, local_idx, :]
            if anchors.numel():
                anchor_idx = anchors.view(1, -1).expand(cur, -1)
                local_lo = (q_pos - self.sublinear_window).clamp_min(0).view(-1, 1)
                local_hi = (q_pos + self.sublinear_window).clamp_max(max(0, k_len - 1)).view(-1, 1)
                # Drop anchor copies already present in the local window; duplicates bias softmax mass.
                anchor_valid = (anchor_idx < local_lo) | (anchor_idx > local_hi)
                idx = torch.cat([local_idx, anchor_idx], dim=1)
                valid = torch.cat([local_valid, anchor_valid], dim=1)
                if anchor_k is not None and anchor_v is not None:
                    k_anchor = anchor_k.unsqueeze(2).expand(-1, -1, cur, -1, -1)
                    v_anchor = anchor_v.unsqueeze(2).expand(-1, -1, cur, -1, -1)
                else:
                    k_anchor = k[:, :, anchor_idx, :]
                    v_anchor = v[:, :, anchor_idx, :]
                k_sel = torch.cat([k_local, k_anchor], dim=-2)
                v_sel = torch.cat([v_local, v_anchor], dim=-2)
            else:
                idx = local_idx
                valid = local_valid
                k_sel = k_local
                v_sel = v_local

            structured_valid = self._structured_valid(attn_mask, q_pos, idx)
            if structured_valid is not None:
                valid = valid & structured_valid

            scores = (q[:, :, q_start:q_end, :].unsqueeze(-2) * k_sel).sum(dim=-1) * scale

            if slopes is not None:
                dist = (q_pos.view(1, 1, cur, 1) - idx.view(1, 1, cur, -1)).abs().to(torch.float32)
                scores = scores + (-slopes * dist).to(scores.dtype)

            if torch.is_tensor(attn_mask) and attn_mask.size(-1) == k_len and attn_mask.size(-2) >= q_end:
                mask_q = attn_mask[..., q_start:q_end, :]
                gather_idx = idx.view(1, 1, cur, -1).expand(mask_q.size(0), mask_q.size(1), cur, idx.size(1))
                scores = scores + torch.gather(mask_q, -1, gather_idx)

            scores = scores.masked_fill(~valid.view(1, 1, cur, -1), float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            outputs.append((weights.unsqueeze(-1) * v_sel).sum(dim=-2))

        return torch.cat(outputs, dim=2)

    def forward(self, x, mask=None, rel_bias_tokens=None, kv_cache=None, use_cache=False):
        q_lin, k_lin, v_lin = self.qkv(x).chunk(3, dim=-1)
        v_new = self._reshape_v(v_lin)
        if self.r > self.dk:
            q = self._reshape_heads(q_lin) @ self._get_metric()
            k_new = self._reshape_heads(k_lin)
        else:
            q = self._proj_qk(q_lin)
            k_new = self._proj_qk(k_lin)
        if kv_cache is None:
            k, v = k_new, v_new
        elif isinstance(kv_cache, KVBuffer):
            if use_cache:
                kv_cache.append(k_new, v_new)
                k, v = kv_cache.view()
            else:
                k, v = k_new, v_new
        else:
            k_cached, v_cached = kv_cache
            if use_cache:
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k, v = k_new, v_new
        attn_mask = mask
        if self.attn_backend != "sublinear" and _is_structured_attention_mask(attn_mask):
            attn_mask = attn_mask.to_dense(device=q.device, dtype=q.dtype)
        if self.attn_backend != "sublinear" and self.use_relpos and rel_bias_tokens is not None:
            rel = alibi_bias(self.h, rel_bias_tokens)[:, :, -q.size(2):, :]
            attn_mask = rel if attn_mask is None else attn_mask + rel
        if self.attn_backend == "sdpa":
            try:
                z = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    scale=1.0 / math.sqrt(self.dk),
                )
            except TypeError:
                # Older torch lacks the scale kwarg. Rescale q so SDPA's default sqrt(r)
                # denominator matches the historical AGILLM sqrt(d_k) denominator.
                q_scaled = q * math.sqrt(q.size(-1) / self.dk)
                z = F.scaled_dot_product_attention(q_scaled, k, v, attn_mask=attn_mask, dropout_p=0.0)
        elif self.attn_backend == "sublinear":
            z = self._sublinear_attention(q, k, v, attn_mask=attn_mask, rel_bias_tokens=rel_bias_tokens)
        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if attn_mask is not None:
                att = att + attn_mask
            z = att.softmax(-1) @ v
        z = z.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        out = self.drop(self.proj(z))
        if not use_cache:
            return out
        new_kv = kv_cache if isinstance(kv_cache, KVBuffer) else (k, v)
        return out, new_kv


class Block(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        r: int,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
    ):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = TuneableAttentionMHA(
            d,
            h,
            r,
            attn_backend=attn_backend,
            sublinear_window=sublinear_window,
            sublinear_stride=sublinear_stride,
            sublinear_max_anchors=sublinear_max_anchors,
            sublinear_chunk=sublinear_chunk,
            sublinear_sinks=sublinear_sinks,
            sublinear_recent_anchors=sublinear_recent_anchors,
            sublinear_pooled_landmarks=sublinear_pooled_landmarks,
        )
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, kv=None, use_cache=False, total_seq_len=None):
        if use_cache:
            y, new_kv = self.mha(self.ln1(x), mask, rel_bias_tokens=total_seq_len, kv_cache=kv, use_cache=True)
            x = x + y + self.ff(self.ln2(x + y))
            return x, new_kv
        else:
            n = x.size(1)
            x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
            return x + self.ff(self.ln2(x))


class Encoder(nn.Module):
    def __init__(
        self,
        cfg,
        tie_weights: bool = False,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        grad_checkpoint: bool = False,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
        anchor_memory: bool = DEFAULT_ANCHOR_MEMORY,
        anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
        anchor_max: int = DEFAULT_ANCHOR_MAX,
        anchor_position: int = DEFAULT_ANCHOR_POSITION,
    ):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([
            Block(
                d,
                h,
                r,
                attn_backend=attn_backend,
                sublinear_window=sublinear_window,
                sublinear_stride=sublinear_stride,
                sublinear_max_anchors=sublinear_max_anchors,
                sublinear_chunk=sublinear_chunk,
                sublinear_sinks=sublinear_sinks,
                sublinear_recent_anchors=sublinear_recent_anchors,
                sublinear_pooled_landmarks=sublinear_pooled_landmarks,
            )
            for _ in range(l)
        ])
        self.ln = nn.LayerNorm(d)
        self.tie_weights = tie_weights
        self.attn_backend = attn_backend
        self.grad_checkpoint = grad_checkpoint
        self.sublinear_window = sublinear_window
        self.sublinear_stride = sublinear_stride
        self.sublinear_max_anchors = sublinear_max_anchors
        self.sublinear_chunk = sublinear_chunk
        self.sublinear_sinks = sublinear_sinks
        self.sublinear_recent_anchors = sublinear_recent_anchors
        self.sublinear_pooled_landmarks = bool(sublinear_pooled_landmarks)
        self.anchor_memory_enabled = bool(anchor_memory)
        self.anchor_stride = int(anchor_stride)
        self.anchor_max = int(anchor_max)
        n_layers = int(cfg["layers"])
        if int(anchor_position) < 0:
            self.anchor_position = n_layers // 2
        else:
            self.anchor_position = min(int(anchor_position), n_layers - 1)
        if self.anchor_memory_enabled:
            am_cfg = AnchorMemoryConfig(
                d_model=int(cfg["d"]),
                heads=int(cfg["heads"]),
                anchor_stride=self.anchor_stride,
                max_anchors=self.anchor_max,
            )
            self.anchor = AnchorMemoryLayer(am_cfg)
        else:
            self.anchor = None

    def forward(self, ids, mask, kv_caches=None, use_cache=False, total_seq_len=None):
        x = self.emb(ids)
        if not use_cache:
            for i, blk in enumerate(self.blocks):
                if self.grad_checkpoint and self.training:
                    x = torch_checkpoint.checkpoint(lambda y, block=blk: block(y, mask), x, use_reentrant=False)
                else:
                    x = blk(x, mask)
                if self.anchor is not None and i == self.anchor_position:
                    if self.grad_checkpoint and self.training:
                        x, _ = torch_checkpoint.checkpoint(self.anchor, x, use_reentrant=False)
                    else:
                        x, _ = self.anchor(x)
            return self.ln(x)
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, kv_out = blk(x, mask, kv, use_cache=True, total_seq_len=total_seq_len)
            new_kvs.append(kv_out)
            if self.anchor is not None and i == self.anchor_position:
                x, _ = self.anchor(x)
        return self.ln(x), new_kvs


class ARHead(nn.Module):
    def __init__(self, d, tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)
    
    def forward(self, h): 
        return self.proj(h)


class NATHead(nn.Module):
    def __init__(self, d, tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)

    def forward(self, h):
        return self.proj(h)


class SATHead(nn.Module):
    def __init__(self, d, mode="var", tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)
        self.gate = nn.Linear(d, 2) if mode == "var" else None
    def forward(self, h_last):
        return self.proj(h_last), (self.gate(h_last[:, 0]) if self.gate else None)


# ───────────────────────── Masks ─────────────────────────
def causal_mask(n, structured: bool = False):
    if structured:
        return StructuredAttentionMask("causal", q_len=n, k_len=n, query_base=0)
    return torch.triu(torch.full((1, 1, n, n), float("-inf"), device=DEV), 1)

def sat_mask(n, block=SAT_BLOCK, structured: bool = False):
    if structured:
        return StructuredAttentionMask("sat", q_len=n, k_len=n, query_base=0, block=block)
    idx = torch.arange(n, device=DEV)
    grp = idx.unsqueeze(0) // block
    allow = (grp.T == grp) | (grp.T > grp)
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

def sat_mask_cached(new_len: int, cached_len: int, block=SAT_BLOCK, structured: bool = False):
    total_len = cached_len + new_len
    if structured:
        return StructuredAttentionMask("sat", q_len=new_len, k_len=total_len, query_base=cached_len, block=block)
    q_idx = torch.arange(cached_len, total_len, device=DEV).unsqueeze(1)
    k_idx = torch.arange(total_len, device=DEV).unsqueeze(0)
    q_grp = q_idx // block
    k_grp = k_idx // block
    allow = q_grp >= k_grp
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)


# ───────────────────────── Checkpoint helpers ─────────────────────────

# ───────────────────────── Delta Checkpoints (weight-only, async) ─────────────────────────
_delta_lock = threading.Lock()
_delta_thread: Optional[threading.Thread] = None

def _sha256_file(path: pathlib.Path) -> str:
    """Compute SHA256 of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _do_delta_save(tensors: dict, path: pathlib.Path, meta: dict):
    """Background worker: write weight-only checkpoint + checksum."""
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        tmp = path.with_suffix(path.suffix + ".dtmp")
        torch.save({"weights": tensors, **meta}, tmp, _use_new_zipfile_serialization=False)
        digest = _sha256_file(tmp)
        tmp.replace(path)
        # Write sidecar checksum
        path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
        print(f"  [delta] saved {path.name} ({digest[:12]}...)")
    except Exception as e:
        print(f"  [delta] FAILED {path.name}: {e}")


def _delete_delta_artifacts(path: pathlib.Path):
    for sidecar in (
        path,
        path.with_suffix(".sha256"),
        path.with_suffix(path.suffix + ".upload.sha256"),
        path.with_suffix(path.suffix + ".dtmp"),
    ):
        try:
            if sidecar.exists():
                sidecar.unlink()
        except Exception:
            pass


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module when torch.compile wrapped it."""
    return getattr(module, "_orig_mod", module)

def _checkpoint_state_dict(module: nn.Module) -> dict:
    """State dict with stable keys, even when module is torch.compile'd."""
    return _unwrap_compiled_module(module).state_dict()

def _strip_orig_mod_prefix(state: dict) -> dict:
    """Accept older deltas accidentally saved from compiled modules."""
    if not isinstance(state, dict):
        return state
    prefix = "_orig_mod."
    if not any(isinstance(k, str) and k.startswith(prefix) for k in state):
        return state
    return {
        (k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k): v
        for k, v in state.items()
    }

def _cat_legacy_weight_blocks(blocks: list) -> Optional[torch.Tensor]:
    if not blocks or not all(torch.is_tensor(t) for t in blocks):
        return None
    first = blocks[0]
    tail_shape = tuple(first.shape[1:])
    if any(t.dtype != first.dtype or t.device != first.device for t in blocks):
        return None
    if any(t.ndim != first.ndim or tuple(t.shape[1:]) != tail_shape for t in blocks):
        return None
    return torch.cat(blocks, dim=0).contiguous()

def _fuse_qkv_in_state_dict(sd: dict) -> dict:
    """Fold legacy q/k/v.weight triples into qkv.weight before loading/filtering."""
    if not isinstance(sd, dict):
        return sd
    prefixes = set()
    for key in list(sd.keys()):
        for suffix in (".q.weight", ".k.weight", ".v.weight"):
            if isinstance(key, str) and key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
    for prefix in prefixes:
        qk, kk, vk = prefix + ".q.weight", prefix + ".k.weight", prefix + ".v.weight"
        fk = prefix + ".qkv.weight"
        if qk in sd and kk in sd and vk in sd and fk not in sd:
            fused = _cat_legacy_weight_blocks([sd[qk], sd[kk], sd[vk]])
            if fused is not None:
                sd[fk] = fused
                sd.pop(qk)
                sd.pop(kk)
                sd.pop(vk)
    return sd

def _split_qkv_in_state_dict_for_test(sd: dict) -> dict:
    out = dict(sd)
    for key in list(out.keys()):
        if not isinstance(key, str) or not key.endswith(".qkv.weight"):
            continue
        base = key[: -len(".qkv.weight")]
        q, k, v = out.pop(key).chunk(3, dim=0)
        out[base + ".q.weight"] = q.clone()
        out[base + ".k.weight"] = k.clone()
        out[base + ".v.weight"] = v.clone()
    return out

def _clone_opt_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)

def _optimizer_param_name_lookup(core, ar_h, sat_h, nat_h=None) -> dict[int, str]:
    out = {}
    for prefix, module in (("core", core), ("ar", ar_h), ("sat", sat_h), ("nat", nat_h)):
        if module is None:
            continue
        for name, param in module.named_parameters():
            out.setdefault(id(param), f"{prefix}.{name}")
    return out

def _optimizer_group_param_names(opt, core, ar_h, sat_h, nat_h=None) -> List[List[str]]:
    lookup = _optimizer_param_name_lookup(core, ar_h, sat_h, nat_h)
    return [
        [lookup.get(id(param), f"<unknown:{id(param)}>") for param in group["params"]]
        for group in opt.param_groups
    ]

def _legacy_names_for_current_param(name: str) -> List[str]:
    if name.endswith(".qkv.weight"):
        base = name[: -len(".qkv.weight")]
        return [base + ".q.weight", base + ".k.weight", base + ".v.weight"]
    return [name]

def _fuse_legacy_optimizer_param_state(states: List[dict]) -> Optional[dict]:
    if len(states) < 2 or any(not isinstance(state, dict) for state in states):
        return None
    common = set(states[0])
    for state in states[1:]:
        common &= set(state)
    out = {}
    for key in common:
        vals = [state[key] for state in states]
        if all(torch.is_tensor(v) for v in vals):
            shape = vals[0].shape
            if vals[0].ndim > 0 and all(v.shape == shape for v in vals[1:]):
                out[key] = torch.cat([v.detach().clone() for v in vals], dim=0).contiguous()
            else:
                out[key] = vals[0].detach().clone()
        else:
            out[key] = copy.deepcopy(vals[0])
    return out

def _fuse_legacy_qkv_optimizer_state(opt_state: dict, opt, core, ar_h, sat_h, nat_h=None) -> Optional[dict]:
    """Remap pre-QKV-fusion AdamW state to the current fused parameter layout."""
    if not isinstance(opt_state, dict) or "state" not in opt_state or "param_groups" not in opt_state:
        return None
    current_sd = opt.state_dict()
    current_names = _optimizer_group_param_names(opt, core, ar_h, sat_h, nat_h)
    legacy_names = [
        [legacy for name in group_names for legacy in _legacy_names_for_current_param(name)]
        for group_names in current_names
    ]
    if len(legacy_names) != len(opt_state.get("param_groups", [])):
        return None

    legacy_name_to_pid = {}
    for group_idx, names in enumerate(legacy_names):
        old_params = list(opt_state["param_groups"][group_idx].get("params", []))
        if len(names) != len(old_params):
            return None
        for name, pid in zip(names, old_params):
            legacy_name_to_pid[name] = pid

    new_groups = []
    for group_idx, current_group in enumerate(current_sd["param_groups"]):
        new_group = copy.deepcopy(opt_state["param_groups"][group_idx])
        new_group["params"] = list(current_group["params"])
        if "param_names" in new_group:
            new_group["param_names"] = list(current_names[group_idx])
        new_groups.append(new_group)

    old_states = opt_state.get("state", {})
    new_states = {}
    for group_names, current_group in zip(current_names, current_sd["param_groups"]):
        for name, new_pid in zip(group_names, current_group["params"]):
            legacy_set = _legacy_names_for_current_param(name)
            if len(legacy_set) > 1:
                old_pids = [legacy_name_to_pid.get(legacy) for legacy in legacy_set]
                if all(pid in old_states for pid in old_pids):
                    fused = _fuse_legacy_optimizer_param_state([old_states[pid] for pid in old_pids])
                    if fused is not None:
                        new_states[new_pid] = fused
                continue
            old_pid = legacy_name_to_pid.get(name)
            if old_pid in old_states:
                new_states[new_pid] = {key: _clone_opt_value(value) for key, value in old_states[old_pid].items()}

    return {"state": new_states, "param_groups": new_groups}

def save_delta(core, ar_h, sat_h, nat_h, step: int, seen_tok: int, save_dir: pathlib.Path, phase_name: str):
    """Save weight-only delta in background thread. Non-blocking."""
    global _delta_thread
    # Wait for any previous delta write to finish
    if _delta_thread is not None and _delta_thread.is_alive():
        _delta_thread.join(timeout=60)
    # Snapshot weights to CPU (detach from GPU graph)
    with _delta_lock:
        tensors = {
            "core": {k: v.detach().cpu() for k, v in _checkpoint_state_dict(core).items()},
            "ar":   {k: v.detach().cpu() for k, v in _checkpoint_state_dict(ar_h).items()},
            "sat":  {k: v.detach().cpu() for k, v in _checkpoint_state_dict(sat_h).items()},
        }
        if nat_h is not None:
            tensors["nat"] = {k: v.detach().cpu() for k, v in _checkpoint_state_dict(nat_h).items()}
    meta = {"step": step, "seen_tok": seen_tok, "wall_time": time.time(), "delta": True}
    path = save_dir / f"{phase_name}_delta_step{step:08d}.pt"
    _delta_thread = threading.Thread(target=_do_delta_save, args=(tensors, path, meta), daemon=True)
    _delta_thread.start()

def _prune_delta_files_to_count(save_dir: pathlib.Path, phase_name: str, keep_count: int):
    """Keep only the newest keep_count complete delta files."""
    try:
        pattern = f"{phase_name}_delta_step*.pt"
        deltas = sorted(
            [p for p in save_dir.glob(pattern) if p.stat().st_size > 0],
            key=lambda p: p.stat().st_mtime
        )
        excess = len(deltas) - max(0, keep_count)
        if excess > 0:
            for p in deltas[:excess]:
                _delete_delta_artifacts(p)
                print(f"  [delta-prune] deleted {p.name}")
    except Exception as e:
        print(f"  [delta-prune] error: {e}")


def _prune_deltas(save_dir: pathlib.Path, phase_name: str, max_deltas: int):
    """Keep only the most recent max_deltas delta files."""
    if max_deltas is None or max_deltas <= 0:
        return
    _prune_delta_files_to_count(save_dir, phase_name, max_deltas)

def _load_module_state_compatible(module: nn.Module, state: dict, label: str = "module") -> int:
    """Load matching tensors only; skip obsolete untied vocab matrices for tied heads."""
    if not isinstance(state, dict):
        return 0
    state = _strip_orig_mod_prefix(state)
    tgt_sd = module.state_dict()
    tied = bool(getattr(module, "tie_weights", False))
    filt = {}
    skipped = []
    for k, v in state.items():
        if tied and k == "proj.weight":
            skipped.append(k)
            continue
        if k in tgt_sd and hasattr(v, "shape") and v.shape == tgt_sd[k].shape:
            filt[k] = v
        else:
            skipped.append(k)
    if filt:
        module.load_state_dict(filt, strict=False)
    if tied and skipped:
        print(f"[ckpt] {label}: tied head active; skipped old untied tensors: {', '.join(skipped[:4])}{'...' if len(skipped)>4 else ''}")
    return len(filt)

def load_delta(path: pathlib.Path, core, ar_h, sat_h, nat_h=None):
    """Load weight-only delta. Returns (step, seen_tok) or raises."""
    # Verify checksum if sidecar exists
    sha_path = path.with_suffix(".sha256")
    if sha_path.exists():
        expected = sha_path.read_text().split()[0]
        actual = _sha256_file(path)
        if expected != actual:
            raise ValueError(f"Checksum mismatch for {path.name}: expected {expected[:12]}... got {actual[:12]}...")
        print(f"  [delta] checksum OK for {path.name}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("delta"):
        raise ValueError(f"{path.name} is not a delta checkpoint")
    core.load_state_dict(_fuse_qkv_in_state_dict(_strip_orig_mod_prefix(ck["weights"]["core"])))
    _load_module_state_compatible(ar_h, ck["weights"].get("ar", {}), "ar")
    _load_module_state_compatible(sat_h, ck["weights"].get("sat", {}), "sat")
    if nat_h is not None:
        nat_sd = ck["weights"].get("nat")
        if nat_sd is not None:
            _load_module_state_compatible(nat_h, nat_sd, "nat")
        else:
            print("[nat] Delta has no NAT head; keeping fresh NAT initialization")
    return ck.get("step", 0), ck.get("seen_tok", 0)

def _flush_delta():
    """Wait for any in-flight delta save to complete."""
    global _delta_thread
    if _delta_thread is not None and _delta_thread.is_alive():
        print("  [delta] flushing in-flight write...")
        _delta_thread.join(timeout=120)

def save_ckpt(path: pathlib.Path, core, ar_h, sat_h, nat_h, opt, scaler, meta):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {
        "core": _checkpoint_state_dict(core), "ar": _checkpoint_state_dict(ar_h), "sat": _checkpoint_state_dict(sat_h),
        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "cfg": meta.get("cfg"), "tokenizer_id": TOKENIZER_ID,
        "tokenizer_json": tok.backend_tokenizer.to_str(),
        "transformers_version": __import__("transformers").__version__,
        "tokenizers_version": __import__("tokenizers").__version__,
        "tie_weights": meta.get("tie_weights", False),
        **{k: v for k, v in meta.items() if k not in ("cfg", "tie_weights")}
    }
    if nat_h is not None:
        state["nat"] = _checkpoint_state_dict(nat_h)
    torch.save(state, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(path)
    (path.parent / "latest.json").write_text(json.dumps({"path": str(path), "step": meta["step"]}))
    print(f"\n✓ saved checkpoint {path.name}")

def load_ckpt(path, core, ar_h, sat_h, opt, scaler, nat_h=None):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location="cpu")
    if ck is None: raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(_fuse_qkv_in_state_dict(_strip_orig_mod_prefix(ck["core"])))
    _load_module_state_compatible(ar_h, ck.get("ar", {}), "ar")
    _load_module_state_compatible(sat_h, ck.get("sat", {}), "sat")
    if nat_h is not None:
        if "nat" in ck:
            _load_module_state_compatible(nat_h, ck["nat"], "nat")
        else:
            print("[nat] Checkpoint has no NAT head; keeping fresh NAT initialization")
    try:
        opt.load_state_dict(ck["opt"])
    except Exception as exc:
        fused_opt = _fuse_legacy_qkv_optimizer_state(ck.get("opt"), opt, core, ar_h, sat_h, nat_h)
        if fused_opt is not None:
            try:
                opt.load_state_dict(fused_opt)
                print("[ckpt] Converted legacy q/k/v optimizer state to fused qkv layout")
            except Exception as exc2:
                print(f"[ckpt] WARNING: optimizer state incompatible; resetting optimizer ({type(exc).__name__}: {exc}; qkv remap failed: {type(exc2).__name__}: {exc2})")
        else:
            print(f"[ckpt] WARNING: optimizer state incompatible; resetting optimizer ({type(exc).__name__}: {exc})")
    try:
        scaler.load_state_dict(ck["scaler"])
    except Exception as exc:
        print(f"[ckpt] WARNING: scaler state incompatible; resetting scaler ({type(exc).__name__}: {exc})")
    # Restore tokenizer from checkpoint if available
    if "tokenizer_json" in ck:
        try:
            from tokenizers import Tokenizer as _Tokenizer
            tok.backend_tokenizer = _Tokenizer.from_str(ck["tokenizer_json"])
            print("[tokenizer] Restored from checkpoint")
        except Exception as e:
            print(f"[tokenizer] WARNING: could not restore from checkpoint: {e}")
    # Warn if transformers version changed since checkpoint was saved
    if "transformers_version" in ck:
        import transformers as _tf
        if ck["transformers_version"] != _tf.__version__:
            print(f"[tokenizer] WARNING: checkpoint saved with transformers={ck['transformers_version']}, now running {_tf.__version__}")
    return ck.get("step", 0), ck.get("seen_tok", 0), ck.get("wall_time", time.time())

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return 0
    ck = _try_load(p, map_location="cpu")
    if ck is None: return 0
    sd = ck.get(key, ck) if key else ck
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = _strip_orig_mod_prefix(sd)
    sd = _fuse_qkv_in_state_dict(dict(sd)) if isinstance(sd, dict) else sd
    if not isinstance(sd, dict):
        return 0
    tgt_sd = tgt.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt_sd and hasattr(v, "shape") and v.shape == tgt_sd[k].shape}
    if filt: tgt.load_state_dict(filt, strict=False)
    return len(filt)

def infer_cfg_from_ckpt(path: pathlib.Path):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return None
    sd = _try_load(p, map_location="cpu")
    if sd is None: return None
    if "cfg" in sd: return dict(sd["cfg"])
    return None


# ───────────────────────── Training Logic ─────────────────────────
def _parse_grow_plan(s: str) -> List[int]:
    return sorted(set([int(x.strip()) for x in s.split(",") if x.strip() and int(x.strip()) >= 128]))

def _count_enabled_params(*modules) -> int:
    seen_data_ptrs = set()
    total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.data_ptr() not in seen_data_ptrs:
                seen_data_ptrs.add(p.data_ptr())
                total += p.numel()
    return total

def _target_token_ratio(args) -> float:
    if getattr(args, "token_param_ratio", 0.0) and args.token_param_ratio > 0:
        return float(args.token_param_ratio)
    if str(getattr(args, "preset", "")).startswith("agillm4_"):
        return AGILLM4_TOKEN_PARAM_RATIO
    return 51.2 if args.chilla_max_double else 25.0

def _phase_freeze(core: nn.Module, *, freeze_core: bool, unfreeze_ln: bool, train_emb: bool):
    for p in core.parameters(): p.requires_grad = not freeze_core
    if freeze_core:
        if unfreeze_ln:
            for blk in core.blocks:
                for p in blk.ln1.parameters(): p.requires_grad = True
                for p in blk.ln2.parameters(): p.requires_grad = True
            for p in core.ln.parameters(): p.requires_grad = True
        if train_emb:
            for p in core.emb.parameters(): p.requires_grad = True

def _optimizer_param_groups(core, ar_h, sat_h, lr_core: float, lr_head: float, nat_h=None):
    # Shared/tied vocab projections must appear in only one optimizer group.
    # VRAM-first AGILLM-4 uses one embedding/projection tensor for AR/SAT/NAT.
    seen: set[int] = set()
    groups = []
    def add(params, lr):
        unique = []
        for p in params:
            if not p.requires_grad:
                continue
            key = id(p)
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        if unique:
            groups.append({"params": unique, "lr": lr})
    add(core.parameters(), lr_core)
    add(ar_h.parameters(), lr_head)
    add(sat_h.parameters(), lr_head)
    if nat_h is not None:
        add(nat_h.parameters(), lr_head)
    return groups

def make_optimizer(args, core, ar_h, sat_h, lr_core: float, lr_head: float, nat_h=None):
    groups = _optimizer_param_groups(core, ar_h, sat_h, lr_core, lr_head, nat_h)
    opt_name = getattr(args, "optimizer", "adamw")
    if opt_name == "adamw":
        return torch.optim.AdamW(groups)
    if opt_name in {"adamw8bit", "paged_adamw8bit"}:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                f"--optimizer {opt_name} requires bitsandbytes. Install it in the training env first."
            ) from exc
        if opt_name == "paged_adamw8bit":
            return bnb.optim.PagedAdamW8bit(groups)
        return bnb.optim.AdamW8bit(groups)
    raise ValueError(f"unknown optimizer: {opt_name}")

def _nat_ids_for_training(ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens and max_tokens > 0 and ids.size(1) > max_tokens:
        return ids[:, -max_tokens:]
    return ids

def _train_phase(
    args, phase_name: str,
    core, ar_h, sat_h, nat_h, opt, scaler,
    start_step, seen_tok, resume_wall_time,
    cfg, source, steps, block_size, batch_size,
    chat_cfg: dict,
    max_ckpts: int,
    target_tokens_override: Optional[int] = None,
    tie_weights: bool = False,
    streaming: bool = True
):
    BLOCK = block_size
    BATCH = batch_size
    if target_tokens_override is not None:
        target_tokens = target_tokens_override
    else:
        ratio = _target_token_ratio(args)
        param_count = _count_enabled_params(core, ar_h, sat_h, nat_h)
        target_tokens = int(ratio * param_count)
        print(f"[{phase_name}] token_param_ratio={ratio:g} param_count={param_count:,} target_tokens={target_tokens:,}")
    if steps:
        phase_target_tokens = steps * BLOCK * BATCH
        total_tokens_needed = seen_tok + phase_target_tokens
    else:
        total_tokens_needed = target_tokens
        if total_tokens_needed <= seen_tok:
            print(f"[{phase_name}] target {total_tokens_needed} already reached.")
            return start_step, seen_tok, resume_wall_time
    stream = token_stream(
        source, total_tokens_needed, seed=42,
        chat=chat_cfg.get("chat", False),
        chat_messages_key=chat_cfg.get("key", "messages"),
        sft_add_generation_prompt=chat_cfg.get("gen_prompt", False),
        dataset_field_text=chat_cfg.get("text_field", "text"),
        streaming=streaming
    )
    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_gate = nn.CrossEntropyLoss()
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    pbar = SafeProgress(total=total_tokens_needed, initial=seen_tok, unit="tok")
    grow_plan = _parse_grow_plan(args.grow_plan) if args.auto_grow else []
    buf: list[int] = []
    batch_accum: list[list[int]] = []
    step = start_step
    steps_since_last_grow = 0
    oom_retries = 0
    MAX_OOM_RETRIES = 2
    now_wall = time.time()
    last_save_mono = time.monotonic() - (now_wall - (resume_wall_time or now_wall))
    last_delta_step = start_step
    last_heartbeat_mono = time.monotonic()
    print(f"[{phase_name}] Starting. Goal: {total_tokens_needed:,} tokens. Batch={BATCH}, Block={BLOCK}")
    print(
        f"[{phase_name}] AR_ONLY={args.ar_only}, SAT_EVERY={args.sat_every}, "
        f"NAT_EVERY={args.nat_every}, TIE_WEIGHTS={tie_weights}, STREAMING={streaming}"
    )
    _flush_flag = [False]
    def _on_flush_signal(signum, frame):
        _flush_flag[0] = True
        print(f"\n[{phase_name}] flush signal received; will checkpoint at next step")
    try:
        signal.signal(signal.SIGUSR1, _on_flush_signal)
        print(f"[{phase_name}] on-demand flush ready: kill -USR1 {os.getpid()}  or  touch {pathlib.Path(args.save_dir) / 'FLUSH_NOW'}")
    except (ValueError, OSError):
        pass
    _DBS = _dblock_init(core, args) if getattr(args,'dblock',False) else None
    if DEV.type == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(
                f"[vram] training-start cache cleared: "
                f"alloc={torch.cuda.memory_allocated() / (1024**3):.2f}GB "
                f"reserved={torch.cuda.memory_reserved() / (1024**3):.2f}GB "
                f"structured_masks={use_structured_masks(args)}",
                flush=True,
            )
        except Exception:
            pass
    while seen_tok < total_tokens_needed:
        _profile_batch = _DBS is not None and int(getattr(args, "profile_steps", 0) or 0) > 0 and int(_DBS.get("profile_n", 0)) < int(getattr(args, "profile_steps", 0) or 0)
        _data_t = time.perf_counter() if _profile_batch else None
        try:
            while len(buf) < BLOCK:
                buf.append(next(stream))
        except StopIteration:
            break
        if _profile_batch:
            try:
                import dblocks_train as _db_prof
                _db_prof._profile_add(_DBS, "data_stream", time.perf_counter() - _data_t)
            except Exception:
                pass
        seq = buf[:BLOCK]
        buf = buf[BLOCK:]
        batch_accum.append(seq)
        if len(batch_accum) < BATCH:
            continue
        _tensor_t = time.perf_counter() if _profile_batch else None
        ids = torch.tensor(batch_accum, device=DEV)
        if _profile_batch:
            if DEV.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                import dblocks_train as _db_prof
                _db_prof._profile_add(_DBS, "tensor", time.perf_counter() - _tensor_t)
            except Exception:
                pass
        batch_accum = []
        tgt_ar = ids.clone()
        try:
            if getattr(args, "dblock", False):
                loss_value = _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, _DBS)
            else:
                with amp(args.amp):
                    h_ar = core(ids, causal_mask(ids.size(1), structured=use_structured_masks(args)))
                    logits_ar = ar_h(h_ar)[:, :-1]
                    loss_ar = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1))
                loss_value = float(loss_ar.detach().item())
                scaler.scale(loss_ar).backward()
                del h_ar, logits_ar, loss_ar
                do_sat = (not args.ar_only) and (args.sat_every <= 1 or ((step + 1) % args.sat_every == 0))
                if do_sat:
                    # Same AR+SAT objective as a summed loss, but sequential backward keeps
                    # only one core-forward activation graph live at a time on 24GB cards.
                    with amp(args.amp):
                        h_sat = core(ids, sat_mask(ids.size(1), structured=use_structured_masks(args)))
                        logits_sat, gate = sat_h(h_sat[:, -SAT_BLOCK:])
                        tgt_sat = ids[:, 1:SAT_BLOCK+1]
                        loss_sat = ce_tok(logits_sat.reshape(-1, VOCAB), tgt_sat.reshape(-1))
                        if gate is not None:
                            loss_sat += EMIT_LAMBDA * ce_gate(gate, torch.ones(ids.size(0), device=DEV, dtype=torch.long))
                    loss_value += float(loss_sat.detach().item())
                    scaler.scale(loss_sat).backward()
                    del h_sat, logits_sat, loss_sat
                do_nat = (
                    nat_h is not None
                    and (not args.ar_only)
                    and args.nat_every > 0
                    and (args.nat_every <= 1 or ((step + 1) % args.nat_every == 0))
                )
                if do_nat:
                    nat_ids = _nat_ids_for_training(ids, args.nat_max_tokens)
                    with amp(args.amp):
                        # Mask-predict (CMLM) objective: corrupt a fraction of positions
                        # with BLANK and reconstruct them from surrounding context. The
                        # old CTC objective fed the clean target as input, so the head
                        # only learned to copy and collapsed at inference on all-BLANK
                        # input. This conditions on real context and cannot collapse.
                        nat_in = nat_ids.clone()
                        ratio = min(max(float(args.nat_mask_ratio), 0.05), 0.95)
                        mask = torch.rand(nat_in.shape, device=nat_in.device) < ratio
                        if not bool(mask.any()):
                            mask[..., -1] = True
                        nat_in[mask] = BLANK
                        h_nat = core(nat_in, None)
                        logits_nat = nat_h(h_nat)
                        loss_nat = F.cross_entropy(logits_nat[mask].float(), nat_ids[mask])
                        loss_nat = float(args.nat_loss_weight) * loss_nat
                    loss_value += float(loss_nat.detach().item())
                    scaler.scale(loss_nat).backward()
                    del nat_ids, nat_in, mask, h_nat, logits_nat, loss_nat
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                batch_accum = []
                opt.zero_grad(set_to_none=True)
                scaler = GradScaler(enabled=(args.amp and _needs_grad_scaler()))
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                oom_retries += 1
                if oom_retries <= MAX_OOM_RETRIES:
                    print(f"\n[{phase_name} OOM] Retry {oom_retries}/{MAX_OOM_RETRIES} at Batch={BATCH}, clearing VRAM...")
                    time.sleep(2)
                    continue
                oom_retries = 0
                if BATCH > 1:
                    print(f"\n[{phase_name} OOM] Reducing Batch: {BATCH} -> {BATCH - 1} (after {MAX_OOM_RETRIES} retries)")
                    BATCH -= 1
                    time.sleep(2)
                else:
                    new_block = max(128, int(BLOCK * 0.8))
                    new_block = max(128, (new_block // 128) * 128)
                    if new_block >= BLOCK:
                        new_block = max(128, BLOCK - 128)
                    print(f"\n[{phase_name} OOM] Reducing Block: {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    time.sleep(2)
                steps_since_last_grow = 0
                continue
            raise
        step += 1
        # Periodic tokenizer spot-check: verify training data has spaces
        if step % 1000 == 0:
            try:
                sample_text = tok.decode(ids[0][:50].tolist(), skip_special_tokens=True)
                if len(sample_text) > 20 and " " not in sample_text:
                    print(f"\n[tokenizer] ALERT step {step}: decoded batch has NO SPACES!")
                    print(f"  Sample: {repr(sample_text[:80])}")
                    print("  Check transformers version!")
            except Exception:
                pass
        oom_retries = 0
        toks_processed = BLOCK * BATCH
        seen_tok += toks_processed
        pbar.set_postfix(loss=f"{loss_value:.3f}", B=BATCH, L=BLOCK)
        pbar.update(toks_processed)
        empty_cache_every = int(getattr(args, "empty_cache_every_steps", 0) or 0)
        if DEV.type == "cuda" and empty_cache_every > 0 and (step % empty_cache_every) == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        heartbeat_every = int(getattr(args, "heartbeat_every_sec", 300) or 0)
        now_mono = time.monotonic()
        if heartbeat_every > 0 and now_mono - last_heartbeat_mono >= heartbeat_every:
            mem = ""
            if DEV.type == "cuda":
                try:
                    mem = (
                        f" gpu_alloc={torch.cuda.memory_allocated() / (1024**3):.2f}GB"
                        f" gpu_reserved={torch.cuda.memory_reserved() / (1024**3):.2f}GB"
                        f" gpu_peak={torch.cuda.max_memory_allocated() / (1024**3):.2f}GB"
                    )
                except Exception:
                    mem = ""
            print(
                f"[heartbeat] phase={phase_name} pid={os.getpid()} step={step} "
                f"seen_tok={seen_tok} loss={loss_value:.3f} B={BATCH} L={BLOCK} "
                f"dblock={bool(getattr(args, 'dblock', False))} structured_masks={use_structured_masks(args)}{mem}",
                flush=True,
            )
            last_heartbeat_mono = now_mono
        _flush_sentinel = pathlib.Path(args.save_dir) / "FLUSH_NOW"
        if _flush_flag[0] or _flush_sentinel.exists():
            _flush_flag[0] = False
            try:
                _flush_sentinel.unlink()
            except FileNotFoundError:
                pass
            _ck_name = f"{phase_name}_step{step:08d}.pt"
            _flush_delta()
            _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
            save_ckpt(pathlib.Path(args.save_dir) / _ck_name, core, ar_h, sat_h, nat_h, opt, scaler,
                      meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights})
            last_save_mono = time.monotonic()
            _prune_deltas(pathlib.Path(args.save_dir), phase_name, args.delta_max_keep)
            last_delta_step = step
            print(f"[{phase_name}] ON-DEMAND flush saved {_ck_name} at step {step}")
        if args.save_every_sec > 0:
            now_mono = time.monotonic()
            if now_mono - last_save_mono >= args.save_every_sec:
                ck_name = f"{phase_name}_step{step:08d}.pt"
                _flush_delta()  # wait for any in-flight delta before full save
                _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
                save_ckpt(pathlib.Path(args.save_dir) / ck_name, core, ar_h, sat_h, nat_h, opt, scaler,
                          meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights})
                last_save_mono = now_mono
                # Prune old deltas after a full save (they're superseded)
                _prune_deltas(pathlib.Path(args.save_dir), phase_name, args.delta_max_keep)
                last_delta_step = step  # reset delta counter after full save
        # ── Delta checkpoint (step-based, weight-only, async) ──
        if args.delta_every_steps > 0 and (step - last_delta_step) >= args.delta_every_steps:
            save_root = pathlib.Path(args.save_dir)
            # AGILLM4 production runs on small rented disks. When keep=1, prune
            # old deltas before the async writer creates the next multi-GB file.
            if args.delta_max_keep and args.delta_max_keep > 0:
                _flush_delta()
                _prune_delta_files_to_count(save_root, phase_name, args.delta_max_keep - 1)
            save_delta(core, ar_h, sat_h, nat_h, step, seen_tok, save_root, phase_name)
            last_delta_step = step
        if args.auto_grow:
            steps_since_last_grow += 1
            if steps_since_last_grow >= args.grow_every_steps:
                steps_since_last_grow = 0
                try:
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        BLOCK = grow_plan[idx + 1]
                        print(f"[{phase_name} Grow] Block -> {BLOCK}")
                        if DEV.type == "cuda": torch.cuda.empty_cache()
                except ValueError:
                    grow_plan = sorted(set(grow_plan + [BLOCK]))
    pbar.close()
    _flush_delta()  # ensure any in-flight delta completes before final save
    if phase_name != "sft":
        save_ckpt(pathlib.Path(args.save_dir) / f"{phase_name}_final.pt", core, ar_h, sat_h, nat_h, opt, scaler,
                  meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights})
    else:
        print("[sft] Skipping duplicate sft_final.pt; final.pt will contain the SFT result.")
    return step, seen_tok, time.time()


# ───────────────────────── Main Orchestrator ─────────────────────────
def train(args):
    cfg = PRESETS[args.preset].copy()
    tie_weights = args.tie_weights
    print_expansion_info(cfg, tie_weights)
    if not args.fresh:
        src_probe = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        prev_cfg = infer_cfg_from_ckpt(src_probe)
    else: prev_cfg = None
    if prev_cfg:
        cfg.update({k: v for k, v in prev_cfg.items() if k in cfg})
        if args.x2 and prev_cfg.get("layers"): cfg["layers"] = max(cfg["layers"], prev_cfg["layers"] * 2)
    if args.rank: cfg["rank"] = args.rank
    if args.x2 and not prev_cfg: cfg["layers"] *= 2
    print(f"Config: {cfg}")
    print(
        "AGILLM-4 runtime: "
        f"attn_backend={args.attn_backend} grad_checkpoint={args.grad_checkpoint} "
        f"sublinear_window={args.sublinear_window} sublinear_stride={args.sublinear_stride} "
        f"sublinear_max_anchors={args.sublinear_max_anchors} sublinear_chunk={args.sublinear_chunk} "
        f"sublinear_sinks={args.sublinear_sinks} sublinear_recent_anchors={args.sublinear_recent_anchors} "
        f"sublinear_pooled_landmarks={args.sublinear_pooled_landmarks}"
    )
    core = Encoder(
        cfg,
        tie_weights=tie_weights,
        attn_backend=args.attn_backend,
        grad_checkpoint=args.grad_checkpoint,
        sublinear_window=args.sublinear_window,
        sublinear_stride=args.sublinear_stride,
        sublinear_max_anchors=args.sublinear_max_anchors,
        sublinear_chunk=args.sublinear_chunk,
        sublinear_sinks=args.sublinear_sinks,
        sublinear_recent_anchors=args.sublinear_recent_anchors,
        sublinear_pooled_landmarks=args.sublinear_pooled_landmarks,
        anchor_memory=getattr(args, "anchor_memory", DEFAULT_ANCHOR_MEMORY),
        anchor_stride=getattr(args, "anchor_stride", DEFAULT_ANCHOR_STRIDE),
        anchor_max=getattr(args, "anchor_max", DEFAULT_ANCHOR_MAX),
        anchor_position=getattr(args, "anchor_position", DEFAULT_ANCHOR_POSITION),
    ).to(DEV)
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    sat_h = SATHead(cfg["d"], mode="var", tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    nat_h = NATHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    total_params = _count_enabled_params(core, ar_h, sat_h, nat_h)
    print(f"Total parameters: {total_params:,}")
    if tie_weights:
        print(f"{Colors.WARN}[weight-tying] Embedding and AR/SAT/NAT vocab projections share one tensor (VRAM-first){Colors.RESET}")
    if not args.fresh:
        src = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        src = _resolve_ckpt(src)
        if src:
            loaded = _safe_load_any(src, core, key="core")
            _safe_load_any(src, ar_h, key="ar")
            _safe_load_any(src, sat_h, key="sat")
            nat_loaded = _safe_load_any(src, nat_h, key="nat")
            if not nat_loaded:
                print("[nat] Warm-start source has no NAT head; NAT head initialized fresh")
            if loaded: print(f"Warm-start loaded from {src}")
    _phase_freeze(core, freeze_core=args.freeze_core, unfreeze_ln=args.unfreeze_ln, train_emb=args.train_emb)
    opt = make_optimizer(args, core, ar_h, sat_h, args.lr_core, args.lr_head, nat_h)
    scaler = GradScaler(enabled=(args.amp and _needs_grad_scaler()))
    start_step, seen_tok, last_wall = 0, 0, None
    if args.resume_delta and not args.fresh:
        delta_step, delta_tok = load_delta(pathlib.Path(args.resume_delta), core, ar_h, sat_h, nat_h)
        start_step, seen_tok, last_wall = delta_step, delta_tok, None
        print(f"Resumed from DELTA at step {start_step} (optimizer state reset — momentum rebuilds in ~100 steps)")
    elif args.resume and not args.fresh:
        start_step, seen_tok, last_wall = load_ckpt(pathlib.Path(args.resume), core, ar_h, sat_h, opt, scaler, nat_h)
        print(f"Resumed from step {start_step}")
    if getattr(args, "seed_nat_from_ar", False) and nat_h is not None and ar_h is not None:
        # Seed the non-autoregressive (NAT) head from the trained AR head ("father").
        # Same hidden->vocab projection shape, so NAT starts knowing the token
        # distribution instead of from random/blank -> faster, no collapse.
        with torch.no_grad():
            nat_h.proj.weight.copy_(ar_h.proj.weight)
            if nat_h.proj.bias is not None:
                if getattr(ar_h.proj, "bias", None) is not None:
                    nat_h.proj.bias.copy_(ar_h.proj.bias)
                else:
                    nat_h.proj.bias.zero_()
        print("[nat] Seeded NAT head from the AR head ('father') for the mask-predict objective")
    elif getattr(args, "reinit_nat", False) and nat_h is not None:
        for _m in nat_h.modules():
            if isinstance(_m, nn.Linear):
                nn.init.normal_(_m.weight, mean=0.0, std=0.02)
                if _m.bias is not None:
                    nn.init.zeros_(_m.bias)
        print("[nat] Reinitialized NAT head weights (random) for the mask-predict objective")
    # torch.compile AFTER loading checkpoint (key names differ)
    if args.compile:
        print("[torch.compile] Compiling model...")
        core = torch.compile(core, mode="reduce-overhead")
        ar_h = torch.compile(ar_h, mode="reduce-overhead")
        sat_h = torch.compile(sat_h, mode="reduce-overhead")
        nat_h = torch.compile(nat_h, mode="reduce-overhead")
        print("[torch.compile] Done.")
    step, seen_tok, last_wall = _train_phase(
        args, "pretrain", core, ar_h, sat_h, nat_h, opt, scaler,
        start_step, seen_tok, last_wall, cfg,
        args.source, args.steps, 
        args.block or DEFAULT_BLOCK, 
        args.batch_size or DEFAULT_BATCH,
        chat_cfg={"chat": args.chat, "key": args.chat_messages_key, "gen_prompt": args.sft_add_generation_prompt, "text_field": args.dataset_field_text},
        max_ckpts=args.max_ckpts,
        target_tokens_override=args.target_tokens,
        tie_weights=tie_weights
    )
    if (not args.after_sft_source) and (args.after_sft_steps and args.after_sft_steps > 0):
        args.after_sft_source = DEFAULT_AFTER_SFT_SOURCES
        args.after_sft_chat = True
        if args.after_sft_add_generation_prompt is None: args.after_sft_add_generation_prompt = True
        if not args.after_sft_block: args.after_sft_block = DEFAULT_AFTER_SFT_BLOCK
    if args.after_sft_source and args.after_sft_steps and args.after_sft_steps > 0:
        print("\n[Orchestrator] Starting Post-Pretraining SFT Phase...")
        _phase_freeze(core, 
                      freeze_core=args.after_sft_freeze_core, 
                      unfreeze_ln=args.after_sft_unfreeze_ln, 
                      train_emb=args.after_sft_train_emb)
        opt = make_optimizer(
            args,
            core,
            ar_h,
            sat_h,
            args.after_sft_lr_core or args.lr_core,
            args.after_sft_lr_head or args.lr_head,
            nat_h,
        )
        step, seen_tok, last_wall = _train_phase(
            args, "sft", core, ar_h, sat_h, nat_h, opt, scaler,
            step, seen_tok, last_wall, cfg,
            args.after_sft_source, args.after_sft_steps,
            args.after_sft_block or DEFAULT_AFTER_SFT_BLOCK,
            args.batch_size or DEFAULT_BATCH,
            chat_cfg={
                "chat": args.after_sft_chat, 
                "key": args.after_sft_chat_messages_key,
                "gen_prompt": args.after_sft_add_generation_prompt if args.after_sft_add_generation_prompt is not None else args.sft_add_generation_prompt,
                "text_field": args.after_sft_dataset_field_text
            },
            max_ckpts=args.max_ckpts,
            target_tokens_override=None,
            tie_weights=tie_weights,
            streaming=True
        )
    save_ckpt(pathlib.Path(args.save_dir) / "final.pt", core, ar_h, sat_h, nat_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights})
    print("🎉 All Training Complete")


# ───────────────────────── Sampling ─────────────────────────
def _apply_penalties(logits, ids, n, rep_p, pres_p, freq_p):
    if ids.numel() == 0: return logits
    hist = ids[0, -n:].long() if n > 0 else ids[0].long()
    uniq, counts = torch.unique(hist, return_counts=True)
    if pres_p or freq_p:
        logits[..., uniq] -= (pres_p + freq_p * counts.float())
    if rep_p != 1.0:
        sel = logits[..., uniq]
        logits[..., uniq] = torch.where(sel > 0, sel / rep_p, sel * rep_p)
    return logits

def _sample(logits, T, top_k, top_p, min_p, greedy):
    if greedy: return logits.argmax(-1, keepdim=True)
    probs = (logits / max(T, 1e-8)).softmax(-1)
    if top_k:
        v, i = torch.topk(probs, min(top_k, probs.size(-1)))
        probs = torch.zeros_like(probs).scatter_(-1, i, v)
    if top_p < 1.0:
        s_probs, s_idx = torch.sort(probs, descending=True, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, s_idx, s_probs * (torch.cumsum(s_probs, -1) <= top_p).float())
    if min_p > 0: probs[probs < min_p] = 0
    if probs.sum() == 0: return logits.argmax(-1, keepdim=True)
    return probs.div_(probs.sum()).multinomial(1)

@torch.no_grad()
def infer(args):
    if args.mode == "ar":
        if args.temperature is None: args.temperature = 0.7
        if args.top_k is None: args.top_k = 0
        if args.repetition_penalty is None: args.repetition_penalty = 1.3
        if args.presence_penalty is None: args.presence_penalty = 0.0
        if args.frequency_penalty is None: args.frequency_penalty = 0.3
        if args.penalty_last_n is None: args.penalty_last_n = 128
        if args.var is None: args.var = False
    elif args.mode == "sat":
        if args.temperature is None: args.temperature = 0.5
        if args.top_k is None: args.top_k = 30
        if args.repetition_penalty is None: args.repetition_penalty = 2.0
        if args.presence_penalty is None: args.presence_penalty = 0.6
        if args.frequency_penalty is None: args.frequency_penalty = 1.0
        if args.penalty_last_n is None: args.penalty_last_n = 200
        if args.var is None: args.var = True
    else:
        if args.temperature is None: args.temperature = 1.0
        if args.top_k is None: args.top_k = 0
        if args.repetition_penalty is None: args.repetition_penalty = 1.0
        if args.presence_penalty is None: args.presence_penalty = 0.0
        if args.frequency_penalty is None: args.frequency_penalty = 0.0
        if args.penalty_last_n is None: args.penalty_last_n = 0
        if args.var is None: args.var = False
    path = _resolve_ckpt(pathlib.Path(args.ckpt)) or pathlib.Path(args.ckpt)
    sd = torch.load(path, map_location="cpu")
    # Restore tokenizer from checkpoint if available
    if "tokenizer_json" in sd:
        try:
            from tokenizers import Tokenizer as _Tokenizer
            tok.backend_tokenizer = _Tokenizer.from_str(sd["tokenizer_json"])
            print("[tokenizer] Restored from checkpoint")
        except Exception as e:
            print(f"[tokenizer] WARNING: could not restore from checkpoint: {e}")
    # Warn if transformers version changed since checkpoint was saved
    if "transformers_version" in sd:
        import transformers as _tf
        if sd["transformers_version"] != _tf.__version__:
            print(f"[tokenizer] WARNING: checkpoint saved with transformers={sd['transformers_version']}, now running {_tf.__version__}")
    # Handle delta checkpoints (weight-only, no cfg)
    if sd.get("delta"):
        print("[infer] Delta checkpoint detected, using large preset cfg")
        cfg = PRESETS["large"].copy()
        tie_weights = False
        # Remap: delta stores under sd["weights"]["core"/"ar"/"sat"/"nat"]
        sd["core"] = sd["weights"]["core"]
        sd["ar"]   = sd["weights"]["ar"]
        sd["sat"]  = sd["weights"]["sat"]
        if "nat" in sd["weights"]:
            sd["nat"] = sd["weights"]["nat"]
    else:
        cfg = sd["cfg"]
        tie_weights = sd.get("tie_weights", False)
    plain_output = (
        bool(getattr(args, "plain_output", False))
        or bool(getattr(args, "claude_friendly", False))
        or not sys.stdout.isatty()
    )
    uk_time = get_uk_time()
    ckpt_name = path.name
    if plain_output:
        print(f"[infer] inference_time={uk_time}")
        print(f"[infer] checkpoint={ckpt_name}")
    else:
        print(f"┌─────────────────────────────────────────────────┐")
        print(f"│ INFERENCE @ {uk_time:<35s} │")
        print(f"├─────────────────────────────────────────────────┤")
        print(f"│ Checkpoint: {ckpt_name:<35s} │")
        print(f"└─────────────────────────────────────────────────┘")
    print_expansion_info(cfg, tie_weights, plain=plain_output)
    core = Encoder(
        cfg,
        tie_weights=tie_weights,
        attn_backend=args.attn_backend,
        sublinear_window=args.sublinear_window,
        sublinear_stride=args.sublinear_stride,
        sublinear_max_anchors=args.sublinear_max_anchors,
        sublinear_chunk=args.sublinear_chunk,
        sublinear_sinks=args.sublinear_sinks,
        sublinear_recent_anchors=args.sublinear_recent_anchors,
        sublinear_pooled_landmarks=args.sublinear_pooled_landmarks,
        anchor_memory=getattr(args, "anchor_memory", DEFAULT_ANCHOR_MEMORY),
        anchor_stride=getattr(args, "anchor_stride", DEFAULT_ANCHOR_STRIDE),
        anchor_max=getattr(args, "anchor_max", DEFAULT_ANCHOR_MAX),
        anchor_position=getattr(args, "anchor_position", DEFAULT_ANCHOR_POSITION),
    ).to(DEV)
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    sat_h = SATHead(cfg["d"]).to(DEV)
    nat_h = NATHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV) if ("nat" in sd or args.mode == "nat") else None
    core.load_state_dict(sd["core"])
    ar_h.load_state_dict(sd["ar"])
    sat_h.load_state_dict(sd["sat"])
    if nat_h is not None:
        if "nat" not in sd:
            raise ValueError("NAT inference requested, but this checkpoint has no NAT head")
        nat_h.load_state_dict(sd["nat"])
    core.eval()
    ar_h.eval()
    sat_h.eval()
    if nat_h is not None:
        nat_h.eval()
    total_params = _count_enabled_params(core, ar_h, sat_h, nat_h)
    if total_params >= 1_000_000_000:
        param_str = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params >= 1_000_000:
        param_str = f"{total_params / 1_000_000:.2f}M"
    elif total_params >= 1_000:
        param_str = f"{total_params / 1_000:.2f}K"
    else:
        param_str = f"{total_params}"
    print(f"Model size: {param_str} parameters ({total_params:,})")
    prompt_tokens = tok.encode(args.prompt)
    prompt_len = len(prompt_tokens)
    ids = torch.tensor([prompt_tokens], device=DEV)
    if ids.size(1) == 0: 
        ids = torch.tensor([[EOS]], device=DEV)
        prompt_len = 1
    mode_str = args.mode
    if args.mode == "sat":
        mode_str = f"sat-{'var' if args.var else 'fixed'}"
    if plain_output:
        print(f"Generating ({mode_str})...")
    else:
        print(f"{Colors.INFO}Generating ({mode_str})...{Colors.RESET}")
    start = time.time()
    if args.mode == "ar":
        h, kvs = core(ids, causal_mask(ids.size(1), structured=use_structured_masks(args)), use_cache=True, total_seq_len=ids.size(1))
        for _ in range(args.max_new):
            logits = ar_h(h)[:, -1]
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            ids = torch.cat([ids, nxt], 1)
            h, kvs = core(ids[:, -1:], None, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
    elif args.mode == "nat":
        # Iterative mask-predict decode (CMLM): keep the prompt fixed and fill the
        # BLANK slots, committing the most-confident predictions each pass.
        n_fill = max(1, int(args.max_new))
        ids = torch.tensor([prompt_tokens + [BLANK] * n_fill], device=DEV)
        remaining = set(range(prompt_len, prompt_len + n_fill))
        passes = max(1, int(args.nat_passes))
        for p in range(passes):
            if not remaining:
                break
            h = core(ids, None)
            logits = nat_h(h)
            logits[..., BLANK] = -1e9
            conf, pred = logits.softmax(-1).max(-1)
            k = max(1, -(-len(remaining) // (passes - p)))
            for pos in sorted(remaining, key=lambda q: float(conf[0, q]), reverse=True)[:k]:
                ids[0, pos] = int(pred[0, pos])
                remaining.discard(pos)
        if remaining:
            h = core(ids, None)
            logits = nat_h(h)
            logits[..., BLANK] = -1e9
            pred = logits.argmax(-1)
            for pos in remaining:
                ids[0, pos] = int(pred[0, pos])
    else:
        cached_len = ids.size(1)
        h, kvs = core(ids, sat_mask(ids.size(1), structured=use_structured_masks(args)), use_cache=True, total_seq_len=cached_len)
        h_buffer = h[:, -SAT_BLOCK:]
        added = 0
        stop = False
        
        # Align to block boundary if prompt is off-boundary
        if ids.size(1) % SAT_BLOCK != 0:
            logits = ar_h(h)[:, -1]
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            ids = torch.cat([ids, nxt], 1)
            added += 1
            h, kvs = core(nxt, None, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
            cached_len = ids.size(1)
            h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
            
        while added < args.max_new:
            logits_all, gate = sat_h(h_buffer)
            stride = SAT_BLOCK if (not args.var or gate is None) else (gate.softmax(-1).multinomial(1).item() + 1)
            stride = min(int(stride), logits_all.size(1))
            new_tokens = []
            for i in range(int(stride)):
                logits = logits_all[:, i]
                logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
                nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt], 1)
                added += 1
                if added >= args.max_new: break
            if added >= args.max_new: break
            new_ids = torch.cat(new_tokens, dim=1)
            mask = sat_mask_cached(new_ids.size(1), cached_len, structured=use_structured_masks(args))
            h, kvs = core(new_ids, mask, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
            cached_len = ids.size(1)
            h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
    elapsed = time.time() - start
    gen_tokens = len(ids[0]) - prompt_len
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    all_tokens = ids[0].tolist()
    prompt_text = tok.decode(all_tokens[:prompt_len], skip_special_tokens=True)
    gen_text = tok.decode(all_tokens[prompt_len:], skip_special_tokens=True)
    safe_prompt = _ascii_safe(prompt_text) if plain_output else prompt_text
    safe_gen = _ascii_safe(gen_text) if plain_output else gen_text
    if plain_output:
        print(f"{safe_prompt}{safe_gen}")
        print(f"[{elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s]")
    else:
        print(f"{Colors.PROMPT}{safe_prompt}{Colors.RESET}{safe_gen}")
        print(f"{Colors.INFO}[{elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s]{Colors.RESET}")
    if getattr(args, "claude_friendly", False):
        claude_prompt = _ascii_safe(prompt_text)
        claude_gen = _ascii_safe(gen_text)
        print("[CLAUDE_FRIENDLY_START]")
        print(f"[mode={mode_str}]")
        print("[prompt_input]")
        print(claude_prompt)
        print("[completion]")
        print(claude_gen)
        print("[prompt_plus_completion]")
        print(f"{claude_prompt}{claude_gen}")
        print(f"[stats] {elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s")
        print("[CLAUDE_FRIENDLY_END]")


# ───────────────────────── CLI ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="AGILLM Expansion Ratio Testing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--preset", choices=PRESETS.keys(), default="nano_3x")
    tr.add_argument("--rank", type=int)
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    tr.add_argument("--source", default=DEFAULT_PRETRAIN_SOURCES)
    tr.add_argument("--target_tokens", type=int)
    tr.add_argument("--token_param_ratio", type=float, default=0.0,
                    help="If --target_tokens is omitted, train to this tokens:param ratio. AGILLM-4 presets default to 100.")
    tr.add_argument("--steps", type=int)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    tr.add_argument("--attn_backend", choices=["manual", "sdpa", "sublinear"], default=DEFAULT_ATTN_BACKEND,
                    help="AGILLM-4 attention backend. sublinear uses local-window plus landmark candidates.")
    tr.add_argument("--grad_checkpoint", action="store_true",
                    help="Recompute transformer blocks during backward to trade speed for longer context.")
    tr.add_argument("--sublinear_window", type=int, default=DEFAULT_SUBLINEAR_WINDOW,
                    help="For --attn_backend sublinear, attend to this many local tokens on each side.")
    tr.add_argument("--sublinear_stride", type=int, default=DEFAULT_SUBLINEAR_STRIDE,
                    help="For --attn_backend sublinear, use every Nth token as a landmark candidate.")
    tr.add_argument("--sublinear_max_anchors", type=int, default=DEFAULT_SUBLINEAR_MAX_ANCHORS,
                    help="For --attn_backend sublinear, cap landmark candidates per query chunk.")
    tr.add_argument("--sublinear_chunk", type=int, default=DEFAULT_SUBLINEAR_CHUNK,
                    help="For --attn_backend sublinear, query chunk size controlling peak gather memory.")
    tr.add_argument("--sublinear_sinks", type=int, default=DEFAULT_SUBLINEAR_SINKS,
                    help="For sublinear attention, always include this many first-token attention sinks.")
    tr.add_argument("--sublinear_recent_anchors", type=int, default=DEFAULT_SUBLINEAR_RECENT_ANCHORS,
                    help="For capped sublinear anchors, reserve this many anchors for the recent tail; -1 uses half.")
    tr.add_argument("--sublinear_pooled_landmarks", action=argparse.BooleanOptionalAction,
                    default=DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
                    help="Use stride-segment pooled K/V summaries for sublinear landmark anchors.")
    tr.add_argument("--no_structured_masks", action="store_true",
                    help="Disable structured causal/SAT masks for sublinear attention and fall back to dense masks.")
    tr.add_argument("--anchor_memory", action="store_true",
                    help="Enable anchor-memory long-context augmentation (one AnchorMemoryLayer at mid-stack).")
    tr.add_argument("--anchor_stride", type=int, default=DEFAULT_ANCHOR_STRIDE,
                    help="Token span compressed into one anchor (default 256).")
    tr.add_argument("--anchor_max", type=int, default=DEFAULT_ANCHOR_MAX,
                    help="Max anchors retained in the rolling memory bank.")
    tr.add_argument("--anchor_position", type=int, default=DEFAULT_ANCHOR_POSITION,
                    help="Block index after which to insert anchor memory (-1 = stack middle).")
    tr.add_argument("--kv_buffer", action="store_true",
                    help="Use preallocated KV buffer instead of torch.cat-based cache growth.")
    tr.add_argument("--optimizer", choices=["adamw", "adamw8bit", "paged_adamw8bit"], default="adamw",
                    help="Optimizer backend. 8-bit options reduce VRAM on 24GB production runs.")
    tr.add_argument("--save_every_sec", type=int, default=DEFAULT_SAVE_SEC)
    tr.add_argument("--heartbeat_every_sec", type=int, default=300,
                    help="Print lightweight trainer heartbeat/status lines every N seconds; 0 disables.")
    tr.add_argument("--empty_cache_every_steps", type=int, default=0,
                    help="Call torch.cuda.empty_cache() every N train steps; useful for VRAM-first runs where lower reserved VRAM matters more than speed.")
    tr.add_argument("--profile_steps", type=int, default=0,
                    help="Profile the first N DBlock training steps with in-process CUDA timers; 0 disables.")
    tr.add_argument("--profile_log_every", type=int, default=25,
                    help="Print averaged profiler timings every N profiled steps.")
    tr.add_argument("--delta_every_steps", type=int, default=DEFAULT_DELTA_STEPS, help="Weight-only delta save every N steps (0=off)")
    tr.add_argument("--delta_max_keep", type=int, default=DEFAULT_MAX_DELTAS, help="Max delta checkpoints to keep")
    tr.add_argument("--resume_delta", type=str, help="Resume from a delta (weight-only, no optimizer state)")
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str)
    tr.add_argument("--x2", action="store_true")
    tr.add_argument("--warmstart_from", type=str)
    tr.add_argument("--fresh", action="store_true")
    tr.add_argument("--max_ckpts", type=int, default=None)
    tr.add_argument("--chilla_max_double", action="store_true")
    tr.add_argument("--tie_weights", action="store_true")
    tr.add_argument("--ar_only", action="store_true")
    tr.add_argument("--sat_every", type=int, default=1,
                    help="Train SAT every N steps. Default 1 keeps AR+SAT every step.")
    tr.add_argument("--nat_every", type=int, default=1,
                    help="Train NAT every N steps with a CTC objective. Default 1 keeps AR+SAT+NAT every step.")
    tr.add_argument("--nat_loss_weight", type=float, default=1.0)
    tr.add_argument("--nat_expand", type=int, default=2,
                    help="Repeat tokens this many times for the NAT CTC input length.")
    tr.add_argument("--nat_max_tokens", type=int, default=0,
                    help="Optional cap for NAT target tokens per batch; 0 uses the whole block.")
    tr.add_argument("--nat_mask_ratio", type=float, default=0.5,
                    help="Fraction of positions masked to BLANK for the NAT mask-predict (CMLM) objective.")
    tr.add_argument("--dblock", action="store_true", help="DiffusionBlocks block-wise denoising training (low VRAM).")
    tr.add_argument("--dblock_blocks", type=int, default=4, help="Partition layers into this many DiffusionBlocks blocks.")
    tr.add_argument("--dblock_schedule", choices=["random", "roundrobin", "loss_balanced"], default="loss_balanced",
                    help="How --dblock chooses the next layer block. loss_balanced focuses blocks whose EMA loss is highest after warmup.")
    tr.add_argument("--dblock_warmup_steps", type=int, default=16,
                    help="Initial DBlock steps spent covering every block before loss-balanced scheduling.")
    tr.add_argument("--dblock_explore", type=float, default=0.05,
                    help="Exploration rate for loss-balanced DBlock scheduling.")
    tr.add_argument("--dblock_log_every", type=int, default=25,
                    help="Print DBlock block/loss/VRAM diagnostics every N DBlock steps; 0 disables.")
    tr.add_argument("--dblock_checkpoint_stride", type=int, default=1,
                    help="With --grad_checkpoint in --dblock mode, checkpoint one layer every N selected block layers; 1=all layers, 2=alternate, 0=off.")
    tr.add_argument("--dblock_checkpoint_skip_tail", type=int, default=0,
                    help="Experimental DBlock speed knob: do not checkpoint this many final layers in the selected block, reducing backward recompute at higher VRAM cost.")
    tr.add_argument("--dblock_activation_offload", action="store_true",
                    help="Experimental DBlock speed knob: for non-checkpointed block layers, offload saved backward tensors to CPU RAM instead of recomputing.")
    tr.add_argument("--dblock_activation_offload_min_mb", type=float, default=1.0,
                    help="Minimum CUDA tensor size in MB to offload under --dblock_activation_offload.")
    tr.add_argument("--dblock_sigma_curriculum_steps", type=int, default=2000,
                    help="Warm sigma ranges from easy to full span over this many DBlock steps; 0 disables.")
    tr.add_argument("--dblock_edm_wmax", type=float, default=5.0,
                    help="Cap for EDM loss weighting in DBlock mode.")
    tr.add_argument("--dblock_ar_weight", type=float, default=1.0)
    tr.add_argument("--dblock_sat_weight", type=float, default=1.0)
    tr.add_argument("--dblock_nat_weight", type=float, default=1.0)
    tr.add_argument("--dblock_objective_mode", choices=["periodic", "stochastic"], default="periodic",
                    help="DBlock objective scheduler. stochastic samples one objective per step to reduce redundant AR/SAT/NAT forwards.")
    tr.add_argument("--dblock_ar_prob", type=float, default=0.80, help="Stochastic DBlock probability for AR objective.")
    tr.add_argument("--dblock_sat_prob", type=float, default=0.10, help="Stochastic DBlock probability for SAT objective.")
    tr.add_argument("--dblock_nat_prob", type=float, default=0.10, help="Stochastic DBlock probability for NAT objective.")
    tr.add_argument("--dblock_ar_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many AR target positions per DBlock step for stochastic token-level CE.")
    tr.add_argument("--dblock_sat_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many SAT target positions per DBlock step.")
    tr.add_argument("--dblock_nat_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many NAT target positions per DBlock step.")
    tr.add_argument("--reinit_nat", action="store_true",
                    help="Reinitialize NAT head weights after load (use once when switching to mask-predict).")
    tr.add_argument("--seed_nat_from_ar", action="store_true",
                    help="Seed the NAT head from the trained AR head ('father') after load instead of random init.")
    tr.add_argument("--freeze_core", action="store_true")
    tr.add_argument("--unfreeze_ln", action="store_true")
    tr.add_argument("--train_emb", action="store_true")
    tr.add_argument("--lr_core", type=float, default=LR_CORE)
    tr.add_argument("--lr_head", type=float, default=LR_HEAD)
    tr.add_argument("--chat", action="store_true")
    tr.add_argument("--chat_messages_key", default="messages")
    tr.add_argument("--dataset_field_text", default="text")
    tr.add_argument("--sft_add_generation_prompt", action="store_true")
    tr.add_argument("--auto_grow", action="store_true")
    tr.add_argument("--grow_plan", default="576,640,768,896,1024,1122")
    tr.add_argument("--grow_every_steps", type=int, default=50000)
    tr.add_argument("--after_sft_source", default="")
    tr.add_argument("--after_sft_steps", type=int, default=0)
    tr.add_argument("--after_sft_chat", action="store_true")
    tr.add_argument("--after_sft_chat_messages_key", default="messages")
    tr.add_argument("--after_sft_dataset_field_text", default="text")
    tr.add_argument("--after_sft_add_generation_prompt", type=bool, default=None)
    tr.add_argument("--after_sft_block", type=int, default=0)
    tr.add_argument("--after_sft_freeze_core", action="store_true")
    tr.add_argument("--after_sft_unfreeze_ln", action="store_true")
    tr.add_argument("--after_sft_train_emb", action="store_true")
    tr.add_argument("--after_sft_lr_core", type=float, default=0.0)
    tr.add_argument("--after_sft_lr_head", type=float, default=0.0)
    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar", "sat", "nat"], required=True)
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--temperature", type=float, default=None)
    inf.add_argument("--greedy", action="store_true")
    inf.add_argument("--top_k", type=int, default=None)
    inf.add_argument("--top_p", type=float, default=0.9)
    inf.add_argument("--min_p", type=float, default=0.0)
    inf.add_argument("--repetition_penalty", type=float, default=None)
    inf.add_argument("--presence_penalty", type=float, default=None)
    inf.add_argument("--frequency_penalty", type=float, default=None)
    inf.add_argument("--penalty_last_n", type=int, default=None)
    inf.add_argument("--var", action="store_true", default=None)
    inf.add_argument("--no-var", dest="var", action="store_false")
    inf.add_argument("--claude-friendly", action="store_true", help="Also print an artifact-free prompt/completion block for downstream JSON consumers")
    inf.add_argument("--plain-output", "--no-color", dest="plain_output", action="store_true", help="Use plain ASCII/no ANSI output for redirected inference logs")
    inf.add_argument("--attn_backend", choices=["manual", "sdpa", "sublinear"], default=DEFAULT_ATTN_BACKEND)
    inf.add_argument("--sublinear_window", type=int, default=DEFAULT_SUBLINEAR_WINDOW)
    inf.add_argument("--sublinear_stride", type=int, default=DEFAULT_SUBLINEAR_STRIDE)
    inf.add_argument("--sublinear_max_anchors", type=int, default=DEFAULT_SUBLINEAR_MAX_ANCHORS)
    inf.add_argument("--sublinear_chunk", type=int, default=DEFAULT_SUBLINEAR_CHUNK)
    inf.add_argument("--sublinear_sinks", type=int, default=DEFAULT_SUBLINEAR_SINKS)
    inf.add_argument("--sublinear_recent_anchors", type=int, default=DEFAULT_SUBLINEAR_RECENT_ANCHORS)
    inf.add_argument("--sublinear_pooled_landmarks", action=argparse.BooleanOptionalAction,
                     default=DEFAULT_SUBLINEAR_POOLED_LANDMARKS)
    inf.add_argument("--no_structured_masks", action="store_true")
    inf.add_argument("--nat_expand", type=int, default=2)
    inf.add_argument("--nat_passes", type=int, default=1)
    st = sub.add_parser("status", help="Read-only training status")
    st.add_argument("--json", dest="json_output", action="store_true")
    st.add_argument("--log", type=str, default=str(STATUS_DEFAULT_LOG))
    st.add_argument("--save_dir", type=str, default=str(STATUS_DEFAULT_SAVE_DIR))
    args = ap.parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "infer": infer(args)
    else: raise SystemExit(_emit_status(Path(args.log), Path(args.save_dir), args.json_output))


if __name__ == "__main__":
    main()
