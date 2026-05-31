"""GPU telemetry. Prefers `nvidia-smi`; falls back to torch for memory only."""
import shutil
import subprocess
from typing import Any

_FIELDS = ["index", "name", "utilization.gpu", "memory.used", "memory.total",
           "temperature.gpu", "driver_version"]


def gpu_info() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, f"--query-gpu={','.join(_FIELDS)}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            gpus = []
            for line in out.splitlines():
                vals = [v.strip() for v in line.split(",")]
                row = dict(zip(_FIELDS, vals))
                gpus.append({
                    "index": int(row["index"]),
                    "name": row["name"],
                    "utilization_pct": _num(row["utilization.gpu"]),
                    "memory_used_mb": _num(row["memory.used"]),
                    "memory_total_mb": _num(row["memory.total"]),
                    "temperature_c": _num(row["temperature.gpu"]),
                    "driver_version": row["driver_version"],
                })
            return {"source": "nvidia-smi", "gpus": gpus}
        except Exception as e:
            return {"source": "nvidia-smi", "error": str(e), "gpus": _torch_fallback()}
    return {"source": "torch", "gpus": _torch_fallback()}


def _torch_fallback() -> list[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "utilization_pct": None,
                "memory_used_mb": round((total - free) / 1024 / 1024),
                "memory_total_mb": round(total / 1024 / 1024),
                "temperature_c": None,
                "driver_version": None,
            })
        return gpus
    except Exception:
        return []


def _num(v: str):
    try:
        return float(v) if "." in v else int(v)
    except (ValueError, TypeError):
        return None
