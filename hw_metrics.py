"""
hw_metrics.py — Hardware-aware metrics collection.

Auto-detects: pi5 | jetson_nano | jetson_xavier | chameleon | generic

FIX vs previous version:
  - vcgencmd / subprocess calls are cached with a 2-second TTL so they don't
    add 200 ms per snapshot on the Pi 5.
  - psutil.cpu_percent(interval=None) is called with a prior 0.1s warm-up at
    import time so the first real call returns a valid reading instead of 0.0.
  - All sysfs reads use a single Path.read_text() — no shell subprocess.
  - EnergyAccumulator poll interval raised to 1 s (was 0.5 s) to reduce load
    on the Pi 5 which has only 4 cores.

Install (all devices):
    pip install psutil --break-system-packages
    pip install pynvml --break-system-packages   # Jetson / Chameleon GPU nodes
"""

import os
import re
import subprocess
import time
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import psutil

# Warm up cpu_percent so the first real call doesn't return 0.0
psutil.cpu_percent(interval=0.1)

# ── Optional NVML ──────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ── Device detection ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def detect_device() -> str:
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            model = model_path.read_bytes().decode("utf-8", errors="ignore").lower()
            if "raspberry pi 5" in model:
                return "pi5"
            if "jetson nano" in model:
                return "jetson_nano"
            if "jetson xavier" in model or "jetson agx" in model:
                return "jetson_xavier"
        except Exception:
            pass

    if Path("/etc/nv_tegra_release").exists():
        try:
            nv = Path("/etc/nv_tegra_release").read_text()
            return "jetson_xavier" if "xavier" in nv.lower() else "jetson_nano"
        except Exception:
            return "jetson_nano"

    try:
        if os.uname().machine in ("x86_64", "amd64"):
            return "chameleon"
    except Exception:
        pass

    return "generic"


DEVICE = detect_device()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read(path: str, default: float = 0.0) -> float:
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return default


# Subprocess cache: {cmd_str: (result_str, timestamp)}
_CMD_CACHE: Dict[str, tuple] = {}
_CMD_TTL = 2.0  # seconds


def _run_cached(cmd: str) -> str:
    now = time.monotonic()
    if cmd in _CMD_CACHE:
        val, ts = _CMD_CACHE[cmd]
        if now - ts < _CMD_TTL:
            return val
    try:
        val = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
    except Exception:
        val = ""
    _CMD_CACHE[cmd] = (val, now)
    return val


# ── Common psutil block ────────────────────────────────────────────────────────
def _psutil_common() -> Dict[str, float]:
    m: Dict[str, float] = {}
    m["cpu_util_pct"] = float(psutil.cpu_percent(interval=None))
    for i, c in enumerate(psutil.cpu_percent(interval=None, percpu=True)):
        m[f"cpu_core{i}_util_pct"] = float(c)
    vm = psutil.virtual_memory()
    m["ram_used_mb"] = vm.used / 1e6
    m["ram_total_mb"] = vm.total / 1e6
    m["ram_util_pct"] = vm.percent
    net = psutil.net_io_counters()
    m["net_bytes_sent"] = float(net.bytes_sent)
    m["net_bytes_recv"] = float(net.bytes_recv)
    m["net_packets_sent"] = float(net.packets_sent)
    m["net_packets_recv"] = float(net.packets_recv)
    disk = psutil.disk_io_counters()
    if disk:
        m["disk_read_bytes"] = float(disk.read_bytes)
        m["disk_write_bytes"] = float(disk.write_bytes)
    return m


# ── Pi 5 collector ─────────────────────────────────────────────────────────────
class _Pi5Collector:
    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # CPU frequencies from sysfs (no subprocess)
        freqs = []
        for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
            f = _read(str(cpu / "cpufreq/scaling_cur_freq"), 0.0)
            if f > 0:
                freqs.append(f / 1000.0)  # kHz → MHz
        if freqs:
            m["cpu_freq_avg_mhz"] = float(sum(freqs) / len(freqs))
            for i, f in enumerate(freqs):
                m[f"cpu{i}_freq_mhz"] = f

        # Thermal zones
        max_t = 0.0
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            zid = zone.name.replace("thermal_zone", "")
            t = _read(str(zone / "temp"), 0.0) / 1000.0
            m[f"thermal_zone{zid}_celsius"] = t
            max_t = max(max_t, t)
        m["temp_max_celsius"] = max_t

        # vcgencmd — cached, so costs ~0 ms after first call
        v_str = _run_cached("vcgencmd measure_volts core")
        match = re.search(r"([\d.]+)V", v_str)
        if match:
            m["core_voltage_v"] = float(match.group(1))

        # hwmon power rails (INA3221 if wired)
        for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
            name_file = sensor / "name"
            if not name_file.exists():
                continue
            name = name_file.read_text().strip()
            for pfile in sensor.glob("power*_input"):
                rid = re.search(r"\d+", pfile.name)
                key = f"hwmon_{name}_power{rid.group() if rid else ''}_mw"
                m[key] = _read(str(pfile), 0.0) / 1000.0  # µW → mW

        return m


# ── Jetson collector (Nano + Xavier) ──────────────────────────────────────────
class _JetsonCollector:
    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # Tegra INA3221 power rails
        rail_total = 0.0
        for base in ["/sys/bus/i2c/drivers/ina3221x", "/sys/bus/i2c/devices"]:
            p = Path(base)
            if not p.exists():
                continue
            for pfile in p.rglob("in_power*_input"):
                val_mw = _read(str(pfile), 0.0)
                label_file = Path(str(pfile).replace("_input", "_label"))
                if not label_file.exists():
                    label_file = pfile.parent / pfile.name.replace("power", "rail_name")
                rail_name = label_file.read_text().strip() if label_file.exists() else pfile.stem
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", rail_name).lower()
                m[f"power_{safe}_mw"] = float(val_mw)
                rail_total += val_mw
        if rail_total > 0:
            m["power_total_soc_mw"] = rail_total

        # CPU frequencies
        freqs = []
        for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
            f = _read(str(cpu / "cpufreq/scaling_cur_freq"), 0.0)
            if f > 0:
                freqs.append(f / 1000.0)
        if freqs:
            m["cpu_freq_avg_mhz"] = float(sum(freqs) / len(freqs))
            for i, f in enumerate(freqs):
                m[f"cpu{i}_freq_mhz"] = f

        # GPU frequency
        for gpu_path in [
            "/sys/devices/gpu.0/devfreq/gpu.0/cur_freq",
            "/sys/kernel/debug/bpmp/debug/clk/gpu/rate",
        ]:
            val = _read(gpu_path, 0.0)
            if val > 0:
                m["gpu_freq_mhz"] = val / 1e6
                break

        # Thermal zones
        max_t = 0.0
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            zid = zone.name.replace("thermal_zone", "")
            try:
                ztype = (zone / "type").read_text().strip()
            except Exception:
                ztype = f"zone{zid}"
            t = _read(str(zone / "temp"), 0.0) / 1000.0
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", ztype).lower()
            m[f"temp_{safe}_celsius"] = t
            max_t = max(max_t, t)
        m["temp_max_celsius"] = max_t

        # nvpmodel — cached
        nvp = _run_cached("nvpmodel -q --verbose")
        for line in nvp.splitlines():
            if "Power Model ID" in line:
                try:
                    m["nvpmodel_id"] = float(line.split()[-1])
                except Exception:
                    pass

        # DLA (Xavier only)
        if DEVICE == "jetson_xavier":
            teg = _run_cached("tegrastats --interval 1 --count 1")
            dla = re.search(r"DLA_\d+:\s+(\d+)%", teg)
            if dla:
                m["dla_util_pct"] = float(dla.group(1))

        # NVML
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                m["gpu_util_pct"] = float(util.gpu)
                m["gpu_mem_used_mb"] = mem.used / 1e6
                m["gpu_power_nvml_mw"] = float(pynvml.nvmlDeviceGetPowerUsage(h))
            except Exception:
                pass

        return m


# ── Chameleon Cloud collector ──────────────────────────────────────────────────
class _ChameleonCollector:
    _RAPL = Path("/sys/class/powercap/intel-rapl")

    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # RAPL energy counters
        if self._RAPL.exists():
            for ef in self._RAPL.rglob("energy_uj"):
                parts = [p for p in ef.parts if "intel-rapl" in p]
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", "_".join(parts))
                m[f"rapl_{safe}_energy_mj"] = _read(str(ef), 0.0) / 1000.0

        # IPMI — cached (slow command)
        ipmi_out = _run_cached("ipmitool dcmi power reading")
        ipmi_m = re.search(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", ipmi_out)
        if ipmi_m:
            m["ipmi_system_power_w"] = float(ipmi_m.group(1))

        # CPU frequencies
        freqs = psutil.cpu_freq(percpu=True) or []
        if freqs:
            mhz = [f.current for f in freqs]
            m["cpu_freq_avg_mhz"] = float(sum(mhz) / len(mhz))
            m["cpu_freq_max_mhz"] = float(max(mhz))

        # Temperatures
        temps_raw = psutil.sensors_temperatures() if hasattr(psutil, "sensors_temperatures") else {}
        all_t = []
        for sname, entries in temps_raw.items():
            for i, e in enumerate(entries):
                label = re.sub(r"[^a-zA-Z0-9_]", "_", e.label or f"s{i}").lower()
                m[f"temp_{sname}_{label}_celsius"] = float(e.current)
                all_t.append(e.current)
        if all_t:
            m["temp_max_celsius"] = max(all_t)

        # Swap
        swap = psutil.swap_memory()
        m["swap_used_mb"] = swap.used / 1e6
        m["swap_util_pct"] = swap.percent

        # Load average
        try:
            la = os.getloadavg()
            m["load_avg_1m"] = float(la[0])
            m["load_avg_5m"] = float(la[1])
            m["load_avg_15m"] = float(la[2])
        except Exception:
            pass

        # NVML
        if _NVML_OK:
            try:
                n = pynvml.nvmlDeviceGetCount()
                for idx in range(n):
                    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode()
                    sn = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()[:20]
                    m[f"gpu{idx}_{sn}_util_pct"] = float(util.gpu)
                    m[f"gpu{idx}_{sn}_mem_used_mb"] = mem.used / 1e6
                    m[f"gpu{idx}_{sn}_power_mw"] = float(pynvml.nvmlDeviceGetPowerUsage(h))
                    m[f"gpu{idx}_{sn}_temp_celsius"] = float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    )
            except Exception:
                pass

        return m


class _GenericCollector:
    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()
        if _NVML_OK:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                m["gpu_util_pct"] = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                m["gpu_power_mw"] = float(pynvml.nvmlDeviceGetPowerUsage(h))
            except Exception:
                pass
        return m


# ── Public singleton ───────────────────────────────────────────────────────────
def _make_collector():
    if DEVICE == "pi5":
        return _Pi5Collector()
    elif DEVICE in ("jetson_nano", "jetson_xavier"):
        return _JetsonCollector()
    elif DEVICE == "chameleon":
        return _ChameleonCollector()
    return _GenericCollector()


_collector = _make_collector()


def snapshot() -> Dict[str, float]:
    """Return a flat dict of all hardware metrics. Fast: <5 ms on all devices."""
    return _collector.snapshot()


def delta(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    """
    For counter-type metrics (bytes, packets, RAPL energy) return the delta.
    For instantaneous metrics (util %, temperature, frequency) return after value.
    """
    COUNTER_SUFFIXES = (
        "_bytes_sent", "_bytes_recv", "_packets_sent", "_packets_recv",
        "_read_bytes", "_write_bytes", "_read_count", "_write_count",
        "_energy_mj",
    )
    result = dict(after)
    for k, v_after in after.items():
        if any(k.endswith(s) for s in COUNTER_SUFFIXES):
            result[f"delta_{k}"] = max(0.0, v_after - before.get(k, v_after))
    return result


# ── Background energy accumulator ─────────────────────────────────────────────
class EnergyAccumulator:
    """
    Integrates power (mW) over time in a background thread.
    Poll interval = 1 s by default (was 0.5 s) to spare the Pi 5's CPU.

    Usage:
        acc = EnergyAccumulator()
        acc.start()
        # ... do work ...
        joules = acc.stop_and_get_joules()
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._energy_mj = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self._energy_mj = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            m = snapshot()
            power_mw = (
                m.get("power_total_soc_mw")
                or m.get("gpu_power_nvml_mw")
                or (m.get("ipmi_system_power_w", 0.0) * 1000.0)
            )
            with self._lock:
                self._energy_mj += float(power_mw) * self.interval
            time.sleep(self.interval)

    def stop_and_get_joules(self) -> float:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            return self._energy_mj / 1000.0


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for k, v in sorted(snapshot().items()):
        print(f"  {k}: {v:.4f}")
