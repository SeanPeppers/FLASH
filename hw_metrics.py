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

from __future__ import annotations

import logging
import logging.handlers
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


# ── Logging setup ──────────────────────────────────────────────────────────────
# Set FLASH_LOG_SERVER=<chameleon_ip>:9020 on each device to stream all logs
# to the central log_server.py running on Chameleon. Falls back to local file only.

LOG_SERVER_PORT = 9020


class _DeviceFilter(logging.Filter):
    """Injects %(device)s into every log record so the aggregated log shows the source."""
    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def filter(self, record: logging.LogRecord) -> bool:
        record.device = self.device
        return True


def _setup_logging(device: str) -> logging.Logger:
    log_dir = Path("fl_results_hfl")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"flash_debug_{device}.log"

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(device)-14s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    device_filter = _DeviceFilter(device)

    root = logging.getLogger("flash")
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        fh.addFilter(device_filter)
        root.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        sh.addFilter(device_filter)
        root.addHandler(sh)

    # Optional: stream to central log server on Chameleon
    # Set FLASH_LOG_SERVER=<chameleon_ip>  (port defaults to 9020)
    log_server_env = os.environ.get("FLASH_LOG_SERVER", "").strip()
    if log_server_env:
        host, _, port_str = log_server_env.partition(":")
        port = int(port_str) if port_str else LOG_SERVER_PORT
        try:
            socket_handler = logging.handlers.SocketHandler(host, port)
            socket_handler.setLevel(logging.DEBUG)
            socket_handler.addFilter(device_filter)
            root.addHandler(socket_handler)
            root.info("Log streaming enabled -> %s:%d", host, port)
        except Exception as exc:
            root.warning("Could not attach log socket handler (%s:%d): %s", host, port, exc)

    root.info("=" * 60)
    root.info("FLASH logging started  device=%s  log=%s", device, log_path)
    root.info("=" * 60)
    return root


_log = _setup_logging(DEVICE)
_hw_log = logging.getLogger("flash.hw")


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


# tegrastats cache (separate: uses Popen+readline, no --count flag support)
_teg_cache: tuple = ("", 0.0)


def _tegrastats_once() -> str:
    """Start tegrastats, read one output line, kill it. Cached for _CMD_TTL seconds."""
    global _teg_cache
    val, ts = _teg_cache
    if time.monotonic() - ts < _CMD_TTL:
        return val
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        val = proc.stdout.readline().decode("utf-8", errors="ignore").strip()
        proc.kill()
        proc.wait(timeout=2)
    except Exception:
        val = ""
    _teg_cache = (val, time.monotonic())
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
# CPU utilization power model constants — calibrate these offline with a USB
# power meter: measure wall power at idle and at 100% CPU (e.g. `stress -c 4`).
# Current defaults are typical Pi 5 values; update before final paper run.
_PI5_IDLE_MW = 3200.0   # ~3.2 W at idle
_PI5_MAX_MW  = 7500.0   # ~7.5 W at 100% CPU load


class _Pi5Collector:
    def __init__(self):
        self._logged = False

    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # CPU frequencies from sysfs (no subprocess)
        freqs = []
        for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
            f = _read(str(cpu / "cpufreq/scaling_cur_freq"), 0.0)
            if f > 0:
                freqs.append(f / 1000.0)  # kHz -> MHz
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
        hwmon_total_mw = 0.0
        for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
            name_file = sensor / "name"
            if not name_file.exists():
                continue
            name = name_file.read_text().strip()
            for pfile in sensor.glob("power*_input"):
                rid = re.search(r"\d+", pfile.name)
                val_mw = _read(str(pfile), 0.0) / 1000.0  # µW -> mW
                key = f"hwmon_{name}_power{rid.group() if rid else ''}_mw"
                m[key] = val_mw
                hwmon_total_mw += val_mw
        if hwmon_total_mw > 0:
            m["power_total_soc_mw"] = hwmon_total_mw
        else:
            # No hardware power sensor — use CPU utilization linear power model.
            # P = P_idle + (P_max - P_idle) * cpu_fraction
            # Calibrate _PI5_IDLE_MW / _PI5_MAX_MW offline before paper runs.
            cpu_frac = m.get("cpu_util_pct", 0.0) / 100.0
            model_mw = _PI5_IDLE_MW + (_PI5_MAX_MW - _PI5_IDLE_MW) * cpu_frac
            m["power_model_cpu_mw"] = model_mw
            m["power_total_soc_mw"] = model_mw

        if not self._logged:
            self._logged = True
            rail_details = {k: v for k, v in m.items() if k.startswith("hwmon_") and k.endswith("_mw")}
            if rail_details:
                detail_str = "  ".join(f"{k}={v:.1f}mW" for k, v in sorted(rail_details.items()))
                _hw_log.info("[Pi5] hwmon rails found: %s -> power_total_soc_mw=%.1f mW",
                             detail_str, m.get("power_total_soc_mw", 0.0))
            else:
                _hw_log.info("[Pi5] No hwmon power sensor — using CPU utilization model "
                             "(idle=%.0f mW, max=%.0f mW)", _PI5_IDLE_MW, _PI5_MAX_MW)

        return m


# ── Jetson collector (Nano + Xavier) ──────────────────────────────────────────
class _JetsonCollector:
    def __init__(self):
        self._logged = False

    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # Tegra INA3221 power rails — try i2c driver paths first, then hwmon
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
        if rail_total == 0:
            # Xavier NX/AGX/Nano expose INA3221 via hwmon on newer kernels.
            # The ina3221 hwmon driver uses curr{N}_input (mA) + in{N}_input (mV),
            # NOT power*_input files — power must be computed as I * V.
            for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
                name_file = sensor / "name"
                if not name_file.exists():
                    continue
                if "ina3221" not in name_file.read_text().strip().lower():
                    continue
                channel_powers: Dict[int, float] = {}
                main_rail_ch: Optional[int] = None
                for ch in range(1, 5):
                    curr_file = sensor / f"curr{ch}_input"
                    volt_file = sensor / f"in{ch}_input"
                    if not curr_file.exists() or not volt_file.exists():
                        continue
                    curr_ma = _read(str(curr_file), 0.0)
                    volt_mv = _read(str(volt_file), 0.0)
                    if curr_ma <= 0 or volt_mv <= 0:
                        continue
                    power_mw = curr_ma * volt_mv / 1000.0
                    channel_powers[ch] = power_mw
                    label_file = sensor / f"in{ch}_label"
                    if label_file.exists():
                        label = label_file.read_text().strip().upper()
                        if any(x in label for x in ("VDD_IN", "5V_IN", "TOTAL", "POM_5V_IN")):
                            main_rail_ch = ch
                if channel_powers:
                    if main_rail_ch and main_rail_ch in channel_powers:
                        rail_total = channel_powers[main_rail_ch]
                    else:
                        # Channel 1 = main board input on all known Jetson INA3221 configs
                        rail_total = channel_powers.get(1, max(channel_powers.values()))
                    for ch, pw in channel_powers.items():
                        m[f"power_hwmon_ch{ch}_mw"] = pw
        if rail_total > 0:
            m["power_total_soc_mw"] = rail_total

        # tegrastats — INA3221 power rails via NVIDIA's driver (bypasses hwmon).
        # Parsed here, before the logging block, so power keys are visible to it.
        # Format: ... VDD_IN 5000mW/5000mW VDD_CPU_GPU_CV 1234mW/1234mW ...
        teg = _tegrastats_once()
        if teg and "power_total_soc_mw" not in m:
            teg_total = 0.0
            for rail, cur_mw in re.findall(r"(\w+)\s+(\d+)mW/\d+mW", teg):
                m[f"power_teg_{rail.lower()}_mw"] = float(cur_mw)
                if rail == "VDD_IN":
                    teg_total = float(cur_mw)
            if teg_total == 0.0:
                teg_total = sum(v for k, v in m.items() if k.startswith("power_teg_"))
            if teg_total > 0:
                m["power_total_soc_mw"] = teg_total

        if not self._logged:
            self._logged = True
            rail_keys = {k: v for k, v in m.items() if k.startswith("power_") and k.endswith("_mw")}
            if rail_keys:
                detail_str = "  ".join(f"{k}={v:.1f}mW" for k, v in sorted(rail_keys.items()))
                sources = set()
                if any("teg_" in k for k in rail_keys): sources.add("tegrastats")
                if any("hwmon" in k for k in rail_keys): sources.add("hwmon")
                if not sources: sources.add("i2c")
                _hw_log.info("[Jetson] INA3221 via %s: power_total_soc_mw=%.1f mW",
                             "+".join(sources), m.get("power_total_soc_mw", 0.0))
            else:
                _hw_log.warning("[Jetson] NO INA3221 rails found (tried i2c + hwmon + tegrastats) — energy will read 0.0 J")

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

        # DLA util (Xavier only) — reuse cached tegrastats output from above
        if DEVICE == "jetson_xavier" and teg:
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

    def __init__(self):
        # State for RAPL delta-based instantaneous power estimation.
        # RAPL exposes cumulative energy counters (uJ); power = delta / dt.
        self._prev_rapl_uj: Dict[str, float] = {}
        self._prev_ts: float = 0.0
        self._logged = False
        # Pre-warm: seed RAPL baseline so the first real poll has a valid delta.
        self.snapshot()

    def snapshot(self) -> Dict[str, float]:
        m = _psutil_common()

        # RAPL energy counters + instantaneous power via delta
        if self._RAPL.exists():
            now = time.monotonic()
            cur_uj: Dict[str, float] = {}
            for ef in self._RAPL.rglob("energy_uj"):
                parts = [p for p in ef.parts if "intel-rapl" in p]
                safe = re.sub(r"[^a-zA-Z0-9_]", "_", "_".join(parts))
                val = _read(str(ef), 0.0)
                cur_uj[safe] = val
                m[f"rapl_{safe}_energy_mj"] = val / 1000.0

            dt = now - self._prev_ts
            if self._prev_rapl_uj and dt > 0:
                # Sum only top-level package domains (2 "intel-rapl" path segments)
                # to avoid double-counting sub-domains (core, uncore).
                # Package keys: "intel_rapl_intel_rapl_0"  (count == 2)
                # Sub-domain:   "intel_rapl_intel_rapl_0_intel_rapl_0_0" (count == 3)
                total_delta_uj = 0.0
                for key, val in cur_uj.items():
                    if key.count("intel_rapl_") == 2:
                        prev = self._prev_rapl_uj.get(key, val)
                        delta_uj = val - prev
                        if delta_uj < 0:
                            delta_uj += 262144.0 * 1e6  # counter wrap (~262 kJ max)
                        total_delta_uj += delta_uj
                if total_delta_uj > 0:
                    m["rapl_power_mw"] = (total_delta_uj / 1e3) / dt  # uJ/s -> mW

            self._prev_rapl_uj = cur_uj
            self._prev_ts = now

        # IPMI -- try plain then sudo (Chameleon often requires elevated perms)
        ipmi_out = _run_cached("ipmitool dcmi power reading")
        if not ipmi_out:
            ipmi_out = _run_cached("sudo ipmitool dcmi power reading")
        ipmi_m = re.search(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", ipmi_out)
        if ipmi_m:
            m["ipmi_system_power_w"] = float(ipmi_m.group(1))

        if not self._logged:
            self._logged = True
            rapl_files = list(self._RAPL.rglob("energy_uj")) if self._RAPL.exists() else []
            rapl_vals = {str(f): _read(str(f), -1.0) for f in rapl_files}
            readable = {k: v for k, v in rapl_vals.items() if v >= 0}
            unreadable = [k for k, v in rapl_vals.items() if v < 0]
            _hw_log.info("[Chameleon] RAPL path exists=%s  files_found=%d  readable=%d  unreadable=%d",
                         self._RAPL.exists(), len(rapl_files), len(readable), len(unreadable))
            if unreadable:
                _hw_log.warning("[Chameleon] RAPL files permission-denied (run with sudo?): %s",
                                unreadable[:3])
            if readable:
                sample = list(readable.items())[:2]
                _hw_log.info("[Chameleon] RAPL sample values: %s",
                             "  ".join(f"{Path(k).parent.name}={v:.0f}uJ" for k, v in sample))
            else:
                _hw_log.warning("[Chameleon] RAPL returned all zeros/unreadable — will fall back to GPU power")
            if "ipmi_system_power_w" in m:
                _hw_log.info("[Chameleon] IPMI available: %.1f W", m["ipmi_system_power_w"])
            else:
                _hw_log.warning("[Chameleon] IPMI unavailable — energy source will be RAPL or GPU NVML")
            gpu_keys = [k for k in m if k.endswith("_power_mw")]
            if gpu_keys:
                gpu_str = "  ".join(f"{k}={m[k]:.0f}mW" for k in gpu_keys)
                _hw_log.info("[Chameleon] GPU NVML power: %s", gpu_str)
            else:
                _hw_log.warning("[Chameleon] No GPU NVML power keys found")

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
                gpu_total_mw = 0.0
                for idx in range(n):
                    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode()
                    sn = re.sub(r"[^a-zA-Z0-9]", "_", name).lower()[:20]
                    pw = float(pynvml.nvmlDeviceGetPowerUsage(h))
                    m[f"gpu{idx}_{sn}_util_pct"] = float(util.gpu)
                    m[f"gpu{idx}_{sn}_mem_used_mb"] = mem.used / 1e6
                    m[f"gpu{idx}_{sn}_power_mw"] = pw
                    m[f"gpu{idx}_{sn}_temp_celsius"] = float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    )
                    gpu_total_mw += pw
                # Explicit key so EnergyAccumulator's gpu_power_nvml_mw check fires.
                m["gpu_power_nvml_mw"] = gpu_total_mw
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
        self._last_ts: float = 0.0
        self._last_power_mw: float = 0.0
        self._poll_count: int = 0
        self._log_ts: float = 0.0  # time of last periodic log line
        self._log = logging.getLogger("flash.energy")

    def start(self):
        self._energy_mj = 0.0
        self._last_ts = time.monotonic()
        self._last_power_mw = 0.0
        self._poll_count = 0
        self._log_ts = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        self._last_ts = time.monotonic()
        self._last_power_mw = 0.0
        while self._running:
            m = snapshot()
            now = time.monotonic()
            elapsed = now - self._last_ts
            self._last_ts = now

            if m.get("power_total_soc_mw") is not None:
                source = "power_total_soc_mw"
                power_mw = m["power_total_soc_mw"]
            elif "ipmi_system_power_w" in m:
                source = "ipmi_system_power_w"
                power_mw = m["ipmi_system_power_w"] * 1000.0
            elif "rapl_power_mw" in m:
                source = "rapl_power_mw"
                power_mw = m["rapl_power_mw"]
            elif m.get("gpu_power_nvml_mw") is not None:
                source = "gpu_power_nvml_mw"
                power_mw = m["gpu_power_nvml_mw"]
            else:
                gpu_powers = [v for k, v in m.items() if k.endswith("_power_mw") and v > 0]
                power_mw = sum(gpu_powers) if gpu_powers else 0.0
                source = "gpu_dynamic" if power_mw > 0 else "NONE"

            with self._lock:
                self._energy_mj += float(power_mw) * elapsed
                self._last_power_mw = float(power_mw)
                self._poll_count += 1
                poll_n = self._poll_count
                energy_so_far = self._energy_mj

            # Log on first poll, then every 30 s
            if poll_n == 1:
                if power_mw == 0.0:
                    self._log.warning(
                        "poll#1  source=%s  power=0.0 mW — energy will accumulate as 0 J  elapsed=%.3fs",
                        source, elapsed,
                    )
                else:
                    self._log.info(
                        "poll#1  source=%s  power=%.1f mW  elapsed=%.3fs",
                        source, power_mw, elapsed,
                    )
            elif now - self._log_ts >= 30.0:
                self._log_ts = now
                self._log.debug(
                    "poll#%d  source=%s  power=%.1f mW  energy_so_far=%.3f J",
                    poll_n, source, power_mw, energy_so_far / 1000.0,
                )

            time.sleep(self.interval)

    def stop_and_get_joules(self) -> float:
        self._running = False
        stop_ts = time.monotonic()
        if self._thread:
            self._thread.join(timeout=3)
        # Account for time elapsed since the last poll (covers short rounds where
        # the background thread may not have fired again before stop was called).
        with self._lock:
            tail_elapsed = stop_ts - getattr(self, "_last_ts", stop_ts)
            if tail_elapsed > 0:
                self._energy_mj += self._last_power_mw * tail_elapsed
            joules = self._energy_mj / 1000.0

        self._log.info(
            "stop  polls=%d  last_power=%.1f mW  tail_elapsed=%.3fs  total=%.4f J",
            self._poll_count, self._last_power_mw, tail_elapsed, joules,
        )
        return joules


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    for k, v in sorted(snapshot().items()):
        print(f"  {k}: {v:.4f}")
