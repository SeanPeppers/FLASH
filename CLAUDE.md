# FLASH — Federated Learning Architecture & System Reference

## Project Overview

FLASH is a hierarchical federated learning (HFL) research project studying energy efficiency across a three-tier edge-to-cloud topology. The goal is an energy efficiency paper comparing FLASH (top-k delta compression + latency-aware scheduling) against FLARE (fixed compression) and FedAvg (baseline).

**Topology:**
```
Chameleon Cloud (GPU workstation)     ← server.py
        ↕ gRPC (Flower)
Jetson Xavier NX/AGX (mid-tier)       ← aggregator.py
        ↕ gRPC (inner Flower server)
Raspberry Pi 5 / Jetson Nano (leaves) ← clients.py
```

Hardware telemetry for all tiers is in `hw_metrics.py`.

---

## File Map

| File | Role | Lines |
|------|------|-------|
| `server.py` | Global coordinator; runs on Chameleon Cloud | 1,418 |
| `aggregator.py` | Mid-tier aggregator; runs on Jetson Xavier | 1,423 |
| `clients.py` | Leaf clients; runs on Pi 5 / Jetson Nano | 1,579 |
| `hw_metrics.py` | Hardware-aware metrics collection for all tiers | 1,440 |
| `log_server.py` | Central log aggregator; run on Chameleon before experiment | — |
| `verify_flash.py` | Local pipeline verification (compression + aggregation + training) | — |
| `fl_results_hfl/` | Output directory for all CSV results and per-device debug logs | — |

---

## Key Classes & Locations

### server.py
- `RoundHWLogger` (lines 85–142): Wraps strategy to capture server-side hardware per round. Writes `flash_server_hw.csv` at line 140.
- `FLASHGlobalStrategy` (lines 146–180): Latency threshold-based compression scheduling.
- `FLAREGlobalStrategy` (lines 184–191): Fixed compression ratio.
- `_global_agg` (lines 51–81): Weighted aggregation of client metrics.
- `save_csvs` (lines 195–229): Writes all four result CSVs.

### aggregator.py
- `_InnerStrategy` (lines 107–241): Runs the inner Flower server. Handles per-client compression ratio injection and cost model calibration.
  - `_pick_r` (lines 177–193): Selects compression ratio based on latency vs threshold.
  - Cost model calibration (lines 211–231): Updates `k_comp`, `k_comm` using a 3-round rolling window.
- `AggregatorClient` (lines 245–348): Connects upstream to global server; orchestrates inner leaf training.

### clients.py
- `FLASHClient` (lines 393–443): Top-k delta compression + energy accounting.
- `FLAREClient` (lines 447–482): Fixed compression, no delta encoding.
- `FedAvgClient` (lines 486–517): Vanilla FedAvg baseline.
- `train_one_epoch` (lines 205–256): Per-epoch training; returns loss, accuracy, gradient norms, batch times.
- `build_metrics` (lines 278–353): Assembles per-round metric dict sent to server.
- `compress_topk` / `decompress_topk` (lines 101–147): Sparse delta compression pipeline.

### hw_metrics.py
- `_Pi5Collector` (lines 132–173): vcgencmd + INA3221 hwmon.
- `_JetsonCollector` (lines 177–262): Tegra INA3221 + NVML + DLA.
- `_ChameleonCollector` (lines 266–338): Intel RAPL + IPMI + NVML.
- `_GenericCollector` (lines 341–351): psutil + NVML fallback.
- `EnergyAccumulator` (lines 391–433): Polls every 1 s, integrates `power_mw * real_elapsed` → joules. Tail-window added in `stop_and_get_joules` to capture time since last poll (critical for server-side rounds < 1 s).
  - Power source priority: `power_total_soc_mw` → `ipmi_system_power_w` → `rapl_power_mw` → `gpu_power_nvml_mw` → dynamic `_power_mw` key
- `_setup_logging` / `_DeviceFilter`: auto-configures on import → `fl_results_hfl/flash_debug_{device}.log`. Set `FLASH_LOG_SERVER=<ip>` to also stream to `log_server.py` on Chameleon.

---

## Hardcoded Hyperparameters (no config file)

| Parameter | Value | File | Line |
|-----------|-------|------|------|
| `NUM_ROUNDS` | 60 | server.py | 44 |
| `LATENCY_THRESHOLD` | 0.5 s | server.py:47, aggregator.py:59 | — |
| `ROLLING_WINDOW` | 5 | server.py | 46 |
| `NUM_AGGREGATORS` | 1 | server.py | 47 |
| `NUM_LEAF_CLIENTS` | 2 | aggregator.py | 58 |
| `TARGET_TAU` | 5.0 | clients.py | 53 |
| `MAX_LOCAL_EPOCHS` | 2 | clients.py | 59 |
| `BATCH_SIZE` | 64 | clients.py | 62 |
| `COMPRESSION_OPTIONS` | {1.0, 0.75, 0.5, 0.25} | clients.py | 54 |
| `k_comp` (cost model init) | 0.1 | aggregator.py | 124 |
| `k_comm` (cost model init) | 0.5 | aggregator.py | 125 |
| `_window` (cost model) | 3 | aggregator.py | 128 |

---

## Output CSVs

| File | Written by | Content |
|------|-----------|---------|
| `flash_HFL_eval_loss.csv` | `server.py:199–205` | Per-round eval loss (60 rows) |
| `flash_HFL_eval_metrics.csv` | `server.py:220–229` | Per-round eval accuracy etc. |
| `flash_HFL_fit_metrics.csv` | `server.py:208–217` | Per-round client fit metrics (400+ cols) |
| `flash_server_hw.csv` | `server.py:140–141` via `RoundHWLogger.save()` | Server hardware per round |

Metric prefixes: leaf metrics → `leaf_`, aggregator metrics → `agg_`.

---

## Known Bugs & Active Work

### OPEN: RAPL reads 0 on Chameleon (minor — GPU NVML is active fallback)
- RAPL sysfs files likely permission-denied without sudo. Energy is captured via GPU NVML (~53 W idle P100s).
- CPU+memory power not included. Acceptable for paper; note in methodology.
- To fix: `sudo python server.py` or `sudo setcap cap_sys_rawio+ep /usr/bin/python3` on Chameleon.

### FIXED (2026-04-28)
- **BUG 1 — Power logging 0.0 mW** (`hw_metrics.py`): Pi5 never emitted `power_total_soc_mw`; Jetson INA3221 i2c paths failed on newer kernels. Fixed: Pi5 sums hwmon rails; Jetson adds hwmon fallback. EnergyAccumulator now uses real elapsed time + tail-window.
- **SHOWSTOPPER — Learning collapse** (`aggregator.py:~225`): FLASHClient always sends deltas but aggregator only reconstructed `base+delta` when `r < 1.0`. Round 2 (r=1.0) averaged raw deltas as full weights → near-zero global model → loss stuck at 2.303 for all 60 rounds. Fixed: always add base to delta for FLASH strategy.
- Verified locally with `verify_flash.py`: compression round-trip PASS, aggregation regression PASS, 5-round MNIST training PASS (loss 2.30→0.34, acc 7.5%→90.8%).

### FIXED (2026-04-27)
- **`FLASHClient.__init__` signature mismatch** (`clients.py:406`): dropped `num_total_clients` param. Added and passed to `super()`.
- **Signal handler in daemon thread** (`aggregator.py:444`): flwr 1.29.0 raises `ValueError` in non-main threads. Patched to no-op.
- **Unicode chars in print statements**: caused `UnicodeEncodeError` on Windows. Replaced with ASCII.
- **BUG 2 (`inf` values)**: Verified guarded. Not a live bug.

---

## Planned Work (Action Plan)

### Phase 1 — Critical Fixes
1. ~~Fix power logging (0.0 mW on Chameleon)~~ DONE
2. Implement early stopping in `FLASHGlobalStrategy`
3. ~~Sanitize inf values~~ — verified not a live bug

### Phase 2 — Baselines
4. Pure local training baseline script
5. Simulated global pooled training baseline script

### Phase 3 — Advanced Features
6. Top-k sparsification ablation study (compare payload size vs accuracy)
7. Autonomous adaptive compression (feedback loop on real-time jitter)
8. Online cost model calibration (dynamic `k_comp`/`k_comm` weighting)

---

## MCP Tools Available

| Tool | Purpose | How to use |
|------|---------|------------|
| **Serena** | LSP-based Python code navigation — jump-to-definition, find-references, symbol search | Ask to "find all references to `_pick_r`" or "navigate to `FLASHClient`" |
| **ast-grep** | Structural AST pattern search — finds code by shape, not just text | Ask to "find all `np.mean(...)` calls with an empty list risk" or "find every dict return with an `energy` key" |
| **fetch** | Fetch external docs on demand (Flower, PyTorch, gRPC) | Ask to "fetch the Flower strategy API docs" |
| **gemini-cli** | Google Gemini CLI as a second AI — parallel code review, web research, architecture analysis | Invoke via `/gemini-cli`; useful for second opinions or current web info |

---

## Invocation

```bash
# 0. Start central log aggregator on Chameleon (do this first)
python log_server.py

# 1. Server (Chameleon Cloud)
export FLASH_LOG_SERVER=<chameleon_ip>
python server.py --rounds 60 --port 8080 --experiment all

# 2. Aggregator (Xavier)
export FLASH_LOG_SERVER=<chameleon_ip>
python aggregator.py --strategy flash --agg-port 8081 --server-address <IP>:8080 --rounds 60

# 3. Client (Pi 5 / Nano)
export FLASH_LOG_SERVER=<chameleon_ip>
python clients.py --cid 0 --agg-address <IP>:8081 --strategy flash

# Verify pipeline locally before any hardware run
python verify_flash.py
```

---

## Architecture Notes

- **Persistent connections:** Inner Flower server stays running for the full experiment (avoids per-round restart overhead).
- **Threading:** Inner server runs in daemon thread; main thread is the AggregatorClient communicating upstream.
- **Synchronization:** `_round_ready` and `_round_done` threading events gate each round.
- **Delta encoding:** Only FLASH sends compressed parameter deltas; FLARE/FedAvg send full parameters.
- **gRPC 4 MB limit:** `MAX_METRIC_KEYS = 200` caps columns per round to stay under the gRPC message size limit.
