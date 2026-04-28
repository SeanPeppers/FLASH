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
| `fl_results_hfl/` | Output directory for all CSV results | — |

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
- `EnergyAccumulator` (lines 391–433): Polls every 1 s, integrates `power_mw * interval` → joules.
  - Power source priority (lines 419–423): `power_total_soc_mw` → `gpu_power_nvml_mw` → `ipmi_system_power_w * 1000`

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

### BUG 1: Power logging returns 0.0 mW (showstopper for energy paper)
- **Location:** `hw_metrics.py:419–423` (`EnergyAccumulator._loop`)
- **Cause:** On Chameleon Cloud, IPMI command likely unavailable or permission-denied; no fallback to RAPL.
- **Fix needed:** Diagnose which source is active; fall back to RAPL energy counters if IPMI returns 0.

### CONCERN: Accuracy volatility (4.69%–20.31%) + stagnant loss (0.73% improvement over 60 rounds)
- Peak accuracy at Round 10 (20.31%), drops to 7.81% by Round 60.
- Only 2 clients per round — global update is noisy.
- No early stopping, no LR decay.
- **Fix needed:** Early stopping on validation loss; consider LR scheduler.

### FIXED (2026-04-27)
- **`FLASHClient.__init__` signature mismatch** (`clients.py:406`): dropped `num_total_clients` param, crashing the 3-arg registry call. Added param and passed to `super()`.
- **Signal handler in daemon thread** (`aggregator.py:444`): flwr 1.29.0 calls `signal.signal()` inside `start_server()`, which raises `ValueError` in non-main threads. Patched `signal.signal` to a no-op inside the inner server thread.
- **Unicode chars in print statements** (`server.py`, `baseline_global.py`, `baseline_local.py`, `run_all.py`): `—`, `×`, `–`, `──`, `→` caused `UnicodeEncodeError` on Windows (CP1252). Replaced with ASCII equivalents.
- **BUG 2 (`inf` values)**: Verified all `np.mean/max/min` calls in `build_metrics` are guarded by `if per_epoch:` or `if grad_norms else 0.0`; all divisions have `> 0` guards. Not a live bug.

---

## Planned Work (Action Plan)

### Phase 1 — Critical Fixes
1. Fix power logging (0.0 mW on Chameleon)
2. Implement early stopping in `FLASHGlobalStrategy`
3. Sanitize inf values in `build_metrics` / `train_one_epoch`

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
# Server (Chameleon Cloud)
python server.py --rounds 60 --port 8080 --experiment all

# Aggregator (Xavier)
python aggregator.py --strategy flash --agg-port 8081 --server-address <IP>:8080 --rounds 60

# Client (Pi 5 / Nano)
python clients.py --cid 0 --agg-address <IP>:8081 --strategy flash
```

---

## Architecture Notes

- **Persistent connections:** Inner Flower server stays running for the full experiment (avoids per-round restart overhead).
- **Threading:** Inner server runs in daemon thread; main thread is the AggregatorClient communicating upstream.
- **Synchronization:** `_round_ready` and `_round_done` threading events gate each round.
- **Delta encoding:** Only FLASH sends compressed parameter deltas; FLARE/FedAvg send full parameters.
- **gRPC 4 MB limit:** `MAX_METRIC_KEYS = 200` caps columns per round to stay under the gRPC message size limit.
