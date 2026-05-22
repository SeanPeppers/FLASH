# FLASH: Energy-Efficient Hierarchical Federated Learning for Sustainable Smart Home IoT Networks

FLASH is a hierarchical federated learning (HFL) framework designed for energy-efficient, latency-aware model training across resource-constrained IoT devices. It combines top-k delta compression, adaptive epoch scheduling, fidelity-weighted aggregation, and error-feedback residual compression to reduce communication overhead and edge-tier energy consumption without sacrificing model quality.

This repository accompanies the paper:

> Sean Peppers, Leon Fischer, Katharine Ringo, Tiying Gao, Juheng Zhang, Chengyi Qu.
> "FLASH: Energy-Efficient Hierarchical Federated Learning with Adaptive Compression for Sustainable Smart Home IoT Networks."
> *IEEE Transactions on Green Communications and Networking*, 2026 (under review).

---

## Strategies

| Strategy | Description |
|----------|-------------|
| **FLASH** | Adaptive top-k sparsification with latency-aware scheduling, fidelity-weighted aggregation, and error-feedback residuals |
| **FixedCompress** | Fixed top-k sparsification at r=0.75, no adaptive scheduling |
| **FedAvg** | Standard federated averaging with no compression (baseline) |
| **adaMC** | Adaptive per-layer compression with L2-norm-proportional ratio assignment (approximate; see paper) |

Run `--experiment all` to execute all four strategies back-to-back in a single invocation.

---

## Topology

```
Chameleon Cloud GPU node        server.py       (coordinator)
        | gRPC (port 8080)
Jetson Xavier NX                aggregator.py   (mid-tier aggregator)
        | gRPC (port 8081)
Raspberry Pi 5 + Jetson Nano    clients.py      (leaf clients, CID 0 and 1)
```

---

## Hardware Testbed

The evaluation in the paper used the following hardware:

| Role | Device |
|------|--------|
| Coordinator | Chameleon Cloud P100 GPU node (TACC, Texas) |
| Aggregator | NVIDIA Jetson Xavier NX |
| Leaf client 0 | Raspberry Pi 5 |
| Leaf client 1 | NVIDIA Jetson Nano |

The hardware testbed scripts in this repository require access to equivalent physical devices with gRPC connectivity between tiers. They are not runnable on general-purpose hardware without replicating this topology. The simulation mode below requires no special hardware.

---

## Setup

Run on every device that will participate in the experiment:

```bash
git clone https://github.com/SeanPeppers/FLASH.git
cd FLASH
pip install -r requirements.txt
```

Key dependency: `flwr>=1.8.0,<2.0.0`. Flower 2.x dropped the APIs this project relies on and is not compatible.

---

## Simulation (No Hardware Required)

The full FL pipeline can be run on a single machine using Flower's built-in simulation. Hardware power metrics will read 0 (no physical sensors), but training loss, accuracy, and compression behavior are valid.

### Standard IID simulation

```bash
python run_noniid_sim.py \
    --rounds 60 \
    --output-dir ./fl_results_sim
```

This launches one server, one aggregator, and two simulated clients running all four strategies sequentially.

### Non-IID simulation (Dirichlet partitioning)

```bash
# Moderate skew (Dir(0.5))
python run_noniid_sim.py \
    --rounds 60 \
    --dirichlet-alpha 0.5 \
    --output-dir ./fl_results_noniid/alpha_0.50

# Severe skew (Dir(0.1))
python run_noniid_sim.py \
    --rounds 60 \
    --dirichlet-alpha 0.1 \
    --output-dir ./fl_results_noniid/alpha_0.10
```

Under Dir(alpha), each client's class distribution is drawn from a Dirichlet distribution parameterized by alpha. Smaller alpha produces more extreme label imbalance. Results from these runs correspond to Table VII in the paper.

---

## Real Hardware Experiments

Run these steps in order. Start the server first, then the aggregator, then both clients.

### Step 1 — Coordinator (Chameleon Cloud)

```bash
python server.py \
    --rounds 60 \
    --port 8080 \
    --aggregators 1 \
    --experiment all \
    --no-wait \
    --output-dir ./fl_results_hfl
```

### Step 2 — Aggregator (Jetson Xavier NX)

Open an SSH tunnel from the Xavier to the coordinator so it can reach port 8080 via localhost, then run:

```bash
python aggregator.py \
    --strategy all \
    --rounds 60 \
    --leaf-clients 2 \
    --agg-port 8081 \
    --server-address 127.0.0.1:8080
```

### Step 3 — Leaf Clients

**Raspberry Pi 5** (CID 0):
```bash
python clients.py \
    --cid 0 \
    --num-clients 2 \
    --strategy all \
    --agg-address <AGGREGATOR_IP>:8081
```

**Jetson Nano** (CID 1):
```bash
python clients.py \
    --cid 1 \
    --num-clients 2 \
    --strategy all \
    --agg-address <AGGREGATOR_IP>:8081
```

Replace `<AGGREGATOR_IP>` with the local network address of the Xavier. Run three independent trials per dataset to reproduce the mean ± s.d. results reported in the paper.

---

## Datasets

Both MNIST and UCI-HAR are supported. Pass `--dataset mnist` or `--dataset ucihar` to `server.py`, `aggregator.py`, and `clients.py`. MNIST is the default. UCI-HAR data is loaded automatically from the `data/` directory on first run.

---

## Ablation Study

Tests each compression ratio in isolation to measure the payload-size vs. accuracy tradeoff. Run once per ratio with a separate output directory, passing the matching `--fixed-r` to the aggregator and clients for each run:

```bash
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 1.00 --output-dir ./fl_results_hfl/r100
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.75 --output-dir ./fl_results_hfl/r075
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.50 --output-dir ./fl_results_hfl/r050
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.25 --output-dir ./fl_results_hfl/r025
```

Then generate figures:
```bash
python plot_component_study.py
```

---

## Pipeline Verification

Before running on real hardware, verify the full pipeline locally:

```bash
python verify_flash.py
```

This runs a 5-round MNIST training loop, checks compression round-trip correctness, and confirms aggregation produces a valid global model. Expected output: loss falling from ~2.30 to ~0.34 and accuracy reaching ~90% by round 5.

---

## Output CSVs

All experiments write to the specified `--output-dir`. Every strategy produces the same column schema for direct comparison.

| File | Content |
|------|---------|
| `{strategy}_HFL_eval_loss.csv` | Per-round evaluation loss |
| `{strategy}_HFL_eval_metrics.csv` | Per-round accuracy and aggregator hardware metrics |
| `{strategy}_HFL_fit_metrics.csv` | Per-round client fit metrics (400+ columns) |
| `{strategy}_server_hw.csv` | Coordinator hardware telemetry per round |

Metric prefixes: `leaf_` for edge client metrics, `agg_` for aggregator metrics.

---

## Results

Hardware testbed results (60 rounds, MNIST, mean across 3 independent runs):

| Strategy | Loss (R60) | Accuracy (R60) | Edge Energy (kJ) | CO2 (g) | GEI |
|----------|-----------|----------------|------------------|---------|-----|
| FLASH | 0.091 | 99.38% | 67.3 | 8.12 | 1.476 |
| FixedCompress | 0.128 | 99.33% | 102.8 | 12.40 | 0.966 |
| FedAvg | 0.140 | 99.33% | 103.1 | 12.44 | 0.964 |
| adaMC | 0.147 | 99.27% | 103.9 | 12.53 | 0.956 |

GEI (Green Efficiency Index) = peak accuracy (%) per kilojoule consumed. FLASH achieves 34.7% lower edge-tier energy and 53.1% higher GEI than FedAvg while sustaining within 0.05% of uncompressed accuracy.

---

## Plotting

```bash
# Main paper figures (requires fl_results_hfl/ CSVs)
python plot_results.py

# Component ablation figures
python plot_component_study.py

# Cross-strategy summary table
python summarize_flash_results.py
```

Figures are written to `fl_results_hfl/figures/`.
