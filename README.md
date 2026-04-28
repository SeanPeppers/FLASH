# FLASH -- Federated Learning with Adaptive Sparsification and Hybrid Compression

Hierarchical federated learning (HFL) research comparing three strategies across a three-tier edge-to-cloud topology:

| Strategy | Description |
|----------|-------------|
| **FLASH** | Top-k delta compression + latency-aware scheduling |
| **FLARE** | Fixed compression ratio |
| **FedAvg** | Vanilla federated averaging (baseline) |

**Topology:**
```
Chameleon Cloud (GPU)     -- server.py         (1 node)
        | gRPC
Jetson Xavier NX/AGX      -- aggregator.py     (1 node, port 8081)
        | gRPC
Pi 5 + Jetson Nano        -- clients.py        (2 leaf clients, CID 0-1)
```

All three strategies run back-to-back in a single experiment (`--experiment all`).  
All experiments write the same CSV columns so results are directly comparable.

---

## Hardware Setup

| Role | Device | IP |
|------|--------|----|
| Server | Chameleon Cloud node | `<CHAMELEON_PUBLIC_IP>` (public) / `<CHAMELEON_PRIVATE_IP>` (private) |
| Aggregator | Jetson Xavier | `<XAVIER_PRIVATE_IP>` (private) |
| Client 0 | Raspberry Pi 5 | connects to Xavier |
| Client 1 | Jetson Nano | connects to Xavier |

---

## Environment Setup

Run this on **every device** (Chameleon, Xavier, Pi 5, Nano):

```bash
git clone https://github.com/SeanPeppers/FLASH.git
cd FLASH
bash setup_env.sh
```

To activate in future sessions:
```bash
source flash/bin/activate
```

> The setup script auto-detects CUDA and installs the appropriate PyTorch build.

---

## Full Experiment -- Real Hardware

Run these steps in order. Start the server first, then the aggregator, then the clients.

### Step 1 -- Chameleon Cloud (Server)

SSH into the Chameleon node and run:

```bash
python server.py \
    --rounds 60 \
    --port 8080 \
    --aggregators 1 \
    --experiment all \
    --no-wait \
    --output-dir ./fl_results_hfl
```

> `--experiment all` runs FLASH, FLARE, and FedAvg back-to-back automatically.  
> `--aggregators 1` matches the single Xavier in this topology.

---

### Step 2 -- Jetson Xavier (Aggregator)

First, establish an SSH tunnel from the Xavier so it can reach the Chameleon server via localhost:

```bash
ssh -i ../FLASH.pem -L 8080:127.0.0.1:8080 cc@<CHAMELEON_PUBLIC_IP>
```

> Keep this tunnel open in a separate terminal for the duration of the experiment.

Then run the aggregator:

```bash
python aggregator.py \
    --strategy all \
    --rounds 60 \
    --leaf-clients 2 \
    --agg-port 8081 \
    --server-address 127.0.0.1:8080
```

> `--strategy all` matches the server running all three experiments.  
> `--leaf-clients 2` matches the Pi 5 + Nano topology.

---

### Step 3 -- Leaf Clients

**Raspberry Pi 5** (CID 0):
```bash
python clients.py \
    --cid 0 \
    --num-clients 2 \
    --strategy all \
    --agg-address <XAVIER_PRIVATE_IP>:8081
```

**Jetson Nano** (CID 1):
```bash
python clients.py \
    --cid 1 \
    --num-clients 2 \
    --strategy all \
    --agg-address <XAVIER_PRIVATE_IP>:8081
```

> `--num-clients 2` ensures each client gets the correct MNIST shard (every 2nd sample).  
> `--strategy all` keeps the client looping through all three experiments.

---

### Step 4 -- Baselines (run on Chameleon after FL completes)

These run entirely on the Chameleon node -- no other devices needed.

**Global pooled baseline** (accuracy ceiling -- simulates all data on one node):
```bash
python baseline_global.py --rounds 60 --output-dir ./fl_results_hfl
```

**Local-only baseline** (accuracy floor -- single client, no federation):
```bash
python baseline_local.py --rounds 60 --output-dir ./fl_results_hfl
```

---

## Local Simulation (No Hardware Required)

Runs the full FL pipeline on a single machine using localhost. Useful for testing code changes before deploying to real hardware. Hardware energy metrics will read 0 (no real sensors) but training and accuracy results are valid.

```bash
python run_all.py \
    --mode local \
    --rounds 60 \
    --output-dir ./fl_results_hfl
```

> This launches 1 server + 1 aggregator + 2 clients automatically, runs all 3 strategies, then runs both baselines.

To skip the FL portion and only run baselines:
```bash
python run_all.py --mode local --skip-fl
```

To skip baselines and only run FL:
```bash
python run_all.py --mode local --skip-baselines
```

---

## Output CSVs

All experiments write to `fl_results_hfl/`. Every run produces the same columns so results are directly comparable across strategies and hardware.

| File | Written by | Content |
|------|-----------|---------|
| `flash_HFL_eval_loss.csv` | server.py | Per-round eval loss |
| `flash_HFL_eval_metrics.csv` | server.py | Per-round eval accuracy |
| `flash_HFL_fit_metrics.csv` | server.py | Per-round client fit metrics (400+ cols) |
| `flash_server_hw.csv` | server.py | Chameleon hardware per round |
| `global_baseline_*.csv` | baseline_global.py | Pooled training reference |
| `local_baseline_*.csv` | baseline_local.py | Single-client reference |

Metric prefixes: leaf client metrics use `leaf_`, aggregator metrics use `agg_`.

---

## Ablation Study (optional)

Tests each compression ratio in isolation to measure the payload-size vs accuracy tradeoff. Run once per ratio on the real hardware setup above, pointing each run at a separate output directory:

```bash
# On Chameleon -- repeat for each ratio
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 1.0  --output-dir ./fl_results_hfl/r100
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.75 --output-dir ./fl_results_hfl/r075
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.5  --output-dir ./fl_results_hfl/r050
python server.py --experiment flash --rounds 60 --no-wait --fixed-r 0.25 --output-dir ./fl_results_hfl/r025
```

> Pass the matching `--fixed-r` to the aggregator and clients for each run.

Then generate plots:
```bash
python ablation_topk.py \
    --results-dirs fl_results_hfl/r100 fl_results_hfl/r075 fl_results_hfl/r050 fl_results_hfl/r025 \
    --labels "r=1.0" "r=0.75" "r=0.5" "r=0.25" \
    --output-dir ./fl_results_hfl/ablation
```
