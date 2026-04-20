# FLASH — Federated Learning with Adaptive Sparsification and Hybrid compression

## Setup

```bash
git clone https://github.com/SeanPeppers/FLASH.git
cd FLASH
bash setup_env.sh
```

To activate the environment in future sessions:
```bash
source flash/bin/activate
```

---

## Running an Experiment

### 1. Chameleon Cloud Node (Server)

SSH into the node and run:
```bash
python server.py --rounds 60 --port 8080 --experiment all
```

### 2. Jetson Xavier (Aggregator)

Establish an SSH tunnel from your local machine to the Chameleon node:
```bash
ssh -i ../FLASH.pem -L 8080:127.0.0.1:8080 cc@<chameleon-node-ip>
```
> Example: `cc@10.0.0.1`

Then run the aggregator:
```bash
python aggregator.py --strategy flash --agg-port 8081 --server-address 127.0.0.1:8080
```

### 3. Edge Clients

**Raspberry Pi 5** (CID 0):
```bash
python clients.py --cid 0 --strategy flash --agg-address <xavier-ip>:8081
```
> Example: `--agg-address 192.168.1.100:8081`

**Jetson Nano** (CID 1):
```bash
python clients.py --cid 1 --strategy flash --agg-address <xavier-ip>:8081
```
> Example: `--agg-address 192.168.1.100:8081`
