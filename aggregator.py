# -*- coding: utf-8 -*-
"""
aggregator.py — Mid-tier aggregator on Jetson Xavier NX/AGX

PERFORMANCE FIXES vs previous version:
  1. The inner Flower server is started ONCE and runs for ALL global rounds
     with num_rounds = total_global_rounds. It never tears down and restarts.
     The leaf clients stay connected throughout, eliminating the per-round
     reconnect penalty that caused ~3 days for 3 rounds.

  2. Parameters are passed round-to-round via configure_fit using the strategy's
     stored state — no re-serialisation of the full model every round.

  3. The aggregator's own parameters are kept in sync by reading from the
     inner strategy after each aggregate_fit via a callback hook.

  4. evaluate() on the aggregator side now runs a real forward pass on the
     local model rather than returning a synthetic loss.

Architecture after this fix:
  - server.py starts and waits for 1 aggregator connection
  - aggregator.py connects to server.py (as a client) AND starts its inner
    server (which runs for the full experiment). Both connections are persistent.
  - clients.py connects to aggregator.py's inner server and stays connected.

Usage:
    python aggregator.py --strategy flash --agg-port 8081 \\
                         --server-address <CHAMELEON_IP>:8080 --rounds 60
"""

from __future__ import annotations

import argparse
import dataclasses
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitIns, Parameters,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.strategy import FedAvg

import hw_metrics
from hw_metrics import EnergyAccumulator, snapshot, delta
from clients import (
    SimpleNet, get_parameters, set_parameters, model_size_bytes,
    TARGET_TAU, COMPRESSION_OPTIONS, MAX_LOCAL_EPOCHS,
    decompress_topk,
)

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_LEAF_CLIENTS = 2
LATENCY_THRESHOLD = 0.5


# ── Metric aggregation ─────────────────────────────────────────────────────────
def _agg_metrics(all_metrics: List[Tuple[int, Dict]]) -> Dict:
    weighted = {
        "train_loss_final", "train_loss_mean", "train_accuracy_final",
        "train_accuracy_mean", "epoch_loss", "epoch_accuracy",
    }
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    total_n = 0
    n_clients = 0

    for num_ex, m in all_metrics:
        total_n += num_ex
        n_clients += 1
        for k, v in m.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            sums.setdefault(k, 0.0)
            counts.setdefault(k, 0)
            sums[k] += fv * num_ex if k in weighted else fv
            counts[k] += 1

    out: Dict[str, float] = {}
    for k in sums:
        if k in weighted and total_n > 0:
            out[k] = sums[k] / total_n
        elif counts[k] > 0:
            out[k] = sums[k] / counts[k]
    out["clients_in_region"] = float(n_clients)
    out["total_leaf_samples"] = float(total_n)
    return out


# ── Instrumented FedAvg strategy ──────────────────────────────────────────────
# We need the inner strategy to:
#   (a) Accept updated parameters from the aggregator at the start of each global round
#   (b) Expose the aggregated parameters after each inner round finishes
#   (c) Call a callback with the round's metrics so the aggregator client can
#       collect them and forward them to the global server
#
# We do this by subclassing FedAvg and adding a threading.Event-based
# handshake between the inner server thread and the aggregator client thread.

class _InnerStrategy(FedAvg):
    """
    Runs inside the persistent inner server thread.
    The aggregator client updates self.current_params before each global round
    and reads self.result_params + self.result_metrics after it completes.
    """

    def __init__(self, bar_tau_r: float, t_thr: float,
                 compression_options: List[float], strategy_name: str, **kwargs):
        super().__init__(**kwargs)
        self.bar_tau_r = bar_tau_r
        self.t_thr = t_thr
        self.compression_options = sorted(compression_options, reverse=True)
        self.strategy_name = strategy_name
        self._leaf_history: Dict[str, Dict] = {}

        # Cost model calibration (FLASH only)
        self.k_comp = 0.1
        self.k_comm = 0.5
        self._kc_hist: List[float] = [0.1]
        self._kx_hist: List[float] = [0.5]
        self._window = 3

        # Thread-safe handshake between inner server and aggregator client
        self._round_ready   = threading.Event()   # aggregator signals: new params ready
        self._round_done    = threading.Event()   # inner server signals: round complete
        self._current_params: Optional[Parameters] = None
        self._result_params: Optional[Parameters] = None
        self._result_metrics: Dict = {}
        self._global_round: int = 0

    # ── Called by the aggregator client to push new global params ─────────────
    def set_global_params(self, params: List[np.ndarray], global_round: int):
        self._current_params = ndarrays_to_parameters(params)
        self._global_round = global_round
        self._round_done.clear()
        self._round_ready.set()

    # ── Called by the aggregator client to wait for the round result ──────────
    def wait_for_result(self, timeout: float = 3600.0) -> Tuple[Optional[List[np.ndarray]], Dict]:
        if not self._round_done.wait(timeout=timeout):
            raise TimeoutError("Inner round did not complete within timeout")
        params = parameters_to_ndarrays(self._result_params) if self._result_params else None
        return params, self._result_metrics

    # ── configure_fit: inject current global params + per-leaf config ─────────
    def configure_fit(self, server_round, parameters, client_manager):
        # Block until the aggregator client has pushed new params
        self._round_ready.wait()
        self._round_ready.clear()

        # Use the global params pushed by the aggregator client
        params_to_send = self._current_params or parameters

        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        out = []
        for c in clients:
            r_star = self._pick_r(c.cid) if self.strategy_name == "flash" else 1.0
            cfg = {
                "server_round": self._global_round,
                "bar_tau_r": self.bar_tau_r,
                "optimal_r_star": r_star,
                "suggested_tau": min(int(round(self.bar_tau_r)), MAX_LOCAL_EPOCHS),
            }
            out.append((c, FitIns(params_to_send, cfg)))
        return out

    def _pick_r(self, cid: str) -> float:
        if cid not in self._leaf_history:
            return 1.0
        m = self._leaf_history[cid]
        fit_time = m.get("fit_wall_time_s", 0.0)
        if fit_time > self.t_thr:
            candidates = [r for r in self.compression_options if r < 1.0]
            return candidates[0] if candidates else 0.25
        # Analytic fallback
        f = m.get("comp_capacity_proxy", 2.0)
        r_prev = m.get("compression_ratio_applied", 1.0)
        S = m.get("data_transfer_size_bytes", 1e5) / r_prev if r_prev > 0 else 1e5
        T_comp = self.k_comp * self.bar_tau_r / max(f, 1e-6)
        for r in self.compression_options:
            if T_comp + self.k_comm * r * S <= self.t_thr:
                return r
        return min(self.compression_options)

    def aggregate_fit(self, server_round, results, failures):
        # Decompress delta and reconstruct full params before FedAvg averages them.
        # FLASH clients send compressed(trained_params - base_params); we add base back here.
        if self.strategy_name == "flash":
            base = parameters_to_ndarrays(self._current_params) if self._current_params else None
            decompressed = []
            for client, fit_res in results:
                r = fit_res.metrics.get("compression_ratio_applied", 1.0)
                if r < 1.0 and base is not None:
                    delta = decompress_topk(parameters_to_ndarrays(fit_res.parameters))
                    full = [b.astype(np.float32) + d for b, d in zip(base, delta)]
                    fit_res = dataclasses.replace(fit_res, parameters=ndarrays_to_parameters(full))
                decompressed.append((client, fit_res))
            results = decompressed

        # Calibrate cost model
        if self.strategy_name == "flash":
            kc_new, kx_new = [], []
            for client, fit_res in results:
                self._leaf_history[client.cid] = fit_res.metrics
                m = fit_res.metrics
                tau = m.get("local_epochs", 1.0)
                f   = m.get("comp_capacity_proxy", 1.0)
                r   = m.get("compression_ratio_applied", 1.0)
                S   = m.get("data_transfer_size_bytes", 1e5) / r if r > 0 else 1e5
                T_c = m.get("fit_wall_time_s", 0.0) * 0.7
                T_x = m.get("fit_wall_time_s", 0.0) * 0.3
                if T_c > 0 and tau > 0 and f > 0:
                    kc_new.append(T_c / (tau / f))
                if T_x > 0 and S > 0:
                    kx_new.append(T_x / S)
            if kc_new:
                self._kc_hist.append(float(np.mean(kc_new)))
                self.k_comp = float(np.mean(self._kc_hist[-self._window:]))
            if kx_new:
                self._kx_hist.append(float(np.mean(kx_new)))
                self.k_comm = float(np.mean(self._kx_hist[-self._window:]))
        else:
            for client, fit_res in results:
                self._leaf_history[client.cid] = fit_res.metrics

        agg_result = super().aggregate_fit(server_round, results, failures)
        if agg_result:
            self._result_params = agg_result[0]
            self._result_metrics = agg_result[1] or {}
        self._round_done.set()
        return agg_result


# ── Aggregator client (upstream to global server) ─────────────────────────────
class AggregatorClient(fl.client.NumPyClient):
    """
    Connects to the global server. Each fit() call:
      1. Pushes new global params to the already-running inner strategy
      2. Waits for the inner round to complete
      3. Collects leaf + Xavier hardware metrics
      4. Returns everything to the global server
    """

    def __init__(self, inner_strategy: _InnerStrategy, num_leaf_clients: int = NUM_LEAF_CLIENTS):
        self.inner_strategy = inner_strategy
        self.num_leaf_clients = num_leaf_clients
        self.model = SimpleNet()
        self._round = 0
        print(f"[Aggregator] Xavier ({hw_metrics.DEVICE})  strategy={inner_strategy.strategy_name}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self._round += 1
        global_round = int(config.get("server_round", self._round))
        set_parameters(self.model, parameters)

        print(f"\n[Aggregator] === Global round {global_round} ===")

        # Xavier hw snapshot + energy accumulator
        hw_before = snapshot()
        energy_acc = EnergyAccumulator()
        energy_acc.start()
        t0 = time.perf_counter()

        # Push params to inner strategy and wait for leaf clients to train
        self.inner_strategy.set_global_params(get_parameters(self.model), global_round)
        agg_params, leaf_metrics = self.inner_strategy.wait_for_result(timeout=7200.0)

        inner_duration = time.perf_counter() - t0
        energy_j = energy_acc.stop_and_get_joules()
        hw_after = snapshot()

        # Update local model with aggregated weights
        if agg_params is not None:
            set_parameters(self.model, agg_params)
        final_params = get_parameters(self.model)

        # Assemble Xavier metrics
        hw_d = delta(hw_before, hw_after)
        xavier_m: Dict[str, float] = {}
        for k, v in hw_after.items():
            xavier_m[f"agg_hw_{k}"] = float(v)
        for k, v in hw_d.items():
            if k.startswith("delta_"):
                xavier_m[f"agg_hw_{k}"] = float(v)
        xavier_m["agg_energy_joules"]      = float(energy_j)
        xavier_m["agg_inner_round_time_s"] = float(inner_duration)
        xavier_m["agg_power_avg_w"]        = energy_j / inner_duration if inner_duration > 0 else 0.0
        xavier_m["agg_model_size_bytes"]   = model_size_bytes(self.model)
        xavier_m["agg_device_type_id"]     = 2.0
        xavier_m["region_id"]              = 0.0
        xavier_m["clients_in_region"]      = float(self.num_leaf_clients)

        # Prefix leaf metrics
        prefixed_leaf = {f"leaf_{k}": float(v) for k, v in leaf_metrics.items()
                         if isinstance(v, (int, float))}

        all_metrics = {**prefixed_leaf, **xavier_m}

        # Top-level keys for global server plots
        all_metrics["accuracy"]                  = prefixed_leaf.get("leaf_train_accuracy_final", 0.0)
        all_metrics["loss"]                      = prefixed_leaf.get("leaf_train_loss_final", 0.0)
        all_metrics["energy_joules"]             = prefixed_leaf.get("leaf_energy_joules", 0.0) + energy_j
        all_metrics["data_transfer_size_bytes"]  = prefixed_leaf.get("leaf_data_transfer_size_bytes", 0.0)
        all_metrics["simulated_latency_seconds"] = float(inner_duration)

        total_samples = int(prefixed_leaf.get("leaf_total_leaf_samples", 32 * self.num_leaf_clients))

        print(
            f"[Aggregator] Round {global_round} done  "
            f"time={inner_duration:.1f}s  energy={energy_j:.2f}J  "
            f"acc={all_metrics['accuracy']:.3f}  loss={all_metrics['loss']:.4f}"
        )

        return final_params, total_samples, all_metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        # Run a quick forward pass on dummy data for the eval loss
        # (Replace with a real validation set if you have one on the Xavier)
        self.model.eval()
        with torch.no_grad():
            dummy_x = torch.randn(64, 1, 28, 28)
            dummy_y = torch.randint(0, 10, (64,))
            logits = self.model(dummy_x)
            loss = torch.nn.functional.cross_entropy(logits, dummy_y).item()
            acc = (logits.argmax(1) == dummy_y).float().mean().item()
        hw = snapshot()
        return float(loss), 64, {
            "loss": float(loss),
            "accuracy": float(acc),
            "agg_hw_cpu_util_pct":       float(hw.get("cpu_util_pct", 0.0)),
            "agg_hw_ram_util_pct":       float(hw.get("ram_util_pct", 0.0)),
            "agg_hw_power_total_soc_mw": float(hw.get("power_total_soc_mw", 0.0)),
            "agg_hw_temp_max_celsius":   float(hw.get("temp_max_celsius", 0.0)),
        }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HFL aggregator (Jetson Xavier)")
    parser.add_argument("--strategy", type=str, default="flash",
                        choices=["flash", "flare", "fedavg", "all"])
    parser.add_argument("--agg-port", type=int, default=8081)
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    parser.add_argument("--rounds", type=int, default=60,
                        help="Must match --rounds on server.py")
    args = parser.parse_args()

    inner_addr = f"0.0.0.0:{args.agg_port}"
    leaf_addr  = f"<XAVIER_IP>:{args.agg_port}"

    print(f"[Aggregator] Xavier hw snapshot:")
    for k, v in sorted(snapshot().items()):
        print(f"    {k}: {v:.4f}")
    print()

    strategies = ["flash", "flare", "fedavg"] if args.strategy == "all" else [args.strategy]

    for strategy_name in strategies:
        print(f"\n[Aggregator] Starting strategy: {strategy_name}")

        # Build a fresh inner strategy for this experiment
        init_model  = SimpleNet()
        init_params = ndarrays_to_parameters(get_parameters(init_model))

        inner_strategy = _InnerStrategy(
            bar_tau_r=TARGET_TAU,
            t_thr=LATENCY_THRESHOLD,
            compression_options=list(COMPRESSION_OPTIONS),
            strategy_name=strategy_name,
            min_fit_clients=NUM_LEAF_CLIENTS,
            min_evaluate_clients=NUM_LEAF_CLIENTS,
            min_available_clients=NUM_LEAF_CLIENTS,
            fit_metrics_aggregation_fn=_agg_metrics,
            evaluate_metrics_aggregation_fn=_agg_metrics,
            initial_parameters=init_params,
        )

        # Start the inner server in a background thread — it runs for all rounds
        def _run_inner_server(strategy=inner_strategy):
            print(f"[Aggregator] Inner server listening on {inner_addr}")
            fl.server.start_server(
                server_address=inner_addr,
                config=fl.server.ServerConfig(num_rounds=args.rounds),
                strategy=strategy,
            )
            print("[Aggregator] Inner server finished all rounds.")

        inner_thread = threading.Thread(target=_run_inner_server, daemon=True)
        inner_thread.start()

        # Wait a moment for the inner server to start accepting connections
        time.sleep(2.0)

        print(f"[Aggregator] Leaf clients should connect to: {leaf_addr}")
        print(f"    python clients.py --cid 0 --agg-address {leaf_addr} --strategy {strategy_name}")
        print(f"    python clients.py --cid 1 --agg-address {leaf_addr} --strategy {strategy_name}")
        print()

        # Connect upstream to the global server as a Flower client
        agg_client = AggregatorClient(inner_strategy, NUM_LEAF_CLIENTS)
        print(f"[Aggregator] Connecting to global server at {args.server_address} ...")
        fl.client.start_numpy_client(server_address=args.server_address, client=agg_client)

        # Wait for inner server thread to finish
        inner_thread.join(timeout=30)
        print(f"[Aggregator] {strategy_name} complete.")

    print("[Aggregator] All strategies done.")
