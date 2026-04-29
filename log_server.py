# -*- coding: utf-8 -*-
"""
log_server.py -- Central log aggregator, runs on Chameleon Cloud.

Receives log records over TCP from all devices (Pi5, Xavier, Chameleon itself)
and writes everything to fl_results_hfl/flash_debug_aggregated.log.

Usage (run this BEFORE starting the experiment):
    python log_server.py

On each device, set the env var before launching:
    export FLASH_LOG_SERVER=<chameleon_ip>
    python server.py ...       # Chameleon
    python aggregator.py ...   # Xavier
    python clients.py ...      # Pi 5 / Nano

The aggregated log includes a %(device)s column so you can see which
machine each line came from at a glance.
"""

from __future__ import annotations

import logging
import logging.handlers
import pickle
import socketserver
import struct
from pathlib import Path

PORT = 9020
LOG_PATH = Path("fl_results_hfl/flash_debug_aggregated.log")


def _setup_output_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(device)-14s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("aggregated")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


_out = _setup_output_logger()


class _LogRecordHandler(socketserver.StreamRequestHandler):
    """Receives length-prefixed pickled LogRecord frames (standard SocketHandler protocol)."""

    def handle(self):
        client_addr = self.client_address[0]
        _out.info("Connection from %s", client_addr,
                  extra={"device": "log_server"})
        try:
            while True:
                # Each frame: 4-byte big-endian length + pickled LogRecord dict
                header = self._recv_exact(4)
                if not header:
                    break
                length = struct.unpack(">L", header)[0]
                data = self._recv_exact(length)
                if not data:
                    break
                record_dict = pickle.loads(data)
                record = logging.makeLogRecord(record_dict)
                # Ensure device field exists (set by _DeviceFilter on the sender)
                if not hasattr(record, "device"):
                    record.device = client_addr
                _out.handle(record)
        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            _out.warning("Error handling record from %s: %s", client_addr, exc,
                         extra={"device": "log_server"})

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.connection.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf


class _ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    _out.info("=" * 60, extra={"device": "log_server"})
    _out.info("FLASH log server listening on port %d", PORT, extra={"device": "log_server"})
    _out.info("Writing to %s", LOG_PATH, extra={"device": "log_server"})
    _out.info("=" * 60, extra={"device": "log_server"})
    print(f"Log server running on port {PORT}. Ctrl-C to stop.")
    print(f"Output: {LOG_PATH}")

    with _ThreadedTCPServer(("0.0.0.0", PORT), _LogRecordHandler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down log server.")
