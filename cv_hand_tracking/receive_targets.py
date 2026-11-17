#!/usr/bin/env python3
"""Local-only example of the S2 -> S3 interface. Prints targets; never moves hardware.

Both processes must run on the SAME HOST because freshness uses monotonic time.
Replace consume_target() with a robot-model retargeter, then pass its solution to
the independent S4 command gate. Never bypass that gate with these human targets.
"""
import argparse
import json
import socket
import time


def consume_target(packet):
    target = packet["teleop_targets"]
    curl = " ".join(f"{finger}={value:.2f}" for finger, value in target["curl_01"].items())
    print(f"frame {packet['source_frame_id']}: {curl}", flush=True)
    # S3 integration point:
    # 1. Associate this source timestamp with exoskeleton input (I1).
    # 2. Scale target fingertips to the LEAP model; solve constrained hand IK.
    # 3. Submit a coordinated arm/hand target pair to S4 with original timestamps.
    # 4. S4 validates limits, mode, device health and takeover pose before motion.


def usable(packet, now_ns, max_age_ms):
    """Reject delayed, disallowed, missing and malformed producer observations."""
    try:
        source_ns = packet["source_received_monotonic_ns"]
        age_ms = (now_ns - source_ns) / 1e6
        publish_ns = packet["published_monotonic_ns"]
        return (packet["schema"] == "quadradex.hand.v1"
                and packet["state"] == "STREAMING"
                and packet["tracking_valid"] is True
                and packet["calibrated"] is True
                and isinstance(packet["teleop_targets"], dict)
                and 0 <= age_ms <= min(max_age_ms, packet["max_age_ms"])
                and source_ns <= publish_ns <= now_ns)
    except (KeyError, TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--max-age-ms", type=float, default=100)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.max_age_ms < float("inf"):
        parser.error("Port or maximum age out of range")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(.02)
    last_valid_receipt = None
    last_sequence = last_frame = -1
    session = None
    last_session_start = -1
    running = False
    print(f"Listening on 127.0.0.1:{args.port}. Ctrl+C exits.", flush=True)
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                data = None
            now = time.perf_counter_ns()
            if data:
                try:
                    packet = json.loads(data)
                    if not isinstance(packet, dict):
                        raise ValueError("Expected object")
                    if packet.get("session_id") != session:
                        # Ignore delayed datagrams from a previously seen session.
                        start = packet["session_started_unix_ns"]
                        if start <= last_session_start:
                            continue
                        session, last_session_start = packet["session_id"], start
                        last_sequence = last_frame = -1
                    sequence = packet["sequence"]
                    if sequence <= last_sequence:
                        continue
                    last_sequence = sequence
                    source_now = time.monotonic_ns() if packet.get("clock") == "host_monotonic_ns" else now
                    if packet.get("clock") not in ("host_monotonic_ns", "host_perf_counter_ns"):
                        raise ValueError("Unsupported source clock")
                    if usable(packet, source_now, args.max_age_ms):
                        source_frame = packet["source_frame_id"]
                        if source_frame > last_frame:
                            consume_target(packet)
                            last_frame = source_frame
                            last_valid_receipt = now
                            running = True
                    elif running:
                        print("INHIBIT: producer has no usable target", flush=True)
                        running = False
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    if running:
                        print("INHIBIT: malformed packet", flush=True)
                        running = False
            if running and (last_valid_receipt is None or (now - last_valid_receipt) / 1e6 > args.max_age_ms):
                print("INHIBIT: no new valid source frame before watchdog timeout", flush=True)
                running = False
    except KeyboardInterrupt:
        print("INHIBIT: receiver stopped", flush=True)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
