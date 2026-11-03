"""Fake ESP32 node: publishes plausible readings over MQTT (no hardware needed).

Run:  python -m aerotrace.node_sim --broker localhost --node node1
"""
import argparse
import json
import math
import random
import time

from .ingest import make_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--node", default="node1")
    ap.add_argument("--interval", type=float, default=5.0)
    a = ap.parse_args()

    c = make_client()
    c.connect(a.broker, a.port)
    c.loop_start()
    t0 = time.time()
    while True:
        phase = (time.time() - t0) / 60
        pm25 = max(2, 45 + 25 * math.sin(phase) + random.gauss(0, 4))
        msg = {"pm25": round(pm25, 1), "pm10": round(pm25 * 1.6 + random.gauss(0, 5), 1),
               "co": round(max(0.1, 0.8 + 0.4 * math.sin(phase / 2) + random.gauss(0, 0.05)), 2),
               "temp": round(30 + random.gauss(0, 0.3), 1), "rh": round(65 + random.gauss(0, 2), 1)}
        c.publish(f"aerotrace/{a.node}/reading", json.dumps(msg))
        print("sent", msg)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
