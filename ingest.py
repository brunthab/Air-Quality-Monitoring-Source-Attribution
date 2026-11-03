"""MQTT -> SQLite ingestion for real IoT nodes.

Run:  python -m aerotrace.ingest --broker localhost
Topic: aerotrace/<node_id>/reading   Payload: JSON {"pm25":..,"pm10":..,"co":..,"temp":..,"rh":..}
Timestamps come from the receive time (the ESP32 skeleton sends no clock).
"""
import argparse
import json
import sqlite3
import time

import paho.mqtt.client as mqtt

TOPIC = "aerotrace/+/reading"
LIMITS = {"pm25": (0, 1000), "pm10": (0, 1500), "co": (0, 100),
          "temp": (-20, 70), "rh": (0, 100)}


def valid(r: dict) -> bool:
    """Drop obviously invalid readings (preprocessing step 2 of the pipeline)."""
    for k, (lo, hi) in LIMITS.items():
        v = r.get(k)
        if v is None:
            if k in ("pm25", "pm10"):   # particulate readings are mandatory
                return False
            continue
        if not isinstance(v, (int, float)) or not lo <= v <= hi:
            return False
    return True


def init_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS readings(
        ts REAL, node TEXT, pm25 REAL, pm10 REAL, co REAL, temp REAL, rh REAL)""")
    con.commit()
    return con


def make_client():
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # paho-mqtt >= 2
    except AttributeError:
        return mqtt.Client()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--db", default="aerotrace.db")
    a = ap.parse_args()
    con = init_db(a.db)

    def on_connect(client, userdata, *args):
        client.subscribe(TOPIC)

    def on_message(client, userdata, msg):
        try:
            r = json.loads(msg.payload)
        except ValueError:
            return
        if not valid(r):
            print("dropped invalid reading:", r)
            return
        node = msg.topic.split("/")[1]
        con.execute("INSERT INTO readings VALUES (?,?,?,?,?,?,?)",
                    (time.time(), node, r.get("pm25"), r.get("pm10"),
                     r.get("co"), r.get("temp"), r.get("rh")))
        con.commit()

    c = make_client()
    c.on_connect, c.on_message = on_connect, on_message
    c.connect(a.broker, a.port)
    c.loop_forever()


if __name__ == "__main__":
    main()
