# AeroTrace

AI-powered air quality monitoring and pollution **source** intelligence
(monitor -> integrate -> predict & attribute -> inform).

## Quick start (no hardware needed)

```bash
pip install -r requirements.txt
python -m aerotrace.pipeline     # prints validation metrics, a hotspot attribution, 6h forecast
streamlit run app.py             # interactive dashboard
```

## Layout

| Path | Role (slide reference) |
|---|---|
| `aerotrace/simulate.py` | Synthetic IoT + satellite + weather data (replace with real loaders) |
| `aerotrace/model.py` | Random Forest ground-level PM2.5 estimation, validation, 6 h forecast (slide 12) |
| `aerotrace/attribution.py` | Wind + signature based ranked likely sources with confidence (slides 14-15, 19) |
| `aerotrace/aqi.py` | Estimated pollution score from CPCB-style breakpoints (slide 24) |
| `aerotrace/pipeline.py` | End-to-end glue + CLI demo |
| `aerotrace/ingest.py` | MQTT -> SQLite ingestion with invalid-reading filtering (slide 10) |
| `aerotrace/node_sim.py` | Fake ESP32 node publishing over MQTT |
| `app.py` | Streamlit dashboard: AQI, map, attribution, prediction, weather, alerts, history (slide 16) |
| `firmware/` | ESP-IDF C skeleton: PMS5003 + MQ-7 + MQTT (slide 20) |

## Validation (slide 23)

`evaluate()` reports MAE / RMSE / R2 two ways, on sensor rows only:
leave-one-sensor-out (how well an unmonitored location is estimated) and a
temporal hold-out. A mean-only baseline is printed alongside for context.
**These numbers are on simulated data** and say nothing about real-world accuracy.
Source attribution should be checked against documented events once you have them.

## Going from demo to real data

1. **IoT**: run a broker (e.g. `mosquitto`), flash `firmware/`, run `python -m aerotrace.ingest`.
   Read the SQLite `readings` table instead of simulated sensor rows.
2. **Satellite**: Sentinel-5P (NO2, SO2, CO, aerosol index) via Copernicus, MODIS AOD via NASA,
   Landsat via USGS. Resample to your grid with rasterio/xarray, drop cloud-masked pixels,
   keep `sat_age_h` (hours since last valid pass).
3. **Weather**: any API/historical source giving wind speed/direction, humidity, temperature.
4. **Sources**: replace `SOURCES` in `config.py` with real industrial/traffic/construction
   locations and fire hotspots (VIIRS/MODIS active fire).
5. Keep the same dataframe columns as `generate_dataset()` and the rest works unchanged.

## Honest limitations

- `plume_factor` is a screening heuristic, not a dispersion model.
- Attribution output is *likely contributors with confidence*, never proof against one factory.
- The score uses PM2.5 / PM10 / CO only: it is not an official CPCB AQI.
- Firmware is an untested skeleton; the MQ-7 CO value needs real calibration.
- CNN / LSTM / spatio-temporal GNN from slide 12 are proposed extensions, not included.
