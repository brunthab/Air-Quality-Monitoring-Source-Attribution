"""Shared configuration: study area, pollution sources, pollutant signatures."""
from dataclasses import dataclass

# Local study area is a square grid in km east/north of this origin (Chennai-ish).
ORIGIN_LAT, ORIGIN_LON = 13.00, 80.15
GRID_N = 15          # 15 x 15 cells
CELL_KM = 0.8        # => 12 km x 12 km area
N_SENSORS = 8

POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co"]


@dataclass(frozen=True)
class Source:
    name: str
    kind: str        # industrial | traffic | construction | biomass
    x: float         # km east of origin
    y: float         # km north of origin
    strength: float  # relative emission strength (0..1)


# Known source inventory (in a real deployment: industrial registry, traffic
# counts, construction permits, VIIRS/MODIS fire hotspots).
SOURCES = [
    Source("Industrial zone (NE)", "industrial", 9.0, 9.0, 1.0),
    Source("Highway junction (W)", "traffic", 2.5, 6.0, 0.9),
    Source("Ring road crossing (S)", "traffic", 6.5, 2.0, 0.8),
    Source("Construction cluster (C)", "construction", 5.5, 6.5, 0.7),
    Source("Crop/waste burning area (SE)", "biomass", 10.0, 2.5, 0.9),
]

# How strongly each source kind shows up in each pollutant (0..1).
SIGNATURE = {
    "industrial":   {"pm25": 0.6, "pm10": 0.5, "no2": 0.4, "so2": 1.0, "co": 0.3},
    "traffic":      {"pm25": 0.5, "pm10": 0.4, "no2": 1.0, "so2": 0.1, "co": 1.0},
    "construction": {"pm25": 0.3, "pm10": 1.0, "no2": 0.2, "so2": 0.0, "co": 0.1},
    "biomass":      {"pm25": 1.0, "pm10": 0.6, "no2": 0.2, "so2": 0.1, "co": 0.8},
}

# Pollutant scaling used only by the synthetic simulator (ug/m3, CO in mg/m3).
SIM_SCALE = {"pm25": 90.0, "pm10": 130.0, "no2": 80.0, "so2": 60.0, "co": 2.5}
SIM_BACKGROUND = {"pm25": 25.0, "pm10": 50.0, "no2": 15.0, "so2": 6.0, "co": 0.4}
