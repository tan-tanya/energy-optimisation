"""
Import-only. Maps each Met Office climatological district to a representative city and weather station.

Met Office district names and their external identifiers:
  - icao          → data/hdd/{icao}_HDD_15.5C.csv   (Met Office heating degree-days)
  - ukcp_region   → data/climateprojections/UKCP_{ukcp_region}.csv (UKCP18 RCP8.5 temperature anomalies)
  - sunshine_file → data/sunlighthours/{sunshine_file}  (Met Office seasonal sunshine hours)
  - lat / lon     → ERA5 grid point for the representative station (api_temperature_profiles.py;
                    Open-Meteo historical hourly temperature_2m → data/api_temperature_profiles.xlsx)
"""

# One row per Met Office district. (city = the representative ICAO weather station's location;
# lat/lon = station coordinates, used to pull ERA5 hourly temperature for the diurnal shape.)
DISTRICTS = {
    "East Anglia":              {"icao": "EGSH", "ukcp_region": "East England",        "sunshine_file": "East_Anglia.txt",              "city": "Norwich",           "lat": 52.6758, "lon":  1.2828},
    "England E and NE":         {"icao": "EGNT", "ukcp_region": "North East England",  "sunshine_file": "England_E_and_NE.txt",         "city": "Newcastle",         "lat": 55.0375, "lon": -1.6917},
    "England NW and N Wales":   {"icao": "EGCC", "ukcp_region": "North West England",  "sunshine_file": "England_NW_and_N_Wales.txt",   "city": "Manchester",        "lat": 53.3537, "lon": -2.2750},
    "England SE and Central S": {"icao": "EGLL", "ukcp_region": "South East England",  "sunshine_file": "England_SE_and_Central_S.txt", "city": "London Heathrow",   "lat": 51.4706, "lon": -0.4619},
    "England SW and S Wales":   {"icao": "EGTE", "ukcp_region": "South West England",  "sunshine_file": "England_SW_and_S_Wales.txt",   "city": "Exeter",            "lat": 50.7344, "lon": -3.4139},
    "Midlands":                 {"icao": "EGBB", "ukcp_region": "East Midlands",       "sunshine_file": "Midlands.txt",                 "city": "Birmingham",        "lat": 52.4539, "lon": -1.7480},
    "Scotland E":               {"icao": "EGPH", "ukcp_region": "Eastern Scotland",    "sunshine_file": "Scotland_E.txt",               "city": "Edinburgh",         "lat": 55.9500, "lon": -3.3725},
    "Scotland N":               {"icao": "EGPE", "ukcp_region": "Northern Scotland",   "sunshine_file": "Scotland_N.txt",               "city": "Inverness",         "lat": 57.5425, "lon": -4.0475},
    "Scotland W":               {"icao": "EGPK", "ukcp_region": "Western Scotland",    "sunshine_file": "Scotland_W.txt",               "city": "Glasgow Prestwick", "lat": 55.5094, "lon": -4.5867},
}

DISTRICT_STATIONS = {d: r["icao"]          for d, r in DISTRICTS.items()}   # district     -> ICAO station
UKCP_TO_DISTRICT  = {r["ukcp_region"]: d   for d, r in DISTRICTS.items()}   # UKCP region  -> district
SUNSHINE_FILE     = {d: r["sunshine_file"] for d, r in DISTRICTS.items()}   # district     -> sunshine txt filename
DISTRICT_LATLON   = {d: (r["lat"], r["lon"]) for d, r in DISTRICTS.items()} # district     -> (lat, lon)

__all__ = ["DISTRICTS", "DISTRICT_STATIONS", "UKCP_TO_DISTRICT", "SUNSHINE_FILE", "DISTRICT_LATLON"]
