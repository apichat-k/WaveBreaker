"""Synthetic demo data for the Floating Solar Digital Twin (NOT measured data).

Run:  python3 gen.py   -> writes floating_solar_demo_sensor_data.csv and ../demo-data.json
"""
import numpy as np, json, csv, math, os
from datetime import datetime, timedelta, timezone

rng = np.random.default_rng(20261019)
TZ = timezone(timedelta(hours=7))
START = datetime(2026, 10, 12, 0, 0, tzinfo=TZ)
DAYS, DT = 7, 60                       # 7 days of 1-minute records
n = DAYS * 1440
t = np.arange(n)
hour = (t % 1440) / 60.0
g = 9.81

# ---- demo targets / assumptions ----
HS_MIN, HS_MAX = 1.0, 1.5              # significant wave height before breakwater (m)
ER_MIN, ER_MAX = 0.30, 0.50            # wave energy reduction band (1 - Kt^2)
STEEPNESS = 1 / 27                     # Hs / Lp, wind sea
DEPTH, B, DRAFT = 10.0, 7.0, 1.0       # m: water depth, breakwater width and draft

def sig(x): return 1 / (1 + np.exp(-x))
def ar1(sd, rho):
    out = np.zeros(n); e = rng.normal(0, sd, n)
    for i in range(1, n): out[i] = rho * out[i-1] + e[i]
    return out

# ---- Buoy A significant wave height ----
diurnal = 0.11 * np.clip(np.sin(np.pi * (hour - 9) / 12), -0.4, None)        # afternoon sea breeze
syn_day = np.array([-0.08, 0.02, 0.10, -0.03, 0.12, -0.10, 0.00])
syn = np.interp(t / 1440, np.arange(DAYS) + 0.5, syn_day)
sq0 = 4 * 1440 + 16 * 60 + 45                                                  # squall 16 Oct ~16:20-17:40
squall = 0.32 * np.exp(-0.5 * ((t - sq0) / 22.0) ** 2)
x = 1.24 + diurnal + syn + squall + ar1(0.012, 0.985)
HsA = HS_MIN + (HS_MAX - HS_MIN) * sig((x - 1.25) / 0.12)                     # smooth bound to 1.0-1.5 m
HsA = np.clip(HsA * (1 + rng.normal(0, 0.012, n)), HS_MIN, HS_MAX)

# ---- peak period from wave steepness ----
Lp = HsA / (STEEPNESS * (1 + ar1(0.004, 0.98)))
TpA = np.sqrt(2 * np.pi * Lp / g) * (1 + rng.normal(0, 0.015, n))

# ---- floating breakwater transmission (Macagno 1953), bounded to the demo band ----
def wavenumber(T, h):
    w = 2 * np.pi / T; k = w**2 / g
    for _ in range(50): k = w**2 / (g * np.tanh(k * h))
    return k
k = wavenumber(TpA, DEPTH)
Kt_raw = 1 / np.sqrt(1 + (k * B * np.sinh(k * DEPTH) / (2 * np.cosh(k * DEPTH - k * DRAFT))) ** 2)
ER_raw = 1 - Kt_raw ** 2
ER = ER_MIN + (ER_MAX - ER_MIN) * sig((ER_raw - np.median(ER_raw)) / (0.6 * ER_raw.std()))
ER = np.clip(ER + rng.normal(0, 0.008, n), ER_MIN, ER_MAX)
HsB = HsA * np.sqrt(1 - ER)
TpB = TpA * (1.03 + rng.normal(0, 0.01, n))

# ---- context: wind and water temperature ----
U = np.clip(6.5 + (HsA - 1.0) * 13 + ar1(0.25, 0.95), 3, None)
TwA = 29.1 - 0.09 * (t / 1440) + 0.65 * np.sin(2 * np.pi * (hour - 9.5) / 24) - 2.0 * squall + rng.normal(0, 0.03, n)
TwB = TwA + 0.08 + 0.15 * np.clip(np.sin(2 * np.pi * (hour - 9.5) / 24), 0, None) + rng.normal(0, 0.03, n)

# ---- telemetry gaps ----
offA = np.zeros(n, bool); offB = np.zeros(n, bool)
offB[3 * 1440 + 13 * 60 + 52: 3 * 1440 + 14 * 60 + 38] = True                 # B: 15 Oct 13:52-14:37
for s in rng.choice(np.arange(200, n - 200), 5, replace=False): offA[s: s + rng.integers(2, 5)] = True

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, 'floating_solar_demo_sensor_data.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['timestamp', 'buoy', 'wave_height_m', 'period_s', 'temperature_c', 'status', 'wind_speed_ms'])
    for i in range(n):
        ts = (START + timedelta(seconds=int(i * DT))).isoformat()
        for b, off, H, T, C in (('A', offA, HsA, TpA, TwA), ('B', offB, HsB, TpB, TwB)):
            w.writerow([ts, b, '' if off[i] else f'{H[i]:.3f}', '' if off[i] else f'{T[i]:.2f}',
                        '' if off[i] else f'{C[i]:.2f}', 'OFFLINE' if off[i] else 'ONLINE', f'{U[i]:.2f}'])

def col(arr, off, d): return [None if off[i] else round(float(arr[i]), d) for i in range(n)]
meta = {
    'description': 'Synthetic demo data for the Floating Solar Digital Twin. Not measured data.',
    'start': START.isoformat(), 'interval_s': DT, 'count': n,
    'assumptions': {'hs_before_m': [HS_MIN, HS_MAX], 'energy_reduction': [ER_MIN, ER_MAX], 'steepness': round(STEEPNESS, 4),
                    'depth_m': DEPTH, 'breakwater_width_m': B, 'breakwater_draft_m': DRAFT,
                    'transmission_model': 'Macagno (1953) trend, bounded to the energy-reduction band',
                    'events': ['Squall 2026-10-16 16:20-17:40', 'Buoy B offline 2026-10-15 13:52-14:37']},
    'wind': [round(float(v), 2) for v in U],
    'A': {'h': col(HsA, offA, 3), 'p': col(TpA, offA, 2), 'c': col(TwA, offA, 2)},
    'B': {'h': col(HsB, offB, 3), 'p': col(TpB, offB, 2), 'c': col(TwB, offB, 2)},
}
json.dump(meta, open(os.path.join(here, '..', 'demo-data.json'), 'w'), separators=(',', ':'))

ok = ~offA & ~offB
er = (1 - (HsB / HsA) ** 2)[ok] * 100
print('Hs A  min/p5/mean/p95/max', *np.round([HsA.min(), np.percentile(HsA, 5), HsA.mean(), np.percentile(HsA, 95), HsA.max()], 3))
print('Hs B  min/mean/max', *np.round([HsB.min(), HsB.mean(), HsB.max()], 3))
print('Tp A  min/mean/max', *np.round([TpA.min(), TpA.mean(), TpA.max()], 2))
print('Energy reduction % min/p5/mean/p95/max', *np.round([er.min(), np.percentile(er, 5), er.mean(), np.percentile(er, 95), er.max()], 1))
print('Height reduction % min/max', *np.round([(1 - np.sqrt(1 - er.max() / 100)) * 100, (1 - np.sqrt(1 - er.min() / 100)) * 100][::-1], 1))
print('Wind min/max', U.min().round(1), U.max().round(1), ' Temp A', TwA.min().round(2), TwA.max().round(2))
