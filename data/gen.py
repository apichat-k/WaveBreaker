import numpy as np, json, csv, math
from datetime import datetime, timedelta, timezone
rng = np.random.default_rng(20261019)
TZ = timezone(timedelta(hours=7))
START = datetime(2026, 10, 12, 0, 0, tzinfo=TZ)
DAYS, DT = 7, 60                      # 1-minute records
n = DAYS * 1440
t = np.arange(n)
hour = (t % 1440) / 60.0
day = t // 1440
g = 9.81
# ---- assumptions (stated in metadata) ----
FETCH = 6000.0        # m, effective fetch over reservoir
DEPTH = 10.0          # m, water depth at breakwater
B, DRAFT = 4.0, 0.8   # m, floating breakwater width and draft

# ---- wind U10 (m/s): diurnal cycle x daily synoptic factor + gust AR(1) + squall ----
diurnal = 2.2 + 4.0 * np.clip(np.sin(np.pi * (hour - 8.5) / 13), 0, None) ** 1.3
syn_day = np.array([0.85, 1.05, 1.20, 0.95, 1.30, 0.80, 1.00])
syn = np.interp(t / 1440, np.arange(DAYS) + 0.5, syn_day)
gust = np.zeros(n); e = rng.normal(0, 0.35, n)
for i in range(1, n): gust[i] = 0.97 * gust[i-1] + e[i]
U = diurnal * syn + gust
sq0 = 4 * 1440 + 16 * 60 + 20          # squall: day 5 (16 Oct) 16:20-17:40
sq = np.exp(-0.5 * ((t - (sq0 + 25)) / 18.0) ** 2)
U = U + 10.5 * sq
U = np.clip(U, 0.6, None)

# ---- fetch-limited wave growth (SPM/CEM deep water) with lagged response ----
X = g * FETCH / U**2
Hs_eq = 0.0016 * np.sqrt(X) * U**2 / g
Tp_eq = 0.2857 * X ** (1/3) * U / g
Hs = np.zeros(n); Tp = np.zeros(n); Hs[0], Tp[0] = Hs_eq[0], Tp_eq[0]
a = 1 - math.exp(-DT / (12 * 60))    # ~12 min growth/decay time constant
for i in range(1, n):
    Hs[i] = Hs[i-1] + a * (Hs_eq[i] - Hs[i-1])
    Tp[i] = Tp[i-1] + a * (Tp_eq[i] - Tp[i-1])
HsA = np.clip(Hs * (1 + rng.normal(0, 0.05, n)) + 0.02, 0.03, None)
TpA = np.clip(Tp * (1 + rng.normal(0, 0.035, n)), 1.0, None)

# ---- floating breakwater transmission (Macagno 1953, finite depth) ----
def wavenumber(T, h):
    w = 2 * np.pi / T; k = w**2 / g
    for _ in range(40): k = w**2 / (g * np.tanh(k * h))
    return k
k = wavenumber(TpA, DEPTH)
Kt = 1 / np.sqrt(1 + (k * B * np.sinh(k * DEPTH) / (2 * np.cosh(k * DEPTH - k * DRAFT))) ** 2)
Kt = np.clip(Kt * (1 + rng.normal(0, 0.04, n)) + 0.03, 0.05, 1.0)   # +3 % leakage/diffraction
HsB = np.clip(HsA * Kt + np.abs(rng.normal(0, 0.006, n)), 0.01, None)
TpB = np.clip(TpA * (1.06 + rng.normal(0, 0.03, n)), 1.0, None)       # short components filtered out

# ---- surface water temperature (°C) ----
TwA = 29.1 - 0.09 * (t / 1440) + 0.65 * np.sin(2 * np.pi * (hour - 9.5) / 24) - 0.6 * sq + rng.normal(0, 0.03, n)
TwB = TwA + 0.08 + 0.15 * np.clip(np.sin(2 * np.pi * (hour - 9.5) / 24), 0, None) + rng.normal(0, 0.03, n)

# ---- telemetry gaps ----
offA = np.zeros(n, bool); offB = np.zeros(n, bool)
offB[3 * 1440 + 13 * 60 + 52: 3 * 1440 + 14 * 60 + 38] = True           # B: 15 Oct 13:52-14:37 (battery swap)
for s in rng.choice(np.arange(200, n - 200), 5, replace=False): offA[s: s + rng.integers(2, 5)] = True   # A: short packet loss

rows = []
for i in range(n):
    ts = (START + timedelta(seconds=int(i * DT))).isoformat()
    for b, off, H, T, C in (('A', offA, HsA, TpA, TwA), ('B', offB, HsB, TpB, TwB)):
        rows.append([ts, b, '' if off[i] else f'{H[i]:.3f}', '' if off[i] else f'{T[i]:.2f}', '' if off[i] else f'{C[i]:.2f}', 'OFFLINE' if off[i] else 'ONLINE', f'{U[i]:.2f}'])
with open('floating_solar_demo_sensor_data.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['timestamp', 'buoy', 'wave_height_m', 'period_s', 'temperature_c', 'status', 'wind_speed_ms']); w.writerows(rows)

def col(arr, off, d):
    return [None if off[i] else round(float(arr[i]), d) for i in range(n)]
meta = {
  'description': 'Synthetic demo data for the Floating Solar Digital Twin. Not measured data.',
  'start': START.isoformat(), 'interval_s': DT, 'count': n,
  'assumptions': {'fetch_m': FETCH, 'depth_m': DEPTH, 'breakwater_width_m': B, 'breakwater_draft_m': DRAFT,
                  'wave_model': 'SPM fetch-limited deep-water growth, 12-min lag', 'transmission_model': 'Macagno (1953) Kt + 3% leakage',
                  'events': ['Squall 2026-10-16 16:20-17:40', 'Buoy B offline 2026-10-15 13:52-14:37']},
  'wind': [round(float(x), 2) for x in U],
  'A': {'h': col(HsA, offA, 3), 'p': col(TpA, offA, 2), 'c': col(TwA, offA, 2)},
  'B': {'h': col(HsB, offB, 3), 'p': col(TpB, offB, 2), 'c': col(TwB, offB, 2)},
}
json.dump(meta, open('demo-data.json', 'w'), separators=(',', ':'))
ok = ~offA & ~offB
red = (1 - HsB / HsA)[ok] * 100
print('Hs A  min/mean/p99/max', HsA.min().round(3), HsA.mean().round(3), np.percentile(HsA, 99).round(3), HsA.max().round(3))
print('Tp A  min/mean/max', TpA.min().round(2), TpA.mean().round(2), TpA.max().round(2))
print('Hs B  mean/max', HsB.mean().round(3), HsB.max().round(3))
print('Reduction % min/mean/max', red.min().round(1), red.mean().round(1), red.max().round(1))
print('Wind min/mean/max', U.min().round(1), U.mean().round(1), U.max().round(1))
print('Temp A min/max', TwA.min().round(2), TwA.max().round(2))
print('offline A/B minutes', offA.sum(), offB.sum())
