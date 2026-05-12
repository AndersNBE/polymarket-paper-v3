"""
backtest_v3.py — Backtest of V3 strategy on historical data.

V3 changes from V1:
  1. dte >= 30 (no resolution gambling)
  2. Stop-loss: exit if |z| > entry_z + 1.5 OR if mid moves > 15¢ against us
  3. $30 stake instead of $100

Two phases:
  Phase A: Apply ONLY dte>=30 filter to existing V1 trades.
           This tests the critic's "resolution gambling" hypothesis.
  Phase B: Re-fetch price history for filtered trades; replay with stop-loss.
           This tests if stop-loss improves the strategy.
"""
import json
import datetime
import time
from pathlib import Path
import requests
import numpy as np

CLOB = "https://clob.polymarket.com"
STAKE = 30.0
GAS_PER_FILL = 0.15
FEE_RATES = {
    "sports_fees_v2": 0.03, "crypto_fees_v2": 0.07, "politics_fees": 0.04,
    "weather_fees": 0.05, "culture_fees": 0.05, "finance_prices_fees": 0.04,
    "tech_fees": 0.04, "economics_fees": 0.05, "mentions_fees": 0.04,
    "general_fees": 0.05, "crypto_15_min": 0.07, None: 0.05,
}
STOP_Z_EXTRA = 1.5
STOP_PRICE_MOVE = 0.15

trades_all = json.loads(Path("results_B_v3.json").read_text())
markets_local = Path("markets.json")
markets = {m["id"]: m for m in json.loads(markets_local.read_text())} if markets_local.exists() else {}

def annotate(t):
    m = markets.get(t["market_id"])
    t["fee_type"] = m.get("feeType") if m else None
    t["liquidity"] = float(m.get("liquidityNum") or 0) if m else 0
    t["dte"] = None
    t["yes_token"] = None
    if m and m.get("endDate"):
        try:
            end_ts = datetime.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            t["dte"] = (end_ts - t["entry_ts"]) / 86400
        except (ValueError, AttributeError):
            pass
    if m and m.get("clobTokenIds"):
        try:
            t["yes_token"] = (json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"])[0]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    return t

for t in trades_all:
    annotate(t)

def slip_for(p, liq):
    if liq >= 25000: b = 0.005
    elif liq >= 10000: b = 0.008
    elif liq >= 3000: b = 0.012
    elif liq >= 1000: b = 0.020
    else: b = 0.040
    if p < 0.15 or p > 0.85: b *= 1.5
    return b

def fee_for(p, sh, ft):
    return sh * FEE_RATES.get(ft, FEE_RATES[None]) * p * (1 - p)

def net_pnl(t, exit_price=None, ret_per_share=None):
    """Compute net $ PnL on STAKE."""
    if t["direction"] == 1:
        shares = STAKE / t["entry_price"]
    else:
        shares = STAKE / (1 - t["entry_price"])
    if ret_per_share is None:
        ret_per_share = t["ret_per_share"]
    if exit_price is None:
        exit_price = t["exit_price"]
    gross = shares * ret_per_share
    slip = slip_for(t["entry_price"], t["liquidity"])
    spread_cost = shares * slip * 2
    f_e = fee_for(t["entry_price"], shares, t["fee_type"])
    f_x = fee_for(exit_price, shares, t["fee_type"])
    gas = GAS_PER_FILL * 2
    return gross - spread_cost - f_e - f_x - gas

# V1 baseline filter
def v1_filter(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] >= 365 or t["dte"] < 0: return False
    if 7 <= t["dte"] < 30: return False  # blocks 7-30, allows <7
    return True

# V3 filter: dte >= 30
def v3_filter(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] < 30 or t["dte"] > 365: return False
    return True

v1 = [t for t in trades_all if v1_filter(t)]
v3_dte_only = [t for t in trades_all if v3_filter(t)]

print(f"Filtered trades:")
print(f"  V1 (original): {len(v1)}")
print(f"  V3 (dte>=30):  {len(v3_dte_only)}  ({len(v1)-len(v3_dte_only)} removed by dte filter)")

# ── Phase A: dte>=30 alone ──
print(f"\n{'='*70}")
print(f"PHASE A: dte>=30 filter only (no stop-loss yet)")
print(f"{'='*70}")

def summarize(label, ts):
    if not ts:
        print(f"\n  {label}: no trades")
        return
    pnls = np.array([net_pnl(t) for t in ts])
    print(f"\n  {label}: n={len(ts)}")
    print(f"    Mean PnL:     ${pnls.mean():+.2f}")
    print(f"    Median:       ${np.median(pnls):+.2f}")
    print(f"    Win rate:     {(pnls > 0).mean()*100:.1f}%")
    print(f"    Std:          ${pnls.std():.2f}")
    print(f"    Sharpe:       {pnls.mean()/pnls.std():.3f}" if pnls.std() > 0 else "")
    print(f"    Worst:        ${pnls.min():.2f}")
    print(f"    Best:         ${pnls.max():.2f}")
    print(f"    Total:        ${pnls.sum():+.2f}")
    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(pnls, len(pnls), replace=True).mean() for _ in range(5000)])
    print(f"    95% CI:       [${np.percentile(boot, 2.5):+.2f}, ${np.percentile(boot, 97.5):+.2f}]")
    print(f"    P(mean > 0):  {(boot > 0).mean()*100:.1f}%")

summarize("V1 baseline (kept for comparison)", v1)
summarize("V3 dte-only (drops <30d markets)", v3_dte_only)

# Critic's outlier test: remove top trades
print(f"\n--- Critic's outlier check: top trades removed ---")
for label, ts in [("V1", v1), ("V3 dte-only", v3_dte_only)]:
    if not ts: continue
    pnls = sorted([net_pnl(t) for t in ts], reverse=True)
    print(f"  {label} (n={len(pnls)}):")
    for top_n in [0, 1, 3, 5, 10]:
        if top_n >= len(pnls): break
        remaining = pnls[top_n:]
        mean = sum(remaining) / len(remaining)
        print(f"    Top {top_n} removed: mean=${mean:+.2f}  total=${sum(remaining):+.2f}")

# ── Phase B: Apply stop-loss via price-history replay ──
print(f"\n{'='*70}")
print(f"PHASE B: Apply stop-loss to V3 trades (replay price history)")
print(f"{'='*70}")
print(f"  Re-fetching price history for {len(v3_dte_only)} trades...")
print(f"  Stop conditions: z > entry_z+{STOP_Z_EXTRA}  OR  price moves >{STOP_PRICE_MOVE*100:.0f}¢ against us")

def fetch_history(token):
    try:
        r = requests.get(f"{CLOB}/prices-history",
                         params={"market": token, "fidelity": 60, "interval": "max"},
                         timeout=15)
        if r.status_code == 200:
            return r.json().get("history", [])
    except requests.RequestException:
        pass
    return None

def replay_with_stop(t):
    """Replay this trade with stop-loss enabled. Return new exit info."""
    if not t.get("yes_token"):
        return None
    hist = fetch_history(t["yes_token"])
    time.sleep(0.05)
    if not hist or len(hist) < 30:
        return None
    times = np.array([h["t"] for h in hist])
    prices = np.array([h["p"] for h in hist], dtype=float)
    # Find entry index
    entry_ts = t["entry_ts"]
    entry_idx = np.searchsorted(times, entry_ts)
    if entry_idx >= len(times) - 1:
        return None
    direction = t["direction"]
    entry_p = t["entry_price"]
    entry_z = abs(t["entry_z"])
    original_exit_idx = entry_idx + t["hold_hours"]
    if original_exit_idx >= len(prices): original_exit_idx = len(prices) - 1

    # Walk bar by bar, check stop-loss + natural exit
    WINDOW = 24
    EXIT_Z = 0.5
    MAX_HOLD = 48
    for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD + 1, len(prices))):
        w = prices[max(0, j - WINDOW):j]
        if len(w) < 2 or w.std() < 0.005: continue
        mu = w.mean()
        sd = w.std()
        z = (prices[j] - mu) / sd
        held = j - entry_idx

        # Stop-loss check 1: z intensifies
        if abs(z) > entry_z + STOP_Z_EXTRA:
            return {"exit_price": float(prices[j]), "hold_hours": int(held), "exit_reason": "stop_z_intensify",
                    "ret_per_share": direction * (float(prices[j]) - entry_p)}
        # Stop-loss check 2: price moves against
        adverse = direction * (prices[j] - entry_p)  # positive if in our favor
        if adverse < -STOP_PRICE_MOVE:
            return {"exit_price": float(prices[j]), "hold_hours": int(held), "exit_reason": "stop_price_adverse",
                    "ret_per_share": float(adverse)}  # adverse is already direction-adjusted
        # Natural exit
        if abs(z) < EXIT_Z:
            return {"exit_price": float(prices[j]), "hold_hours": int(held), "exit_reason": "z_revert",
                    "ret_per_share": direction * (float(prices[j]) - entry_p)}
        # Max hold
        if held >= MAX_HOLD:
            return {"exit_price": float(prices[j]), "hold_hours": int(held), "exit_reason": "max_hold",
                    "ret_per_share": direction * (float(prices[j]) - entry_p)}
    # Fallback to original exit
    return None

# Parallel-fetch with threads (need to batch sleeps)
from concurrent.futures import ThreadPoolExecutor
v3_with_stop = []
stop_count = 0
import sys
for i, t in enumerate(v3_dte_only):
    if i % 20 == 0:
        sys.stdout.write(f"\r  Processing {i}/{len(v3_dte_only)} ({stop_count} stops triggered)")
        sys.stdout.flush()
    new = replay_with_stop(t)
    if new is not None:
        merged = {**t, **new}
        v3_with_stop.append(merged)
        if new["exit_reason"].startswith("stop_"):
            stop_count += 1
    else:
        # Fallback: keep original trade
        v3_with_stop.append(t)

print(f"\n  Done. {stop_count} trades exited via stop-loss ({100*stop_count/max(1,len(v3_dte_only)):.0f}% of v3 trades)")

summarize("V3 with dte>=30 + stop-loss (full)", v3_with_stop)

# Final comparison
print(f"\n{'='*70}")
print(f"FINAL COMPARISON")
print(f"{'='*70}")
for label, ts in [("V1 (baseline)", v1), ("V3 dte>=30 only", v3_dte_only), ("V3 + stop-loss", v3_with_stop)]:
    if not ts: continue
    pnls = np.array([net_pnl(t) for t in ts])
    win = (pnls > 0).mean() * 100
    sharpe = pnls.mean() / pnls.std() if pnls.std() > 0 else 0
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(pnls, len(pnls), replace=True).mean() for _ in range(5000)])
    print(f"\n  {label}: n={len(ts)}")
    print(f"    Mean: ${pnls.mean():+.2f}  Win: {win:.1f}%  Sharpe: {sharpe:.3f}  Worst: ${pnls.min():.2f}  Total: ${pnls.sum():+.2f}")
    print(f"    95% CI: [${np.percentile(boot, 2.5):+.2f}, ${np.percentile(boot, 97.5):+.2f}]  P(>0): {(boot > 0).mean()*100:.0f}%")

print(f"\nVerdict logic:")
v3s_pnls = [net_pnl(t) for t in v3_with_stop] if v3_with_stop else []
v3d_pnls = [net_pnl(t) for t in v3_dte_only] if v3_dte_only else []
if v3s_pnls and v3d_pnls:
    v3s_mean = np.mean(v3s_pnls)
    v3d_mean = np.mean(v3d_pnls)
    v3s_p_gt_0 = (np.array([np.random.default_rng(42+i).choice(v3s_pnls, len(v3s_pnls), replace=True).mean() for i in range(1000)]) > 0).mean()
    if v3s_mean > 0 and v3s_p_gt_0 > 0.90:
        print(f"  ✓ V3 strategy survives critic's challenges with edge")
    elif v3s_mean > 0:
        print(f"  ⚠ V3 marginally positive but not statistically significant")
    else:
        print(f"  ✗ V3 strategy does NOT survive — critic was right about resolution gambling")
