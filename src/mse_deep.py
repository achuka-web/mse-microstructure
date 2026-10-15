"""
МХБ: 2-р шатны шинжилгээ (mse_analysis.py-ийн дараа).
mse_history.py-ийн гаргасан mse_<T>_<start>_<end>.csv файлуудыг уншина.

Ажиллуулах:
    python mse_deep.py 2010-01-01 2026-10-14

Гаралт (./report2/):
    09_events.csv     |өгөөж|>30% өдрүүд + хувьцаа бүрийн хамгийн муу/сайн 5 өдөр
                      → ногдол ашиг, хуваалт, өгөгдлийн алдааг ГАРААР шалгах жагсаалт
    10_spreads.csv    Абди–Ранальдогийн спред (high/low/close-оос), Ролл, сүүлийн 12 сар
    11_epps.csv       1, 5, 20 хоногийн өгөөжөөрх дундаж корреляц ба PC1-ийн хувь
    12_reversal.csv   "Өчигдөр унасныг ав" (зөвхөн long) стратегийн бохир/цэвэр өгөөж
    13_garch.csv      GARCH(1,1): тогтвортой байдал α+β, хагас задралын хугацаа
    epps.png, reversal.png

Шаардлагатай сан: pandas numpy scipy matplotlib
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize

START_DATE = "2010-01-01"
END_DATE = "2026-10-14"
JUMP = 0.30
MIN_OBS = 250
LIQ_VALUE = 5e6       # хөрвөх чадвартай: өдрийн медиан арилжаа ≥ 5 сая ₮
LIQ_DAYS = 0.90       # ба ажлын өдрийн ≥ 90%-д арилжаалагдсан
OUT = "report2"


def load(start, end):
    data = {}
    for fn in sorted(glob.glob(f"mse_*_{start}_{end}.csv")):
        t = os.path.basename(fn).split("_")[1]
        if t == "quality":
            continue
        df = pd.read_csv(fn, parse_dates=["date"]).sort_values("date")
        df = df.dropna(subset=["close"]).drop_duplicates("date")
        df = df[df["close"] > 0].set_index("date")
        if len(df) >= MIN_OBS:
            data[t] = df
    return data


# ---------------------------------------------------------------- 9. үйл явдал
def events(t, df):
    r = np.log(df["close"]).diff()
    gap = df.index.to_series().diff().dt.days
    tab = pd.DataFrame({"ret_pct": 100 * (np.exp(r) - 1), "gap_days": gap,
                        "close": df["close"], "prev_close": df["close"].shift(1),
                        "value_mnt": df["value"], "deals": df["deals"]}).dropna(subset=["ret_pct"])
    big = tab[np.abs(r.reindex(tab.index)) > np.log(1 + JUMP)].assign(kind="jump>30%")
    worst = tab.nsmallest(5, "ret_pct").assign(kind="worst5")
    best = tab.nlargest(5, "ret_pct").assign(kind="best5")
    out = pd.concat([big, worst, best])
    out = out[~out.index.duplicated()].reset_index().rename(columns={"index": "date"})
    out.insert(0, "ticker", t)
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    return out


# ---------------------------------------------------------------- 10. спред
def abdi_ranaldo(df):
    """Abdi & Ranaldo (2017): s² = 4·E[(c_t − η_t)(c_t − η_{t+1})], η = (ln H + ln L)/2."""
    c = np.log(df["close"])
    eta = (np.log(df["high"]) + np.log(df["low"])) / 2
    x = (c - eta) * (c - eta.shift(-1))
    m = 4 * x.dropna().mean()
    return np.sqrt(m) if m > 0 else 0.0


def roll(df):
    dc = np.log(df["close"]).diff()
    cov = dc.cov(dc.shift(1))
    return 2 * np.sqrt(-cov) if cov < 0 else np.nan


def spreads(t, df):
    last = df[df.index >= df.index.max() - pd.Timedelta(days=365)]
    return {"ticker": t,
            "ar_spread_pct": round(100 * abdi_ranaldo(df), 2),
            "ar_spread_last12m_pct": round(100 * abdi_ranaldo(last), 2),
            "roll_spread_pct": round(100 * roll(df), 2),
            "roll_last12m_pct": round(100 * roll(last), 2)}


# ---------------------------------------------------------------- 11. Эппсийн нөлөө
def grid_prices(data):
    """Ажлын өдрийн тор дээрх лог үнэ: арилжаагүй өдөр өмнөх үнийг хадгална,
    бүртгэгдэхээс өмнө NaN. 30%-иас их үсрэлтийг (хуваалт/алдаа) тэглэнэ."""
    lo = min(d.index.min() for d in data.values())
    hi = max(d.index.max() for d in data.values())
    grid = pd.bdate_range(lo, hi)
    P = {}
    for t, df in data.items():
        r = np.log(df["close"]).diff().fillna(0)
        r[np.abs(r) > np.log(1 + JUMP)] = 0
        lp = r.cumsum()
        P[t] = lp.reindex(grid.union(lp.index)).ffill().reindex(grid)
        P[t][grid < df.index.min()] = np.nan
    return pd.DataFrame(P)


def avg_corr_pc1(R, min_p):
    C = R.corr(min_periods=min_p)
    keep = C.notna().sum() > 1
    C = C.loc[keep, keep]
    off = C.values[~np.eye(len(C), dtype=bool)]
    A = C.fillna(0).to_numpy(copy=True); np.fill_diagonal(A, 1)
    lam = np.clip(np.linalg.eigvalsh(A), 0, None)
    return np.nanmean(off), lam.max() / lam.sum(), len(C)


def epps(LP, groups):
    rows = []
    for name, cols in groups.items():
        for h in (1, 5, 20):
            R = LP[cols].iloc[::h].diff()              # давхцахгүй h хоногийн өгөөж
            R = R.where(R != 0)                         # хоёулаа хөдлөөгүй өдрийг хасна
            mc, pc1, k = avg_corr_pc1(R, max(30, 300 // h))
            rows.append({"universe": name, "horizon_days": h, "n_stocks": k,
                         "avg_pair_corr": round(mc, 3), "pc1_share": round(pc1, 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 12. эсрэг хөдөлгөөн
def reversal(LP, cols, spread):
    """t өдөр унасан (r_t < 0) хувьцааг t-ийн хаалтаар авч, t+1-ийн хаалтаар зарна.
    МХБ-д богино борлуулалт бараг боломжгүй тул зөвхөн long. Зардал = бүтэн спред
    (авахдаа хагас, зарахдаа хагас)."""
    R = LP[cols].diff()
    sig = (R < 0) & R.notna()
    nxt = R.shift(-1)
    w = sig.div(sig.sum(axis=1), axis=0)
    gross = (w * nxt).sum(axis=1, min_count=1)
    cost = (w * pd.Series(spread).reindex(cols)).sum(axis=1)
    ok = sig.sum(axis=1) > 0
    g, c = gross[ok].dropna(), cost[ok].reindex(gross[ok].dropna().index)
    n = (R[cols].notna().sum(axis=1) > 0).sum() / ((LP.index[-1] - LP.index[0]).days / 365.25)

    def summ(x, label):
        return {"series": label, "days": len(x),
                "mean_daily_bp": round(1e4 * x.mean(), 1),
                "t_stat": round(x.mean() / x.std() * np.sqrt(len(x)), 2),
                "ann_sharpe": round(x.mean() / x.std() * np.sqrt(n), 2)}
    return [summ(g, "gross"), summ(g - c, "net_of_spread"),
            {"series": "avg_cost_bp", "mean_daily_bp": round(1e4 * c.mean(), 1)}], g, g - c


# ---------------------------------------------------------------- 13. GARCH
def garch11(r):
    r = (r - r.mean()).values
    v0 = r.var()

    def nll(p):
        om, a, b = p
        if om <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = np.empty_like(r); h[0] = v0
        for i in range(1, len(r)):
            h[i] = om + a * r[i - 1] ** 2 + b * h[i - 1]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)
    best = None
    for a0, b0 in ((0.05, 0.9), (0.15, 0.7), (0.3, 0.3)):
        res = optimize.minimize(nll, [v0 * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
                                options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    s, e = (a[0], a[1]) if len(a) == 2 else (START_DATE, END_DATE)
    os.makedirs(OUT, exist_ok=True)
    data = load(s, e)
    if not data:
        raise SystemExit(f"[!] mse_*_{s}_{e}.csv олдсонгүй.")

    ev = pd.concat([events(t, d) for t, d in data.items()], ignore_index=True)
    ev.to_csv(f"{OUT}/09_events.csv", index=False, date_format="%Y-%m-%d")

    sp = pd.DataFrame([spreads(t, d) for t, d in data.items()])
    sp.to_csv(f"{OUT}/10_spreads.csv", index=False)

    liquid = []
    for t, d in data.items():
        bd = len(pd.bdate_range(d.index.min(), d.index.max()))
        if d["value"].median() >= LIQ_VALUE and len(d) / bd >= LIQ_DAYS:
            liquid.append(t)
    print(f"[i] Хөрвөх чадвартай: {', '.join(liquid) or '—'}")

    LP = grid_prices(data)
    groups = {"all": list(data)}
    if len(liquid) >= 3:
        groups["liquid"] = liquid
    ep = epps(LP, groups)
    ep.to_csv(f"{OUT}/11_epps.csv", index=False)

    spread = (sp.set_index("ticker")["ar_spread_pct"] / 100).to_dict()
    rows, curves = [], {}
    for name, cols in groups.items():
        res, g, n = reversal(LP, cols, spread)
        for r_ in res:
            rows.append({"universe": name, **r_})
        curves[name] = (g, n)
    pd.DataFrame(rows).to_csv(f"{OUT}/12_reversal.csv", index=False)

    G = []
    for t in (liquid or list(data)[:5]):
        r = np.log(data[t]["close"]).diff().dropna()
        r = r[np.abs(r) <= np.log(1 + JUMP)]
        om, al, be = garch11(r)
        p = al + be
        G.append({"ticker": t, "omega": om, "alpha": round(al, 3), "beta": round(be, 3),
                  "persistence": round(p, 3), "at_bound": p >= 0.998,   # хязгаарт тулсан бол итгэх хэрэггүй
                  "half_life_days": round(np.log(0.5) / np.log(p), 1) if 0 < p < 1 else np.nan,
                  "uncond_ann_vol": round(np.sqrt(om / (1 - p) * 250), 3) if p < 1 else np.nan})
    pd.DataFrame(G).to_csv(f"{OUT}/13_garch.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, g in ep.groupby("universe"):
            ax.plot(g["horizon_days"], g["avg_pair_corr"], "o-", label=name)
        ax.set_xscale("log"); ax.set_xticks([1, 5, 20]); ax.set_xticklabels(["1 өдөр", "7 хоног", "сар"])
        ax.set_ylabel("Дундаж хос корреляц"); ax.set_title("Эппсийн нөлөө"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/epps.png", dpi=150); plt.close()
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, (g, n) in curves.items():
            ax.plot(g.cumsum(), label=f"{name}: бохир")
            ax.plot(n.cumsum(), "--", label=f"{name}: спредийн дараа")
        ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("Хуримтлагдсан лог өгөөж")
        ax.set_title("Өчигдөр унасныг ав (long-only)"); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{OUT}/reversal.png", dpi=150); plt.close()
    except Exception as ex:
        print(f"[!] График гаргасангүй: {ex}")

    pd.set_option("display.width", 200)
    print("\n=== СПРЕД ===\n" + sp.to_string(index=False))
    print("\n=== ЭППС ===\n" + ep.to_string(index=False))
    print("\n=== ЭСРЭГ ХӨДӨЛГӨӨН ===\n" + pd.DataFrame(rows).to_string(index=False))
    print("\n=== GARCH ===\n" + pd.DataFrame(G).to_string(index=False))
    print(f"\n[+] ./{OUT}/ — 09_events.csv-г гараар шалга.")


if __name__ == "__main__":
    main()
