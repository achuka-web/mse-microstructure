"""
МХБ-ийн TOP-20 хувьцааны цувааны шинжилгээ (mse_history.py-ийн CSV-үүд дээр).

Ажиллуулах (CSV-үүд байгаа хавтсанд):
    python mse_analysis.py 2010-01-01 2026-10-14
Spyder/Jupyter: доорх START_DATE, END_DATE-ийг засаад Run.

Гаралт (./report/ хавтсанд):
    01_quality.csv       өгөгдлийн шалгалт: үсрэлт, зогсонги ханш, high/low-ийн утга
    02_liquidity.csv     хөрвөх чадвар: арилжаалагдсан өдөр, дундаж дүн, Амихуд, Ролл
    03_returns.csv       өгөөжийн тархалт: хэлбэлзэл, хазайлт, эксцесс, хамгийн их уналт
    04_dependence.csv    автокорреляц, Льюнг–Бокс, дисперсийн харьцаа, |r|-ийн кластер
    05_weekday.csv       гарагийн нөлөө
    06_corr.csv          синхрон корреляцын матриц (зөвхөн хоёулаа арилжаалагдсан өдрүүд)
    07_pca.csv           PCA: зах зээлийн хүчин зүйлийн жин
    08_beta.csv          энгийн ба Димсоны бета (сийрэг арилжааны засвартай)
    *.png                график

Шаардлагатай сан:  pip install pandas numpy scipy matplotlib
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

START_DATE = "2010-01-01"
END_DATE = "2026-10-14"
MIN_OBS = 250            # үүнээс цөөн арилжаатай хувьцааг цувааны шинжилгээнээс хасна
JUMP = 0.30              # |өгөөж| > 30% бол хуваалт/алдаа сэжигтэй
OUT = "report"
LIQ_VALUE = 5e6         # индексийн гишүүн: өдрийн медиан арилжаа ≥ 5 сая ₮
LIQ_DAYS = 90           # ба арилжаалагдсан өдөр ≥ 90%


# ---------------------------------------------------------------- ачаалах
def load(start, end):
    data = {}
    for fn in sorted(glob.glob(f"mse_*_{start}_{end}.csv")):
        t = os.path.basename(fn).split("_")[1]
        if t == "quality":
            continue
        df = pd.read_csv(fn, parse_dates=["date"]).sort_values("date")
        df = df.dropna(subset=["close"]).drop_duplicates("date")
        df = df[df["close"] > 0]
        if len(df):
            data[t] = df.set_index("date")
    return data


# ---------------------------------------------------------------- 1. чанар
def quality(t, df):
    c = df["close"]
    r = np.log(c).diff()
    gap = df.index.to_series().diff().dt.days
    hl_ok = df[["high", "low"]].notna().all(axis=1)
    inside = ((df["low"] <= df[["open", "close"]].min(axis=1)) &
              (df["high"] >= df[["open", "close"]].max(axis=1)))[hl_ok]
    width = ((df["high"] - df["low"]) / df["close"])[hl_ok]
    return {
        "ticker": t, "rows": len(df),
        "first": df.index.min().date(), "last": df.index.max().date(),
        "median_gap_days": gap.median(), "max_gap_days": gap.max(),
        "n_jumps_gt30pct": int((r.abs() > np.log(1 + JUMP)).sum()),
        "pct_zero_return": round(100 * (r == 0).mean(), 1),
        "n_weekend_rows": int((df.index.dayofweek >= 5).sum()),
        "pct_zero_volume": round(100 * (df["volume"].fillna(0) == 0).mean(), 1),
        # high/low нь бодит арилжааны хүрээ бол open, close үргэлж дотор нь байна;
        # үнийн хязгаарын (±%) зурвас бол өргөн нь бараг тогтмол байна
        "pct_oc_inside_hl": round(100 * inside.mean(), 1) if len(inside) else np.nan,
        "hl_width_median": round(width.median(), 4) if len(width) else np.nan,
        "hl_width_cv": round(width.std() / width.mean(), 3) if len(width) and width.mean() else np.nan,
    }


# ---------------------------------------------------------------- 2. хөрвөх чадвар
def liquidity(t, df, bdays):
    r = df["close"].pct_change()
    val = df["value"].astype(float)
    amihud = (r.abs() / val.replace(0, np.nan)).dropna()
    dc = np.log(df["close"]).diff()                 # лог үнээр: үнийн түвшин өөрчлөгдөхөд тогтвортой
    cov = dc.cov(dc.shift(1))
    roll = 2 * np.sqrt(-cov) if cov < 0 else np.nan   # Ролл: cov ≥ 0 бол тодорхойлогдохгүй
    return {
        "ticker": t,
        "pct_days_traded": round(100 * len(df) / bdays, 1),
        "median_value_mnt": float(val.median()),
        "median_deals": float(df["deals"].median()),
        "amihud_x1e9": round(1e9 * amihud.median(), 3) if len(amihud) else np.nan,
        "roll_spread_pct": round(100 * roll, 2),
    }


# ---------------------------------------------------------------- 3. өгөөж
def returns_stats(t, r, n_years):
    mdd = (np.exp(r.cumsum()) / np.exp(r.cumsum()).cummax() - 1).min()
    ann = len(r) / n_years                     # жилд арилжаалагдсан өдрийн бодит тоо
    return {
        "ticker": t, "n": len(r), "obs_per_year": round(ann, 1),
        "ann_return_log": round(r.mean() * ann, 4),
        "ann_vol": round(r.std() * np.sqrt(ann), 4),
        "skew": round(stats.skew(r), 2), "excess_kurt": round(stats.kurtosis(r), 2),
        "jarque_bera_p": stats.jarque_bera(r).pvalue,
        "max_drawdown": round(mdd, 3),
        "pct_abs_gt_3sd": round(100 * (np.abs(r - r.mean()) > 3 * r.std()).mean(), 2),
    }


# ---------------------------------------------------------------- 4. хамаарал
def ljung_box(x, lags=10):
    x = x - x.mean(); n = len(x); d = (x ** 2).sum()
    rho = np.array([(x[k:] * x[:-k]).sum() / d for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return rho, q, 1 - stats.chi2.cdf(q, lags)


def variance_ratio(r, q):
    """Ло–МакКинлэйгийн дисперсийн харьцаа, гетероскедастик тэсвэртэй z."""
    r = r.values; n = len(r); mu = r.mean()
    rq = pd.Series(r).rolling(q).sum().dropna().values
    vr = np.sum((rq - q * mu) ** 2) / (n * q) / (np.sum((r - mu) ** 2) / n)
    d = (r - mu) ** 2
    theta = sum((2 * (q - j) / q) ** 2 *
                np.sum(d[j:] * d[:-j]) / np.sum(d) ** 2 * n for j in range(1, q))
    return vr, np.sqrt(n) * (vr - 1) / np.sqrt(theta) if theta > 0 else np.nan


def dependence(t, r):
    rho, q, p = ljung_box(r.values, 10)
    rho_a, qa, pa = ljung_box(np.abs(r.values), 10)
    out = {"ticker": t, "acf1": round(rho[0], 3), "acf2": round(rho[1], 3),
           "ljungbox10_p": p, "acf1_abs": round(rho_a[0], 3), "ljungbox10_abs_p": pa}
    for q_ in (2, 5, 10):
        vr, z = variance_ratio(r, q_)
        out[f"VR{q_}"], out[f"VR{q_}_z"] = round(vr, 3), round(z, 2)
    return out


def weekday(t, r):
    g = r.groupby(r.index.dayofweek)
    res = {"ticker": t}
    for d, s in g:
        name = ["Да", "Мя", "Лх", "Пү", "Ба", "Бя", "Ня"][d]
        res[name] = round(10000 * s.mean(), 1)          # б.н. (0.01%)
        if d >= 5:
            res[f"n_{name}"] = len(s)
    # амралтын өдрийн цөөн ажиглалтыг тестэд оруулахгүй
    res["kruskal_p"] = stats.kruskal(*[s.values for d, s in g if d < 5 and len(s) > 30]).pvalue
    return res


# ---------------------------------------------------------------- 6–8. хөндлөн огтлол
def nearest_psd(c):
    w, v = np.linalg.eigh(c)
    c2 = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    s, e = (a[0], a[1]) if len(a) == 2 else (START_DATE, END_DATE)
    os.makedirs(OUT, exist_ok=True)
    data = load(s, e)
    if not data:
        raise SystemExit(f"[!] mse_*_{s}_{e}.csv файл олдсонгүй.")
    print(f"[+] {len(data)} хувьцаа ачааллаа: {', '.join(data)}")

    Q, L, R, D, W, rets = [], [], [], [], [], {}
    for t, df in data.items():
        bd = len(pd.bdate_range(df.index.min(), df.index.max()))
        Q.append(quality(t, df)); L.append(liquidity(t, df, bd))
        r = np.log(df["close"]).diff().dropna()
        r = r[r.abs() <= np.log(1 + JUMP)]          # хуваалт/алдааг хасна (01-ээс шалга)
        rets[t] = r
        if len(r) >= MIN_OBS:
            yrs = (df.index.max() - df.index.min()).days / 365.25
            R.append(returns_stats(t, r, yrs)); D.append(dependence(t, r)); W.append(weekday(t, r))

    pd.DataFrame(Q).to_csv(f"{OUT}/01_quality.csv", index=False)
    liq = pd.DataFrame(L).sort_values("median_value_mnt", ascending=False)
    liq.to_csv(f"{OUT}/02_liquidity.csv", index=False)
    pd.DataFrame(R).to_csv(f"{OUT}/03_returns.csv", index=False)
    pd.DataFrame(D).to_csv(f"{OUT}/04_dependence.csv", index=False)
    pd.DataFrame(W).to_csv(f"{OUT}/05_weekday.csv", index=False)

    # 6. синхрон корреляц: хоёулаа арилжаалагдсан өдрүүдээр (pairwise)
    big = [t for t in rets if len(rets[t]) >= MIN_OBS]
    panel = pd.DataFrame({t: rets[t] for t in big})
    corr = panel.corr(min_periods=100)
    corr.to_csv(f"{OUT}/06_corr.csv")
    C = corr.fillna(0).to_numpy(copy=True); np.fill_diagonal(C, 1)
    w = np.linalg.eigvalsh(C)
    print(f"[i] Корреляцын матрицын хамгийн бага хувийн утга: {w.min():.4f}"
          + ("  → эерэг хагас тодорхойлогдсон БИШ, засвар хийв" if w.min() < 0 else ""))
    Cp = nearest_psd(C) if w.min() < 0 else C

    # 7. PCA
    lam, V = np.linalg.eigh(Cp); lam, V = lam[::-1], V[:, ::-1]
    v1 = V[:, 0] * np.sign(V[:, 0].sum())
    pca = pd.DataFrame({"ticker": big, "pc1_loading": v1.round(3), "pc2_loading": V[:, 1].round(3)})
    pca.attrs["explained"] = lam / lam.sum()
    pca.to_csv(f"{OUT}/07_pca.csv", index=False)
    print(f"[i] PC1 нийт дисперсийн {100*lam[0]/lam.sum():.1f}%, PC2 {100*lam[1]/lam.sum():.1f}%")

    # 8. бета: тэнцүү жинтэй индекс, энгийн ба Димсоны (−1, 0, +1 хоцрогдол)
    # сийрэг арилжаа: хувьцааны өгөөж хэд хоногийг хамардаг тул индексийн өөрчлөлтийг
    # яг тэр хугацаанд нь нийлбэрлэж харьцуулна
    sel = (liq["median_value_mnt"] >= LIQ_VALUE) & (liq["pct_days_traded"] >= LIQ_DAYS)
    liquid = [t for t in liq.loc[sel, "ticker"] if t in big] or big[:5]
    print(f"[i] Индекс (тэнцүү жин, хөрвөх чадвартай): {', '.join(liquid)}")
    B = []
    for t in big:
        # тухайн хувьцааг индексээс хасна (өөрөөсөө регресслэх хазайлтаас сэргийлэх)
        members = [u for u in liquid if u != t]
        idx_lvl = panel[members].mean(axis=1, skipna=True).fillna(0).cumsum()
        y = panel[t].dropna()
        lv = idx_lvl.reindex(y.index)
        x0 = lv.diff()
        X = pd.concat([x0.shift(1), x0, x0.shift(-1)], axis=1)
        m = pd.concat([y, X], axis=1).dropna()
        if len(m) < MIN_OBS:
            continue
        b0 = np.polyfit(m.iloc[:, 2], m.iloc[:, 0], 1)[0]
        Xm = np.c_[np.ones(len(m)), m.iloc[:, 1:].values]
        coef = np.linalg.lstsq(Xm, m.iloc[:, 0].values, rcond=None)[0]
        B.append({"ticker": t, "beta_simple": round(b0, 3), "beta_dimson": round(coef[1:].sum(), 3),
                  "lag_share": round(coef[1] / coef[1:].sum(), 3) if coef[1:].sum() else np.nan,
                  "in_index": t in liquid})
    pd.DataFrame(B).to_csv(f"{OUT}/08_beta.csv", index=False)

    # график
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(liq["ticker"], liq["median_value_mnt"] / 1e6, color="#2f5d8a")
        ax.set_ylabel("Өдрийн медиан арилжаа, сая ₮"); ax.set_title("Хөрвөх чадвар")
        plt.xticks(rotation=60); plt.tight_layout(); plt.savefig(f"{OUT}/liquidity.png", dpi=150); plt.close()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(Cp, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(big))); ax.set_xticklabels(big, rotation=90, fontsize=7)
        ax.set_yticks(range(len(big))); ax.set_yticklabels(big, fontsize=7)
        plt.colorbar(im); ax.set_title("Синхрон корреляц"); plt.tight_layout()
        plt.savefig(f"{OUT}/corr.png", dpi=150); plt.close()
    except Exception as ex:
        print(f"[!] График гаргасангүй: {ex}")

    pd.set_option("display.width", 200)
    print("\n=== ХӨРВӨХ ЧАДВАР (дээд 10) ===")
    print(liq.head(10).to_string(index=False))
    if D:
        print("\n=== ХАМААРАЛ ===")
        print(pd.DataFrame(D)[["ticker", "acf1", "ljungbox10_p", "acf1_abs", "VR5", "VR5_z"]].to_string(index=False))
    print(f"\n[+] Үр дүн: ./{OUT}/")


if __name__ == "__main__":
    main()
