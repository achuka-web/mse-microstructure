"""
МХБ: үнийн хязгаарт (±15%) хүрсэн өдрүүдийн үйл явдлын судалгаа (3-р шат).

Санаа: ханш тухайн өдрийн хязгаарт хүрвэл "жинхэнэ" үнэдээ нэг өдөрт хүрч чадахгүй,
тиймээс маргааш нь ижил чиглэлд үргэлжлэх (continuation) хандлагатай байж болно.
Хаалтын ханшаар авах боломжгүй (хязгаарт хүрсэн өдөр санал дуусдаг) тул ДАРААГИЙН
ӨДРИЙН НЭЭЛТЭЭР орж, k өдрийн дараах хаалтаар гарна гэж тооцно.

Ажиллуулах:
    python mse_limits.py 2010-01-01 2026-10-14
Гаралт (./report3/):
    14_limit_rule.csv     жил бүрийн |өгөөж|-ийн дээд утга, ±14.9%-аас дээш өдрийн тоо
    15_limit_runs.csv     дараалсан хязгаарын өдрүүдийн урт
    16_limit_events.csv   хязгаарын өдөр бүр: дараагийн нээлтийн цоорхой, k=1,2,5 өгөөж
    17_limit_summary.csv  дундаж, t, ялах магадлал; спредийн дараах цэвэр өгөөж
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

START_DATE = "2010-01-01"
END_DATE = "2026-10-14"
LIMIT = 0.149          # |энгийн өгөөж| ≥ 14.9% бол хязгаарт хүрсэн гэж үзнэ
JUMP = 0.30
MIN_OBS = 250
LIQ_VALUE = 5e6
OUT = "report3"
H = (1, 2, 5)


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


def ar_spread(df):
    c = np.log(df["close"]); eta = (np.log(df["high"]) + np.log(df["low"])) / 2
    m = 4 * ((c - eta) * (c - eta.shift(-1))).dropna().mean()
    return np.sqrt(m) if m > 0 else 0.0


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    s, e = (a[0], a[1]) if len(a) == 2 else (START_DATE, END_DATE)
    os.makedirs(OUT, exist_ok=True)
    data = load(s, e)
    if not data:
        raise SystemExit(f"[!] mse_*_{s}_{e}.csv олдсонгүй.")

    rule, runs, ev = [], [], []
    for t, df in data.items():
        df = df.copy()
        r = df["close"].pct_change()
        ok = r.abs() <= JUMP                            # хуваалтын өдрийг хасна
        df["r"] = r.where(ok)
        df["year"] = df.index.year
        for y, g in df.groupby("year"):
            rr = g["r"].dropna()
            if len(rr):
                rule.append({"ticker": t, "year": y, "max_up_pct": round(100 * rr.max(), 2),
                             "max_dn_pct": round(100 * rr.min(), 2),
                             "n_ge_limit": int((rr.abs() >= LIMIT).sum()), "n_days": len(rr)})
        side = np.sign(df["r"]).where(df["r"].abs() >= LIMIT, 0).fillna(0)
        # дараалсан хязгаарын өдрүүд
        blk = (side != side.shift()).cumsum()
        for _, g in side.groupby(blk):
            if g.iloc[0] != 0:
                runs.append({"ticker": t, "start": g.index[0].date(), "side": int(g.iloc[0]),
                             "length": len(g)})
        sp = ar_spread(df)
        liquid = df["value"].median() >= LIQ_VALUE
        o, c = df["open"].values, df["close"].values
        idx = np.where(side.values != 0)[0]
        for i in idx:
            if i + 1 >= len(df):
                continue
            row = {"ticker": t, "date": df.index[i].date(), "side": int(side.iloc[i]),
                   "liquid": liquid, "spread": sp,
                   "close_eq_high": bool(df["close"].iloc[i] >= df["high"].iloc[i]),
                   "value_mnt": df["value"].iloc[i],
                   "gap_days_next": (df.index[i + 1] - df.index[i]).days,
                   # маргаашийн нээлт өнөөдрийн хаалтаас хэр зөрсөн бэ
                   "next_open_gap": o[i + 1] / c[i] - 1,
                   "cc_next": c[i + 1] / c[i] - 1}
            for k in H:
                j = i + k
                # дараагийн өдрийн нээлтээр орж, k-р арилжааны өдрийн хаалтаар гарна
                row[f"oc_{k}"] = c[j] / o[i + 1] - 1 if j < len(df) and o[i + 1] > 0 else np.nan
            ev.append(row)

    pd.DataFrame(rule).to_csv(f"{OUT}/14_limit_rule.csv", index=False)
    R = pd.DataFrame(runs)
    R.to_csv(f"{OUT}/15_limit_runs.csv", index=False)
    E = pd.DataFrame(ev)
    if E.empty:
        print("[!] Хязгаарт хүрсэн өдөр олдсонгүй."); return
    # ±30%-аас их = хуваалт, алдаа: хасна
    for k in H:
        E.loc[E[f"oc_{k}"].abs() > JUMP * 2, f"oc_{k}"] = np.nan
    E.to_csv(f"{OUT}/16_limit_events.csv", index=False)

    rows = []
    for (side, liq), g in E.groupby(["side", "liquid"]):
        for k in H:
            x = g[f"oc_{k}"].dropna() * side          # чиглэлээ дагасан өгөөж
            if len(x) < 10:
                continue
            net = x - g.loc[x.index, "spread"]          # авах+зарах = бүтэн спред
            rows.append({"side": "дээд хязгаар" if side > 0 else "доод хязгаар",
                         "liquid": liq, "hold_days": k, "n": len(x),
                         "mean_pct": round(100 * x.mean(), 2),
                         "t_stat": round(x.mean() / x.std() * np.sqrt(len(x)), 2),
                         "hit_rate": round((x > 0).mean(), 3),
                         "mean_net_pct": round(100 * net.mean(), 2),
                         "t_net": round(net.mean() / net.std() * np.sqrt(len(net)), 2),
                         "next_open_gap_pct": round(100 * (g["next_open_gap"] * side).mean(), 2)})
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUT}/17_limit_summary.csv", index=False)

    pd.set_option("display.width", 200)
    yr = pd.DataFrame(rule).groupby("year")[["max_up_pct", "max_dn_pct"]].agg(["max", "min"])
    print("=== Жил бүрийн хамгийн их/бага өгөөж (хязгаар өөрчлөгдсөн эсэх) ===")
    print(yr.to_string())
    if len(R):
        print("\n=== Дараалсан хязгаарын урт ===")
        print(R.groupby("side")["length"].value_counts().sort_index().to_string())
    print("\n=== Хязгаарын дараах өгөөж (дараагийн нээлтээр орсон, чиглэлийг дагасан) ===")
    print(S.to_string(index=False))
    print("  side: дээд хязгаар → long; доод хязгаар → 'short' (МХБ-д бодитоор хийх боломжгүй, зөвхөн мэдээлэл)")
    print(f"\n[+] ./{OUT}/")


if __name__ == "__main__":
    main()
