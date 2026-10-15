"""
МХБ-ийн хувьцааны өдрийн арилжааны ТҮҮХИЙГ CSV болгож татах.
(mse_top20.py-ийн задлагчийг ашигласан, цувааны шинжилгээнд зориулав.)

Ажиллуулах:
    python mse_history.py APU 2010-01-01 2026-10-14
    python mse_history.py APU,TTL,KHAN 2015-01-01 2026-10-14
    python mse_history.py ALL 2010-01-01 2026-10-14      # TOP-20 бүгд
Spyder/Jupyter-оос: доорх TICKERS, START_DATE, END_DATE-ийг засаад Run.

Гаралт:
    mse_<TICKER>_<start>_<end>.csv   (date, open, high, low, close, volume, value, deals)
    mse_quality_<start>_<end>.csv    (хувьцаа бүрийн өгөгдлийн чанарын хураангуй)

Шаардлагатай сан:  pip install requests pandas lxml
"""
import io
import sys
import time
import datetime as dt

import pandas as pd
import requests

TRADE_URL = "https://open.mse.mn/securities/{id}/tab/tradeinfo"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TOP20 = {
    "AARD": "326", "APU": "90", "CUMN": "557", "GLMT": "562", "GOV": "354",
    "INV": "553", "KHAN": "563", "LEND": "545", "MGLA": "572", "MLG": "569",
    "MNDL": "547", "MSE": "510", "QPAY": "570", "SBM": "564", "SUU": "135",
    "TDB": "567", "TTL": "458", "TUM": "549", "UID": "484", "XAC": "568",
}

RENAME = {
    "Огноо": "date", "Нээлтийн ханш": "open", "Хаалтын ханш": "close",
    "Өдрийн хязгаар": "range", "Арилжигдсан ширхэг": "volume",
    "Арилжааны үнийн дүн": "value", "Хэлцлийн тоо": "deals",
}
# --- Spyder/Jupyter-оос ажиллуулах бол эдгээрийг засаад Run дар ---
TICKERS = "ALL"            # "APU", "APU,TTL,KHAN" эсвэл "ALL"
START_DATE = "2010-01-01"
END_DATE = "2026-10-14"

COLS = ["date", "open", "high", "low", "close", "volume", "value", "deals"]


def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False)
         .str.replace("₮", "", regex=False).str.strip(),
        errors="coerce",
    )


def parse_trade_html(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html), displayed_only=False)
    except ValueError:
        return pd.DataFrame(columns=COLS)
    df = next((t for t in tables
               if {"Огноо", "Хаалтын ханш"} <= {str(c).strip() for c in t.columns}), None)
    if df is None:
        return pd.DataFrame(columns=COLS)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    for c in ("open", "close", "volume", "value", "deals"):
        if c in df.columns:
            df[c] = to_num(df[c])
    if "range" in df.columns:
        parts = df["range"].astype(str).str.split(" - ", n=1, expand=True)
        df["low"] = to_num(parts[0])
        df["high"] = to_num(parts[1]) if parts.shape[1] > 1 else df["low"]
    for c in COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLS].sort_values("date").reset_index(drop=True)


def fetch(session, comp_id):
    r = session.get(TRADE_URL.format(id=comp_id), headers=HEADERS, timeout=(5, 60))
    r.raise_for_status()
    r.encoding = "utf-8"
    return parse_trade_html(r.text)


def quality(ticker, df, start, end):
    """Цувааны шинжилгээнд хэрэгтэй өгөгдлийн чанарын үзүүлэлтүүд."""
    if df.empty:
        return {"ticker": ticker, "rows": 0}
    bdays = pd.bdate_range(df["date"].min(), df["date"].max())   # ажлын өдөр (баярыг тооцоогүй)
    ret = df["close"].pct_change()
    gaps = df["date"].diff().dt.days
    big = ret.abs() > 0.3                                          # хуваалт/алдаа сэжигтэй
    return {
        "ticker": ticker,
        "rows": len(df),
        "first_date": df["date"].min().date(),
        "last_date": df["date"].max().date(),
        "pct_bdays_traded": round(100 * len(df) / len(bdays), 1),
        "pct_zero_return": round(100 * (ret == 0).mean(), 1),
        "max_gap_days": int(gaps.max()) if len(df) > 1 else 0,
        "n_abs_ret_gt_30pct": int(big.sum()),
        "dates_abs_ret_gt_30pct": ";".join(df.loc[big, "date"].dt.strftime("%Y-%m-%d")),
        "missing_close": int(df["close"].isna().sum()),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
    if len(args) == 3:                       # командын мөрөөс
        tick_arg, s, e = args
    else:                                    # Spyder / Jupyter: файлын дээд талын утгууд
        tick_arg, s, e = TICKERS, START_DATE, END_DATE
    print(f"[+] {tick_arg}: {s} -> {e}")
    start, end = pd.Timestamp(s), pd.Timestamp(e)
    if start > end:
        raise SystemExit("[!] Эхлэх огноо дуусах огнооноос хойно байна.")
    tickers = sorted(TOP20) if tick_arg.upper() == "ALL" else [t.strip().upper() for t in tick_arg.split(",")]

    rows = []
    with requests.Session() as sess:
        for i, t in enumerate(tickers, 1):
            if t not in TOP20:
                print(f"[{i}/{len(tickers)}] {t}: ID мэдэгдэхгүй (TOP20 толь бичигт нэм)")
                continue
            try:
                df = fetch(sess, TOP20[t])
            except requests.RequestException as ex:
                print(f"[{i}/{len(tickers)}] {t}: сүлжээний алдаа — {ex}")
                continue
            full_first = df["date"].min() if not df.empty else None
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            fn = f"mse_{t}_{s}_{e}.csv"
            df.to_csv(fn, index=False, date_format="%Y-%m-%d")
            q = quality(t, df, start, end)
            rows.append(q)
            note = ""
            if full_first is not None and full_first > start:
                note = f"  (сайт {full_first.date()}-ээс эхэлсэн өгөгдөл өгсөн)"
            print(f"[{i}/{len(tickers)}] {t}: {len(df)} мөр -> {fn}{note}")
            time.sleep(1)  # сайтад ачаалал өгөхгүй

    if rows:
        qdf = pd.DataFrame(rows)
        qfn = f"mse_quality_{s}_{e}.csv"
        qdf.to_csv(qfn, index=False)
        pd.set_option("display.width", 200)
        print("\n=== ӨГӨГДЛИЙН ЧАНАР ===")
        print(qdf.drop(columns=["dates_abs_ret_gt_30pct"]).to_string(index=False))
        print(f"\n[+] {qfn}")


if __name__ == "__main__":
    main()