"""
fetch_cpt_data.py  (v2)
=======================
Fetches free-source proxy data for the 14 asset classes in:
  Luxenberg, Schiele & Boyd (2024), Computational Economics 64, 3027-3047.

Target: July 1972 - June 2022  (600 monthly returns, 14 assets)

INSTALL:
    pip install yfinance fredapi

FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

warnings.filterwarnings("ignore")

FRED_API_KEY = "9f8e9a8de2b4e2847345902878c47f55"
START = "1972-07-01"
END   = "2022-06-30"

fred = Fred(api_key=FRED_API_KEY)


def to_monthly_returns(price_series, name):
    s = price_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    monthly = s.resample("ME").last().dropna()
    ret = monthly.pct_change().dropna()
    ret.name = name
    return ret


def rate_to_monthly(rate_series, name):
    s = rate_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    monthly = s.resample("ME").last().dropna()
    ret = monthly / 100.0 / 12.0
    ret.name = name
    return ret


def from_yahoo(ticker, name, start=START, end=END):
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    price = df["Close"].squeeze()
    price.index = pd.to_datetime(price.index)
    return to_monthly_returns(price, name)


def from_fred_price(series_id, name, start=START, end=END):
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    return to_monthly_returns(s, name)


def from_fred_rate(series_id, name, start=START, end=END):
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    return rate_to_monthly(s, name)


def try_each(label, attempts):
    """Try a list of (description, callable) in order; return first success."""
    for desc, fn in attempts:
        try:
            result = fn()
            n = result.notna().sum()
            print(f"    -> {n} months  [{desc}]")
            return result
        except Exception as e:
            print(f"    x {desc}: {e}")
    return None


def fetch_all():
    series = {}

    # 1. US Equity
    print("1/14  US Equity (S&P 500 TR)...")
    tr  = try_each("^SP500TR", [("^SP500TR", lambda: from_yahoo("^SP500TR", "t"))])
    raw = try_each("^GSPC",    [("^GSPC",    lambda: from_yahoo("^GSPC",    "t"))])
    if tr is not None and raw is not None:
        raw_tr = raw + 0.04 / 12   # crude dividend adjustment pre-1988
        raw_tr.update(tr)
        raw_tr.name = "US_Equity_TR"
        series["US_Equity_TR"] = raw_tr
    elif raw is not None:
        raw.name = "US_Equity_TR"
        series["US_Equity_TR"] = raw

    # 2. European Equity
    print("2/14  European Equity...")
    r = try_each("Europe", [
        ("^STOXX50E", lambda: from_yahoo("^STOXX50E", "Europe_Equity_TR")),
    ])
    if r is not None: series["Europe_Equity_TR"] = r

    # 3. Japan Equity
    print("3/14  Japan Equity...")
    r = try_each("Japan", [
        ("^N225",  lambda: from_yahoo("^N225",  "Japan_Equity_TR")),
        ("^TOPX",  lambda: from_yahoo("^TOPX",  "Japan_Equity_TR")),
        ("EWJ",    lambda: from_yahoo("EWJ",    "Japan_Equity_TR")),
    ])
    if r is not None: series["Japan_Equity_TR"] = r

    # 4. Emerging Markets Equity
    print("4/14  EM Equity (EEM, 2003+)...")
    r = try_each("EM", [
        ("EEM", lambda: from_yahoo("EEM", "EM_Equity_TR")),
        ("VWO", lambda: from_yahoo("VWO", "EM_Equity_TR")),
    ])
    if r is not None: series["EM_Equity_TR"] = r

    # 5. US Government Bonds
    print("5/14  US Government Bonds...")
    r = try_each("US Govt", [
        ("FRED BAMLGVTRLST1TRV", lambda: from_fred_price("BAMLGVTRLST1TRV", "US_GovtBond_TR")),
        ("IEF ETF",              lambda: from_yahoo("IEF", "US_GovtBond_TR")),
        ("TLT ETF",              lambda: from_yahoo("TLT", "US_GovtBond_TR")),
    ])
    if r is not None: series["US_GovtBond_TR"] = r

    # 6. US Corporate Bonds
    print("6/14  US Corporate Bonds...")
    r = try_each("US Corp", [
        ("FRED BAMLCC0A0CMTRIV", lambda: from_fred_price("BAMLCC0A0CMTRIV", "US_CorpBond_TR")),
        ("LQD ETF",              lambda: from_yahoo("LQD",  "US_CorpBond_TR")),
        ("VCIT ETF",             lambda: from_yahoo("VCIT", "US_CorpBond_TR")),
    ])
    if r is not None: series["US_CorpBond_TR"] = r

    # 7. European Government Bonds
    print("7/14  European Government Bonds...")
    r = try_each("EU Govt", [
        ("IBGL.L",   lambda: from_yahoo("IBGL.L",   "EU_GovtBond_TR")),
        ("VGEA.DE",  lambda: from_yahoo("VGEA.DE",  "EU_GovtBond_TR")),
        ("IEGA.L",   lambda: from_yahoo("IEGA.L",   "EU_GovtBond_TR")),
        ("EUGOV.L",  lambda: from_yahoo("EUGOV.L",  "EU_GovtBond_TR")),
    ])
    if r is not None: series["EU_GovtBond_TR"] = r

    # 8. Japan Government Bonds
    print("8/14  Japan Government Bonds...")
    r = try_each("JP Govt", [
        ("IJPN.L",  lambda: from_yahoo("IJPN.L",  "JP_GovtBond_TR")),
        ("BNDX",    lambda: from_yahoo("BNDX",    "JP_GovtBond_TR")),
        ("2511.T",  lambda: from_yahoo("2511.T",  "JP_GovtBond_TR")),
    ])
    if r is not None: series["JP_GovtBond_TR"] = r

    # 9. US Bills
    print("9/14  US Bills (FRED TB3MS)...")
    r = try_each("US Bills", [
        ("TB3MS", lambda: from_fred_rate("TB3MS", "US_Bills")),
    ])
    if r is not None: series["US_Bills"] = r

    # 10. European Bills
    print("10/14 European Bills...")
    r = try_each("EU Bills", [
        ("IR3TIB01DEM156N", lambda: from_fred_rate("IR3TIB01DEM156N", "EU_Bills")),
        ("IRSTCI01EZM156N", lambda: from_fred_rate("IRSTCI01EZM156N", "EU_Bills")),
    ])
    if r is not None: series["EU_Bills"] = r

    # 11. Japan Bills
    print("11/14 Japan Bills...")
    r = try_each("JP Bills", [
        ("IR3TIB01JPM156N", lambda: from_fred_rate("IR3TIB01JPM156N", "JP_Bills")),
        ("IRSTCI01JPM156N", lambda: from_fred_rate("IRSTCI01JPM156N", "JP_Bills")),
    ])
    if r is not None: series["JP_Bills"] = r

    # 12. Commodities
    print("12/14 Commodities (S&P GSCI)...")
    r = try_each("Commodities", [
        ("^SPGSCI", lambda: from_yahoo("^SPGSCI", "Commodities_TR")),
        ("GSG",     lambda: from_yahoo("GSG",     "Commodities_TR")),
        ("DJP",     lambda: from_yahoo("DJP",     "Commodities_TR")),
    ])
    if r is not None: series["Commodities_TR"] = r

    # 13. Gold  -- FIX: use Yahoo Finance (GC=F futures), not broken FRED IDs
    print("13/14 Gold...")
    r = try_each("Gold", [
        ("GC=F", lambda: from_yahoo("GC=F", "Gold")),
        ("GLD",  lambda: from_yahoo("GLD",  "Gold")),
        ("IAU",  lambda: from_yahoo("IAU",  "Gold")),
    ])
    if r is not None: series["Gold"] = r

    # 14. Silver  -- FIX: use Yahoo Finance (SI=F futures), not broken FRED IDs
    print("14/14 Silver...")
    r = try_each("Silver", [
        ("SI=F", lambda: from_yahoo("SI=F", "Silver")),
        ("SLV",  lambda: from_yahoo("SLV",  "Silver")),
    ])
    if r is not None: series["Silver"] = r

    df = pd.concat(series.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[START:END]
    return df


if __name__ == "__main__":
    print("=" * 65)
    print("CPT paper proxy data  (July 1972 - June 2022)")
    print("=" * 65)

    df = fetch_all()

    print("\n-- Summary -------------------------------------------------------")
    print(f"Shape: {df.shape}  (target: 600 x 14)")
    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"\nCoverage (non-NaN months per asset):")
    for col, n in df.notna().sum().sort_values(ascending=False).items():
        bar = "#" * (n // 20)
        print(f"  {col:<22} {n:>4}  {bar}")

    df.to_csv("cpt_proxy_returns.csv")
    print(f"\n[ok] Saved -> cpt_proxy_returns.csv")

    df_clean = df.dropna()
    print(f"\nFully-observed months: {len(df_clean)} of {len(df)}")

    long_assets = df.columns[df.notna().sum() >= 500].tolist()
    df_long = df[long_assets].dropna()
    print(f"Long-history subset ({len(long_assets)} assets, >=500mo): {len(df_long)} months")

    if len(df_clean) > 0:
        np.save("cpt_proxy_returns_full.npy", df_clean.values)
        print(f"[ok] cpt_proxy_returns_full.npy  {df_clean.shape}")

    if len(df_long) > 0:
        np.save("cpt_proxy_returns_longhist.npy", df_long.values)
        print(f"[ok] cpt_proxy_returns_longhist.npy  {df_long.shape}")
        print(f"     Assets: {long_assets}")

    print("""
-- How to use with cptopt ----------------------------------------
import numpy as np
from cptopt.optimizer import (MinorizationMaximizationOptimizer,
                               ConvexConcaveOptimizer,
                               GradientOptimizer)
from cptopt.utility import CPTUtility

r = np.load('cpt_proxy_returns_longhist.npy')  # (N, n)
n = r.shape[1]
print(f"Data: {r.shape[0]} months x {n} assets")

utility = CPTUtility(gamma_pos=8.4, gamma_neg=11.4,
                     delta_pos=0.77, delta_neg=0.79)
w0 = np.ones(n) / n

mm = MinorizationMaximizationOptimizer(utility)
mm.optimize(r, initial_weights=w0, verbose=True)
print("MM weights:", mm.weights.round(4))
print("MM utility:", utility(r, mm.weights))
""")