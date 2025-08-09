#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1 Next Race Predictor (2025) — Dual Model with Calibrated DNF + History Rates
- DNF model: Calibrated (isotonic) LogisticRegression (class_weight='balanced')
- Position model: GradientBoostingRegressor (finish position if they finish)
- Robust to FastF1 2025 columns (FullName/BroadcastName/Abbreviation)
- Trains on 2025 races completed before the next event
- Optional: --use-qual to use qualifying as grid when available
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import fastf1 as f1

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_DIR = "fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
f1.Cache.enable_cache(CACHE_DIR)  # local cache


@dataclass
class RaceKey:
    year: int
    round: int
    event_name: str


# ------------------------------- Helpers -------------------------------

def _driver_name(row: pd.Series) -> str:
    if 'FullName' in row and pd.notna(row['FullName']):
        return str(row['FullName'])
    if 'BroadcastName' in row and pd.notna(row['BroadcastName']):
        return str(row['BroadcastName'])
    if 'Abbreviation' in row and pd.notna(row['Abbreviation']):
        return str(row['Abbreviation'])
    return f"#{row.get('DriverNumber', 'UNK')}"


def get_next_event_2025(today: Optional[pd.Timestamp] = None) -> RaceKey:
    if today is None:
        today = pd.Timestamp.now(tz="UTC")
    schedule = f1.get_event_schedule(2025)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate']).dt.tz_localize("UTC")
    upcoming = schedule[schedule['EventDate'] >= today].sort_values('EventDate')
    if upcoming.empty:
        last_row = schedule.sort_values('EventDate').iloc[-1]
        return RaceKey(2025, int(last_row['RoundNumber']), str(last_row['EventName']))
    row = upcoming.iloc[0]
    return RaceKey(2025, int(row['RoundNumber']), str(row['EventName']))


def _strict_dnf_label(df: pd.DataFrame) -> pd.Series:
    """
    Stricter DNF:
      - Finisher if Status contains 'finish' or 'lap' (e.g., '+1 Lap'), OR ClassifiedPosition is numeric
      - DNF only if neither condition holds
    """
    if 'Status' in df.columns:
        status = df['Status'].astype(str).str.lower()
        finished_mask = status.str.contains('finish') | status.str.contains('lap')
    else:
        finished_mask = pd.Series(False, index=df.index)
    num_pos = pd.to_numeric(df.get('ClassifiedPosition'), errors='coerce').notna()
    return (~finished_mask & ~num_pos).astype(int)


def _safe_load_results(year: int, round_num: int) -> Optional[pd.DataFrame]:
    """Load race results for a given round. Returns None if unavailable/malformed."""
    try:
        ses = f1.get_session(year, round_num, 'R')
        ses.load(telemetry=False, weather=False, messages=False)
        df = ses.results.copy()
        if df is None or df.empty:
            return None

        # Normalize essentials present in 2025 FastF1
        keep = [
            'DriverNumber', 'BroadcastName', 'Abbreviation', 'DriverId',
            'TeamName', 'Position', 'ClassifiedPosition', 'GridPosition',
            'Status', 'Points', 'Laps', 'FullName'
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # Add derived columns
        df['Driver'] = df.apply(_driver_name, axis=1)
        df['Grid'] = pd.to_numeric(df.get('GridPosition'), errors='coerce')

        # Numeric finish position for modeling form; DNFs get a large penalty value for rolling stats
        fin = pd.to_numeric(df.get('ClassifiedPosition'), errors='coerce')
        maxpos = int(np.nanmax(fin)) if fin.notna().any() else 20
        df['FinishPosNum'] = fin.fillna(maxpos + 5)

        # Stricter DNF label
        df['DNF'] = _strict_dnf_label(df)

        return df
    except Exception as e:
        print(f"[WARN] Round {round_num}: could not load results ({e})")
        return None


def load_completed_2025_before(round_exclusive: int) -> pd.DataFrame:
    """Concat results for rounds < round_exclusive."""
    rows = []
    for rnd in range(1, round_exclusive):
        df = _safe_load_results(2025, rnd)
        if df is not None and not df.empty:
            df['Round'] = rnd
            rows.append(df)
    if not rows:
        raise RuntimeError("No usable 2025 race results found before the next round.")
    return pd.concat(rows, ignore_index=True)


def build_season_to_date_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling season-to-date features per driver and per team BEFORE each race."""
    df = results_df.copy()

    # Sort by driver, round for rolling stats
    key_id = 'DriverId' if 'DriverId' in df.columns else 'Driver'
    df = df.sort_values([key_id, 'Round']).reset_index(drop=True)

    # Per-driver rolling (shift to avoid leakage)
    def driver_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g['StartsToDate'] = np.arange(1, len(g) + 1)
        g['AvgFinishSTD'] = g['FinishPosNum'].shift(1).expanding().mean()
        g['AvgGridST'] = g['Grid'].shift(1).expanding().mean()
        g['BestFinishST'] = g['FinishPosNum'].shift(1).expanding().min()
        g['DNFsST'] = g['DNF'].shift(1).expanding().sum()
        g['Form3'] = g['FinishPosNum'].shift(1).rolling(3, min_periods=1).mean()
        # NEW: driver DNF rate S2D
        g['DriverDNFRateST'] = g['DNF'].shift(1).expanding().mean()
        # points proxy using classified position
        points_map = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
        pts = pd.to_numeric(g.get('ClassifiedPosition'), errors='coerce').map(points_map).fillna(0)
        g['CumPtsST'] = pts.shift(1).expanding().sum()
        return g

    df = df.groupby(key_id, group_keys=False).apply(driver_roll)

    # Per-team rolling
    def team_roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g['TeamAvgFinishST'] = g['FinishPosNum'].shift(1).expanding().mean()
        g['TeamCumPtsST'] = pd.to_numeric(g.get('ClassifiedPosition'), errors='coerce') \
                                .map({1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}) \
                                .fillna(0).shift(1).expanding().sum()
        # NEW: team DNF rate S2D
        g['TeamDNFRateST'] = g['DNF'].shift(1).expanding().mean()
        return g

    df = df.groupby('TeamName', group_keys=False).apply(team_roll)

    # Fill first-appearance NaNs with medians
    num_cols = [
        'AvgFinishSTD','AvgGridST','BestFinishST','DNFsST','Form3','CumPtsST',
        'TeamAvgFinishST','TeamCumPtsST','DriverDNFRateST','TeamDNFRateST'
    ]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df


def assemble_datasets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_dnf, y_dnf, X_pos, y_pos)."""
    feature_cols = [
        'Grid', 'AvgFinishSTD', 'AvgGridST', 'BestFinishST',
        'DNFsST', 'Form3', 'CumPtsST', 'TeamAvgFinishST', 'TeamCumPtsST',
        'DriverDNFRateST', 'TeamDNFRateST'
    ]

    # DNF model: all rows
    X_dnf = df[feature_cols].copy()
    y_dnf = df['DNF'].astype(int)

    # Position model: only finishers (DNF==0) with numeric target
    finish_mask = df['DNF'] == 0
    X_pos = df.loc[finish_mask, feature_cols].copy()
    y_pos = df.loc[finish_mask, 'FinishPosNum'].astype(float)  # numeric target

    return X_dnf, y_dnf, X_pos, y_pos


def train_models(X_dnf, y_dnf, X_pos, y_pos):
    # Calibrated, class-balanced logistic regression for DNF
    base = LogisticRegression(max_iter=200, class_weight='balanced', solver='liblinear')
    dnf_clf = CalibratedClassifierCV(base, method='isotonic', cv=5)

    # Keep GBM for finishing position
    pos_reg = GradientBoostingRegressor(
        n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, random_state=42
    )

    # Train/validation splits (guard stratify if single class)
    try:
        Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
            X_dnf, y_dnf, test_size=0.2, random_state=42, stratify=y_dnf
        )
    except ValueError:
        Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
            X_dnf, y_dnf, test_size=0.2, random_state=42
        )

    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_pos, y_pos, test_size=0.2, random_state=42)

    dnf_clf.fit(Xd_tr, yd_tr)
    pos_reg.fit(Xp_tr, yp_tr)

    # Metrics + sanity checks
    if len(np.unique(yd_te)) > 1:
        dnf_val = roc_auc_score(yd_te, dnf_clf.predict_proba(Xd_te)[:, 1])
        print(f"DNF AUC: {dnf_val:.3f}")
    else:
        print("DNF AUC: N/A (single class in validation)")

    pos_val = mean_absolute_error(yp_te, pos_reg.predict(Xp_te))
    print(f"Position MAE (finishers): {pos_val:.2f}")

    print("DNF rate in training set:", float(yd_tr.mean()))
    probs_train = dnf_clf.predict_proba(Xd_tr)[:, 1]
    print("DNF predicted prob range (train):", float(probs_train.min()), "→", float(probs_train.max()))

    return dnf_clf, pos_reg


def get_grid_for_event(year: int, rnd: int) -> Optional[pd.DataFrame]:
    try:
        q = f1.get_session(year, rnd, 'Q')
        q.load(telemetry=False, weather=False, messages=False)
        qres = q.results[['DriverNumber', 'BroadcastName', 'Abbreviation', 'DriverId', 'FullName', 'Position']].copy()
        qres['Driver'] = qres.apply(_driver_name, axis=1)
        qres.rename(columns={'Position': 'QualPos'}, inplace=True)
        qres['Grid'] = pd.to_numeric(qres['QualPos'], errors='coerce')
        return qres[['DriverId', 'Driver', 'Abbreviation', 'Grid']]
    except Exception:
        return None


def prepare_next_event_features(next_event: RaceKey, season_df: pd.DataFrame, use_qual: bool = False) -> pd.DataFrame:
    """Take last S2D row per driver and build feature rows for the next event."""
    key_id = 'DriverId' if 'DriverId' in season_df.columns else 'Driver'
    latest = (season_df.sort_values([key_id, 'Round'])
                      .groupby(key_id)
                      .tail(1)
                      .copy())

    # Before quali, use driver's median grid so far; after quali, swap in actual grid.
    latest['Grid'] = latest.groupby(key_id)['Grid'].transform(
        lambda s: np.nanmedian(s) if np.isfinite(np.nanmedian(s)) else 10.0
    )

    if use_qual:
        q = get_grid_for_event(next_event.year, next_event.round)
        if q is not None:
            latest = latest.merge(q[[key_id, 'Grid']], on=key_id, how='left', suffixes=('', '_Q'))
            latest['Grid'] = latest['Grid_Q'].combine_first(latest['Grid'])
            latest.drop(columns=['Grid_Q'], inplace=True)

    # Only keep the features needed for prediction plus labels for display
    features = [
        'Grid', 'AvgFinishSTD', 'AvgGridST', 'BestFinishST',
        'DNFsST', 'Form3', 'CumPtsST', 'TeamAvgFinishST', 'TeamCumPtsST',
        'DriverDNFRateST', 'TeamDNFRateST'
    ]
    display_cols = ['Driver', 'Abbreviation', 'TeamName']
    keep = [c for c in display_cols + features if c in latest.columns]
    return latest[keep].copy()


# ------------------------------- Pipeline -------------------------------

def predict_next_race(use_qual: bool = False):
    next_ev = get_next_event_2025()
    print(f"==> Next event: Round {next_ev.round} - {next_ev.event_name} ({next_ev.year})")

    # Training data: 2025 races completed so far
    results = load_completed_2025_before(next_ev.round)
    season_df = build_season_to_date_features(results)

    # Build datasets and train
    X_dnf, y_dnf, X_pos, y_pos = assemble_datasets(season_df)
    dnf_clf, pos_reg = train_models(X_dnf, y_dnf, X_pos, y_pos)

    # Next-event features
    X_next = prepare_next_event_features(next_ev, season_df, use_qual=use_qual)

    meta_cols = [c for c in ['Driver', 'Abbreviation', 'TeamName'] if c in X_next.columns]
    meta = X_next[meta_cols].copy()
    feat_cols = [c for c in X_next.columns if c not in meta_cols]

    # Predictions
    dnf_probs = dnf_clf.predict_proba(X_next[feat_cols])[:, 1]
    pos_preds = pos_reg.predict(X_next[feat_cols])

    out = meta.copy()
    out['DNF_Prob_%'] = (dnf_probs * 100).round(1)
    out['PredictedFinishPos'] = pos_preds.round(2)
    # Also include priors so you can inspect them in the CSV
    if 'DriverDNFRateST' in X_next.columns: out['DriverDNFRateST'] = X_next['DriverDNFRateST'].round(3)
    if 'TeamDNFRateST' in X_next.columns:   out['TeamDNFRateST'] = X_next['TeamDNFRateST'].round(3)

    # Sort primarily by predicted finish (lower is better), then by DNF probability
    out['RankIfFinish'] = out['PredictedFinishPos'].rank(method='first')
    out = out.sort_values(['PredictedFinishPos', 'DNF_Prob_%']).reset_index(drop=True)

    print("\n=== Predictions ===")
    for i, r in out.iterrows():
        name = r.get('Driver', r.get('Abbreviation', f"#{i+1}"))
        team = r.get('TeamName', '')
        print(f"{i+1:>2}. {name} {f'({team})' if team else ''}  |  DNF {r['DNF_Prob_%']:>4.1f}%  |  if finish: ~P{r['PredictedFinishPos']:.1f}")

    # Save
    fname = f"predictions_round_{next_ev.round:02d}_{next_ev.event_name.replace(' ', '_')}.csv"
    out.to_csv(fname, index=False)
    print(f"\nSaved predictions to: {fname}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict next F1 race (2025): DNF probability + finishing position.")
    parser.add_argument("--use-qual", action="store_true", help="Use qualifying results for grid if available.")
    args = parser.parse_args()
    predict_next_race(use_qual=args.use_qual)


if __name__ == "__main__":
    main()
