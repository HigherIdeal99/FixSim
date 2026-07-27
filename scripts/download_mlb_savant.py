from __future__ import annotations

import json
import time
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"
YEARS = list(range(2019, 2026))
OUT = Path("mlb_savant_pitcher_raw_2019_2025")

SELECTIONS = [
    "player_age", "ab", "pa", "hit", "single", "double", "triple", "home_run",
    "strikeout", "walk", "k_percent", "bb_percent", "batting_avg", "slg_percent",
    "on_base_percent", "on_base_plus_slg", "xba", "xslg", "woba", "xwoba",
    "xobp", "xiso", "wobacon", "xwobacon", "bacon", "xbacon", "xbadiff",
    "xslgdiff", "wobadiff", "xera", "exit_velocity_avg", "launch_angle_avg",
    "sweet_spot_percent", "barrel", "barrel_batted_rate", "solidcontact_percent",
    "flareburner_percent", "poorlyunder_percent", "poorlytopped_percent",
    "poorlyweak_percent", "hard_hit_percent", "avg_best_speed", "avg_hyper_speed",
    "z_swing_percent", "z_swing_miss_percent", "oz_swing_percent",
    "oz_swing_miss_percent", "oz_contact_percent", "out_zone_swing_miss",
    "out_zone_swing", "out_zone_percent", "out_zone", "meatball_swing_percent",
    "meatball_percent", "iz_contact_percent", "in_zone_swing_miss", "in_zone_swing",
    "in_zone_percent", "in_zone", "edge_percent", "edge", "whiff_percent",
    "swing_percent", "pitch_count_offspeed", "pitch_count_fastball",
    "pitch_count_breaking", "pitch_count", "f_strike_percent", "pull_percent",
    "straightaway_percent", "opposite_percent", "batted_ball", "groundballs_percent",
    "groundballs", "flyballs_percent", "flyballs", "linedrives_percent", "linedrives",
    "popups_percent", "popups", "p_era", "p_opp_batting_avg", "p_quality_start",
    "p_game", "p_formatted_ip", "pitch_hand", "velocity", "release_extension",
    "arm_angle", "n_ff_formatted", "ff_avg_speed", "ff_avg_spin", "ff_avg_break_x",
    "ff_avg_break_z", "n_si_formatted", "si_avg_speed", "si_avg_spin", "si_avg_break_x",
    "si_avg_break_z", "n_sl_formatted", "sl_avg_speed", "sl_avg_spin", "sl_avg_break_x",
    "sl_avg_break_z", "n_ch_formatted", "ch_avg_speed", "ch_avg_spin", "ch_avg_break_x",
    "ch_avg_break_z", "n_cu_formatted", "cu_avg_speed", "cu_avg_spin", "cu_avg_break_x",
    "cu_avg_break_z", "n_fc_formatted", "fc_avg_speed", "fc_avg_spin", "fc_avg_break_x",
    "fc_avg_break_z", "n_fs_formatted", "fs_avg_speed", "fs_avg_spin", "fs_avg_break_x",
    "fs_avg_break_z", "n_kn_formatted", "kn_avg_speed", "kn_avg_spin", "kn_avg_break_x",
    "kn_avg_break_z",
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; research dataset downloader)",
        "Accept": "text/csv,text/plain,*/*",
    })
    frames: list[pd.DataFrame] = []
    manifest: dict = {
        "source": BASE_URL,
        "type": "pitcher",
        "minimum": 1,
        "years": YEARS,
        "selected_columns": SELECTIONS,
        "files": [],
    }

    for year in YEARS:
        params = {
            "year": year, "type": "pitcher", "filter": "", "min": "1",
            "selections": ",".join(SELECTIONS), "chart": "false", "x": "pa",
            "y": "pa", "r": "no", "chartType": "beeswarm", "sort": "xwoba",
            "sortDir": "asc", "csv": "true",
        }
        response = session.get(BASE_URL, params=params, timeout=120)
        response.raise_for_status()
        text = response.text
        if "<html" in text[:500].lower():
            raise RuntimeError(f"{year}: expected CSV but received HTML")
        df = pd.read_csv(StringIO(text))
        if df.empty:
            raise RuntimeError(f"{year}: empty CSV")
        if "year" not in df.columns:
            df.insert(2 if len(df.columns) >= 2 else 0, "year", year)
        else:
            df["year"] = year
        path = OUT / f"baseball_savant_pitcher_{year}_raw.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        frames.append(df)
        manifest["files"].append({
            "year": year, "filename": path.name, "rows": len(df),
            "columns": len(df.columns), "column_names": list(df.columns),
        })
        print(f"{year}: {len(df)} rows x {len(df.columns)} columns", flush=True)
        time.sleep(2)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined_path = OUT / "baseball_savant_pitcher_2019_2025_raw_combined.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    manifest["combined"] = {
        "filename": combined_path.name, "rows": len(combined),
        "columns": len(combined.columns), "column_names": list(combined.columns),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# MLB Baseball Savant Pitcher Raw Dataset (2019-2025)\n\n"
        "- Source: official Baseball Savant Custom Leaderboard CSV endpoint\n"
        "- Unit: one row = one pitcher-season\n"
        "- Train candidate: 2019-2024\n"
        "- Held-out result season: 2025\n"
        "- Minimum: min=1; no feature engineering, imputation, normalization or manual filtering\n"
        "- CSV encoding: UTF-8 with BOM\n",
        encoding="utf-8",
    )
    zip_path = Path("mlb_savant_pitcher_raw_2019_2025.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT.parent))
    print(f"Combined: {len(combined)} rows x {len(combined.columns)} columns", flush=True)
    print(f"ZIP: {zip_path} ({zip_path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
