"""IPL Performance Analysis and Match Outcome Prediction.

Place IPL CSV files in ``data/matches.csv`` and ``data/deliveries.csv``.
If neither file exists, a reproducible demo data set is created so the full
pipeline can be demonstrated. The output CSVs in ``data/processed`` are ready
to import into Power BI or Tableau.
"""

from __future__ import annotations

import os
import sys
import warnings
import json
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "models"
MATCHES_RAW_PATH = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_RAW_PATH = os.path.join(DATA_DIR, "deliveries.csv")
MODEL_EXPORT_PATH = os.path.join(MODELS_DIR, "random_forest_ipl.joblib")
CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_json.zip"
SOURCE_MARKER = os.path.join(DATA_DIR, ".cricsheet_source")
POWERBI_WORKBOOK_PATH = os.path.join(PROCESSED_DIR, "IPL_PowerBI_Data.xlsx")

TEAM_MAPPINGS = {
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
VENUE_MAPPINGS = {
    "M Chinnaswamy Stadium": "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Cricket Stadium",
}


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def normalise_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Rename common IPL dataset variants to the column names used internally."""
    rename_map = {
        source: destination
        for source, destination in aliases.items()
        if source in frame.columns and destination not in frame.columns
    }
    return frame.rename(columns=rename_map)


def download_real_ipl_data() -> None:
    """Download official Cricsheet IPL JSON and convert it to analysis CSVs."""
    print("[INFO] Downloading official Cricsheet IPL data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dir = os.path.join(DATA_DIR, "ipl_json")
    json_available = os.path.isdir(extract_dir) and any(name.endswith(".json") for name in os.listdir(extract_dir))
    if not json_available:
        archive = os.path.join(DATA_DIR, "ipl_json.zip")
        try:
            urllib.request.urlretrieve(CRICSHEET_URL, archive)
        except Exception as error:
            raise RuntimeError("Could not download Cricsheet data. Download ipl_json.zip manually from https://cricsheet.org/downloads/ and place it in data/.") from error
        with zipfile.ZipFile(archive) as source:
            source.extractall(extract_dir)
    matches, deliveries = [], []
    for path in sorted(os.listdir(extract_dir)):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(extract_dir, path), encoding="utf-8") as handle:
            game = json.load(handle)
        info = game.get("info", {})
        teams = info.get("teams", [])
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")
        if len(teams) != 2 or not winner:
            continue
        match_id = os.path.splitext(path)[0]
        dates = info.get("dates", [])
        matches.append({"id": match_id, "date": dates[0] if dates else None,
            "season": info.get("season"), "team1": teams[0], "team2": teams[1],
            "venue": info.get("venue", info.get("city", "Unknown venue")),
            "toss_winner": info.get("toss", {}).get("winner"),
            "toss_decision": info.get("toss", {}).get("decision"), "winner": winner, "dl_applied": 0})
        for inning_no, inning in enumerate(game.get("innings", []), 1):
            batting = inning.get("team", "Unknown")
            bowling = teams[1] if batting == teams[0] else teams[0]
            for over in inning.get("overs", []):
                for ball_no, ball in enumerate(over.get("deliveries", []), 1):
                    runs = ball.get("runs", {})
                    extras = ball.get("extras", {})
                    wickets = ball.get("wickets", [])
                    first_wicket = wickets[0] if wickets else {}
                    deliveries.append({"match_id": match_id, "inning": inning_no,
                        "batting_team": batting, "bowling_team": bowling, "over": over.get("over", 0),
                        "ball": ball_no, "batsman": ball.get("batter"), "bowler": ball.get("bowler"),
                        "batsman_runs": runs.get("batter", 0), "extra_runs": runs.get("extras", 0),
                        "total_runs": runs.get("total", 0), "wide_runs": extras.get("wides", 0),
                        "noball_runs": extras.get("noballs", 0), "bye_runs": extras.get("byes", 0),
                        "legbye_runs": extras.get("legbyes", 0), "is_wicket": int(bool(wickets)),
                        "dismissal_kind": first_wicket.get("kind"), "player_dismissed": first_wicket.get("player_out")})
    if not matches or not deliveries:
        raise ValueError("Cricsheet download contained no usable IPL matches.")
    pd.DataFrame(matches).to_csv(MATCHES_RAW_PATH, index=False)
    pd.DataFrame(deliveries).to_csv(DELIVERIES_RAW_PATH, index=False)
    with open(SOURCE_MARKER, "w", encoding="utf-8") as marker:
        marker.write("Official Cricsheet IPL JSON; https://cricsheet.org/downloads/ipl_json.zip\n")
    print(f"[OK] Real data converted: {len(matches):,} matches, {len(deliveries):,} deliveries.")


def create_demo_datasets() -> None:
    """Use official data by default; demo data is no longer generated."""
    matches_exists = os.path.exists(MATCHES_RAW_PATH)
    deliveries_exists = os.path.exists(DELIVERIES_RAW_PATH)
    if matches_exists and deliveries_exists:
        if os.path.exists(SOURCE_MARKER):
            print("[OK] Official Cricsheet IPL datasets found.")
        else:
            print("[INFO] Existing data is not marked as official; replacing it with Cricsheet data.")
            download_real_ipl_data()
        return
    if matches_exists != deliveries_exists:
        raise FileNotFoundError(
            "Both data/matches.csv and data/deliveries.csv are required. "
            "One exists, so no demo files were generated to avoid overwriting it."
        )

    download_real_ipl_data()
    return

    # Retained below only as an offline fallback for source code reference.
    print("[INFO] Raw datasets not found; creating reproducible demo data.")
    rng = np.random.default_rng(42)
    teams = ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
             "Kolkata Knight Riders", "Delhi Capitals", "Punjab Kings",
             "Rajasthan Royals", "Sunrisers Hyderabad"]
    venues = ["M. Chinnaswamy Stadium", "Wankhede Stadium", "MA Chidambaram Stadium",
              "Eden Gardens", "Narendra Modi Stadium"]
    strengths = dict(zip(teams, [0.63, 0.61, 0.52, 0.56, 0.50, 0.47, 0.51, 0.54]))
    matches, deliveries = [], []
    match_id = 1
    for season in range(2018, 2024):
        for day in range(56):
            team1, team2 = rng.choice(teams, size=2, replace=False)
            venue = rng.choice(venues)
            toss_winner = rng.choice([team1, team2])
            toss_decision = rng.choice(["bat", "field"])
            probability = strengths[team1] / (strengths[team1] + strengths[team2])
            winner = team1 if rng.random() < probability else team2
            match_date = pd.Timestamp(f"{season}-03-20") + pd.Timedelta(days=day)
            matches.append({"id": match_id, "date": match_date, "season": season,
                            "team1": team1, "team2": team2, "venue": venue,
                            "toss_winner": toss_winner, "toss_decision": toss_decision,
                            "winner": winner, "dl_applied": 0})
            for inning, batting, bowling in [(1, team1, team2), (2, team2, team1)]:
                for over in range(20):
                    for ball in range(1, 7):
                        batter_runs = int(rng.choice([0, 1, 2, 3, 4, 6], p=[.34, .36, .12, .01, .12, .05]))
                        wicket = int(rng.random() < .045)
                        deliveries.append({"match_id": match_id, "inning": inning,
                            "batting_team": batting, "bowling_team": bowling, "over": over,
                            "ball": ball, "batsman": f"{batting} Player {rng.integers(1, 12)}",
                            "bowler": f"{bowling} Bowler {rng.integers(1, 7)}",
                            "batsman_runs": batter_runs, "extra_runs": 0, "total_runs": batter_runs,
                            "wide_runs": 0, "noball_runs": 0, "bye_runs": 0, "legbye_runs": 0,
                            "is_wicket": wicket, "dismissal_kind": "bowled" if wicket else np.nan})
            match_id += 1
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(matches).to_csv(MATCHES_RAW_PATH, index=False)
    pd.DataFrame(deliveries).to_csv(DELIVERIES_RAW_PATH, index=False)


def clean_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n[1/5] Cleaning data")
    matches, deliveries = pd.read_csv(MATCHES_RAW_PATH), pd.read_csv(DELIVERIES_RAW_PATH)
    matches = normalise_columns(matches, {
        "match_id": "id", "match_date": "date", "toss_decision": "toss_decision",
        "match_winner": "winner", "venue_name": "venue",
    })
    deliveries = normalise_columns(deliveries, {
        "innings": "inning", "batter": "batsman", "batsman_name": "batsman",
        "bowler_name": "bowler", "runs_off_bat": "batsman_runs",
        "total_run": "total_runs", "extras": "extra_runs", "isWicketDelivery": "is_wicket",
    })
    require_columns(matches, ["id", "date", "team1", "team2", "venue", "toss_winner", "toss_decision", "winner"], "matches.csv")
    require_columns(deliveries, ["match_id", "batting_team", "bowling_team", "batsman", "bowler", "batsman_runs"], "deliveries.csv")
    matches = matches.dropna(subset=["winner"]).copy()
    if "dl_applied" in matches:
        matches = matches[matches["dl_applied"].fillna(0).eq(0)].copy()
    for column in ["team1", "team2", "toss_winner", "winner"]:
        matches[column] = matches[column].replace(TEAM_MAPPINGS)
    matches["venue"] = matches["venue"].replace(VENUE_MAPPINGS).fillna("Unknown venue")
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "season" in matches:
        matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
        matches["season"] = matches["season"].fillna(matches["date"].dt.year).astype(int)
    else:
        matches["season"] = matches["date"].dt.year.astype(int)

    for column in ["batting_team", "bowling_team"]:
        deliveries[column] = deliveries[column].replace(TEAM_MAPPINGS)
    for column in ["extra_runs", "wide_runs", "noball_runs", "bye_runs", "legbye_runs", "is_wicket"]:
        if column not in deliveries:
            deliveries[column] = 0
        deliveries[column] = pd.to_numeric(deliveries[column], errors="coerce").fillna(0)
    deliveries["batsman_runs"] = pd.to_numeric(deliveries["batsman_runs"], errors="coerce").fillna(0)
    deliveries["total_runs"] = pd.to_numeric(deliveries.get("total_runs", deliveries["batsman_runs"] + deliveries["extra_runs"]), errors="coerce").fillna(0)
    deliveries["is_legal_ball"] = (deliveries["wide_runs"] == 0) & (deliveries["noball_runs"] == 0)
    over_column = "over" if "over" in deliveries else "match_over"
    if over_column in deliveries:
        deliveries["over_phase"] = pd.cut(deliveries[over_column], [-1, 5, 14, 20], labels=["Powerplay", "Middle Overs", "Death Overs"])
    else:
        deliveries["over_phase"] = "Unknown"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    matches.to_csv(os.path.join(PROCESSED_DIR, "cleaned_matches.csv"), index=False)
    deliveries.to_csv(os.path.join(PROCESSED_DIR, "cleaned_deliveries.csv"), index=False)
    return matches, deliveries


def _past_matches(df: pd.DataFrame, team: str, date: pd.Timestamp) -> pd.DataFrame:
    return df[(df.date < date) & ((df.team1 == team) | (df.team2 == team))]


def feature_engineering_pipeline(matches: pd.DataFrame) -> pd.DataFrame:
    print("[2/5] Building pre-match features")
    records = []
    for row in matches.sort_values("date").itertuples(index=False):
        past1, past2 = _past_matches(matches, row.team1, row.date).tail(5), _past_matches(matches, row.team2, row.date).tail(5)
        h2h = matches[(matches.date < row.date) & (((matches.team1 == row.team1) & (matches.team2 == row.team2)) | ((matches.team1 == row.team2) & (matches.team2 == row.team1)))]
        venue1 = _past_matches(matches, row.team1, row.date)
        venue1 = venue1[venue1["venue"].eq(row.venue)]
        venue2 = _past_matches(matches, row.team2, row.date)
        venue2 = venue2[venue2["venue"].eq(row.venue)]
        rate = lambda data, team: (data.winner == team).mean() if len(data) else .5
        records.append({"date": row.date, "team1": row.team1, "team2": row.team2, "venue": row.venue,
            "toss_decision": row.toss_decision, "toss_won_by_team1": int(row.toss_winner == row.team1),
            "team1_form": rate(past1, row.team1), "team2_form": rate(past2, row.team2),
            "form_difference": rate(past1, row.team1) - rate(past2, row.team2),
            "h2h_team1_win_rate": rate(h2h, row.team1), "team1_venue_win_rate": rate(venue1, row.team1),
            "team2_venue_win_rate": rate(venue2, row.team2), "target_team1_win": int(row.winner == row.team1)})
    features = pd.DataFrame(records)
    features.to_csv(os.path.join(PROCESSED_DIR, "match_features.csv"), index=False)
    return features


def model_training_pipeline(features: pd.DataFrame) -> Pipeline | None:
    print("[3/5] Training match-outcome model")
    if len(features) < 20 or features.target_team1_win.nunique() < 2:
        print("[SKIP] Not enough balanced matches to train a model.")
        return None
    df = features.sort_values("date").reset_index(drop=True)
    split = max(1, int(len(df) * .8))
    train, test = df.iloc[:split], df.iloc[split:]
    if test.empty or train.target_team1_win.nunique() < 2:
        print("[SKIP] Training/test split does not contain enough classes.")
        return None
    categorical = ["team1", "team2", "venue", "toss_decision"]
    numerical = ["toss_won_by_team1", "team1_form", "team2_form", "form_difference", "h2h_team1_win_rate", "team1_venue_win_rate", "team2_venue_win_rate"]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    model = Pipeline([("preprocessor", ColumnTransformer([("cat", encoder, categorical), ("num", "passthrough", numerical)])),
                      ("classifier", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42, class_weight="balanced"))])
    model.fit(train[categorical + numerical], train.target_team1_win)
    probabilities = model.predict_proba(test[categorical + numerical])[:, 1]
    predictions = model.predict(test[categorical + numerical])
    print(f"Accuracy: {accuracy_score(test.target_team1_win, predictions):.2%}")
    if test.target_team1_win.nunique() == 2:
        print(f"ROC-AUC:  {roc_auc_score(test.target_team1_win, probabilities):.3f}")
        print(f"Log loss: {log_loss(test.target_team1_win, probabilities, labels=[0, 1]):.3f}")
    print(classification_report(test.target_team1_win, predictions, zero_division=0))
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_EXPORT_PATH)
    return model


def export_dashboard_aggregations(matches: pd.DataFrame, deliveries: pd.DataFrame, features: pd.DataFrame | None = None) -> None:
    print("[4/5] Exporting dashboard tables")
    wicket_kinds = ["bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]
    deliveries["is_bowler_wicket"] = deliveries.get("dismissal_kind", pd.Series(index=deliveries.index, dtype="object")).isin(wicket_kinds)
    batsmen = deliveries.groupby("batsman", as_index=False).agg(total_runs=("batsman_runs", "sum"), balls_faced=("is_legal_ball", "sum"), fours=("batsman_runs", lambda x: x.eq(4).sum()), sixes=("batsman_runs", lambda x: x.eq(6).sum()))
    batsmen["strike_rate"] = 100 * batsmen.total_runs / batsmen.balls_faced.clip(lower=1)
    bowling_runs = deliveries.total_runs - deliveries.bye_runs - deliveries.legbye_runs
    deliveries["bowling_runs_conceded"] = bowling_runs.clip(lower=0)
    bowlers = deliveries.groupby("bowler", as_index=False).agg(wickets=("is_bowler_wicket", "sum"), runs_conceded=("bowling_runs_conceded", "sum"), legal_balls=("is_legal_ball", "sum"))
    bowlers["economy_rate"] = 6 * bowlers.runs_conceded / bowlers.legal_balls.clip(lower=1)
    appearances = pd.concat([matches[["team1"]].rename(columns={"team1": "team"}), matches[["team2"]].rename(columns={"team2": "team"})]).value_counts("team").rename("matches_played")
    wins = matches.winner.value_counts().rename("wins")
    team = pd.concat([appearances, wins], axis=1).fillna(0).reset_index().rename(columns={"index": "team"})
    team["win_percentage"] = 100 * team.wins / team.matches_played
    toss = matches.assign(toss_match_same=matches.toss_winner.eq(matches.winner)).groupby("toss_match_same", as_index=False).size().rename(columns={"size": "matches"})
    venue = matches.groupby("venue", as_index=False).agg(matches=("id", "count"), most_wins=("winner", lambda x: x.value_counts().index[0]))
    innings_scores = deliveries.groupby(["match_id", "inning"], as_index=False).total_runs.sum()
    first_scores = innings_scores[innings_scores.inning.eq(1)].groupby("match_id", as_index=False).total_runs.sum().rename(columns={"total_runs": "first_innings_score"})
    venue = venue.merge(matches[["id", "venue"]].merge(first_scores, left_on="id", right_on="match_id", how="left").groupby("venue", as_index=False).first_innings_score.mean(), on="venue", how="left")
    season = matches.groupby("season", as_index=False).agg(matches=("id", "count"), champion=("winner", lambda x: x.value_counts().index[0]))
    season_runs = deliveries.merge(matches[["id", "season"]], left_on="match_id", right_on="id", how="inner").groupby("season", as_index=False).agg(total_runs=("total_runs", "sum"), total_wickets=("is_wicket", "sum"))
    season = season.merge(season_runs, on="season", how="left")
    overview = pd.DataFrame([{"total_matches": len(matches), "total_runs": int(deliveries.total_runs.sum()), "total_wickets": int(deliveries.is_wicket.sum()), "most_successful_team": wins.index[0], "highest_run_scorer": batsmen.loc[batsmen.total_runs.idxmax(), "batsman"], "highest_wicket_taker": bowlers.loc[bowlers.wickets.idxmax(), "bowler"]}])
    tables = {
        "dashboard_overview": overview,
        "dashboard_team_performance": team.sort_values("wins", ascending=False),
        "dashboard_batsman_stats": batsmen.sort_values("total_runs", ascending=False),
        "dashboard_bowler_stats": bowlers.sort_values("wickets", ascending=False),
        "dashboard_toss_analysis": toss,
        "dashboard_venue_analysis": venue,
        "dashboard_season_analysis": season.sort_values("season"),
        "cleaned_matches": matches,
        "cleaned_deliveries": deliveries,
    }
    if features is not None:
        tables["match_features"] = features
    for name, table in tables.items():
        table.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
    with pd.ExcelWriter(POWERBI_WORKBOOK_PATH, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"[OK] Power BI workbook exported: {POWERBI_WORKBOOK_PATH}")


def predict_match_scenario(details: dict) -> None:
    if not os.path.exists(MODEL_EXPORT_PATH):
        print("[SKIP] No trained model is available for prediction.")
        return
    probability = joblib.load(MODEL_EXPORT_PATH).predict_proba(pd.DataFrame([details]))[0, 1]
    print(f"{details['team1']}: {probability:.1%} win probability")
    print(f"{details['team2']}: {1 - probability:.1%} win probability")


if __name__ == "__main__":
    try:
        print("IPL Analytics & Machine Learning Pipeline")
        create_demo_datasets()
        cleaned_matches, cleaned_deliveries = clean_pipeline()
        feature_data = feature_engineering_pipeline(cleaned_matches)
        model_training_pipeline(feature_data)
        export_dashboard_aggregations(cleaned_matches, cleaned_deliveries, feature_data)
        predict_match_scenario({"team1": "Royal Challengers Bengaluru", "team2": "Chennai Super Kings", "venue": "M. Chinnaswamy Stadium", "toss_decision": "field", "toss_won_by_team1": 1, "team1_form": .8, "team2_form": .6, "form_difference": .2, "h2h_team1_win_rate": .45, "team1_venue_win_rate": .6, "team2_venue_win_rate": .55})
        print("\n[SUCCESS] Project outputs are available in data/processed and models/.")
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)
