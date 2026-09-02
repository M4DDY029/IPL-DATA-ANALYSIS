"""
================================================================================
 IPL Performance Analysis & Match Outcome Prediction
================================================================================
 Final Year College Project
 Data Source  : Cricsheet.org (official IPL ball-by-ball JSON data)
 Dataset Size : 1,218 matches | ~395,000 deliveries | 19 seasons (2008-2026)
 ML Algorithm : Random Forest Classifier
 Output       : Cleaned CSVs + Power BI Excel workbook + Trained ML model

 PIPELINE STAGES:
   Stage 1 -> Data Cleaning
   Stage 2 -> Feature Engineering
   Stage 3 -> Model Training (Random Forest)
   Stage 4 -> Dashboard Export (CSV + Excel for Power BI)
   Stage 5 -> Match Prediction Demo
================================================================================
"""

# -- Standard library imports --------------------------------------------------
from __future__ import annotations
import os, sys, warnings, json, urllib.request, zipfile

# -- Third-party library imports -----------------------------------------------
import joblib                               # Save & load trained ML model
import numpy as np                          # Numerical computations
import pandas as pd                         # Data loading & manipulation

# Scikit-learn: everything related to Machine Learning
from sklearn.compose import ColumnTransformer       # Handle mixed feature types
from sklearn.ensemble import RandomForestClassifier # Our chosen ML algorithm
from sklearn.metrics import (
    accuracy_score,         # % of correct predictions
    classification_report,  # Precision, Recall, F1 per class
    log_loss,               # Confidence of probability predictions
    roc_auc_score,          # Ability to distinguish winners from losers
)
from sklearn.pipeline import Pipeline               # Chain preprocessing + model
from sklearn.preprocessing import OneHotEncoder     # Convert team names to numbers

warnings.filterwarnings("ignore")  # Suppress non-critical scikit-learn warnings

# -- File & Directory Paths ----------------------------------------------------
DATA_DIR            = "data"
PROCESSED_DIR       = os.path.join(DATA_DIR, "processed")
MODELS_DIR          = "models"
MATCHES_RAW_PATH    = os.path.join(DATA_DIR, "matches.csv")
DELIVERIES_RAW_PATH = os.path.join(DATA_DIR, "deliveries.csv")
MODEL_EXPORT_PATH   = os.path.join(MODELS_DIR, "random_forest_ipl.joblib")
CRICSHEET_URL       = "https://cricsheet.org/downloads/ipl_json.zip"
SOURCE_MARKER       = os.path.join(DATA_DIR, ".cricsheet_source")
POWERBI_WORKBOOK    = os.path.join(PROCESSED_DIR, "IPL_PowerBI_Data.xlsx")

# -- Data Standardization Mappings ---------------------------------------------
# Teams changed official names over the years.
# Without this fix, model treats "Delhi Daredevils" and "Delhi Capitals"
# as TWO different teams -- which hurts accuracy.
TEAM_MAPPINGS = {
    "Delhi Daredevils"           : "Delhi Capitals",
    "Deccan Chargers"            : "Sunrisers Hyderabad",
    "Rising Pune Supergiant"     : "Rising Pune Supergiants",
    "Kings XI Punjab"            : "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# The same stadium can appear with slightly different spellings in the dataset.
VENUE_MAPPINGS = {
    "M Chinnaswamy Stadium"                               : "M. Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk"                    : "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai"           : "MA Chidambaram Stadium",
    "Punjab Cricket Association Stadium, Mohali"         : "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal"          : "Rajiv Gandhi International Cricket Stadium",
}


# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Validate required columns exist; raise a clear error if any are missing."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def normalise_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns to consistent internal names.
    Different IPL CSV datasets use different column names for the same data.
    Example: some use 'innings', others use 'inning'; some use 'batter', others 'batsman'.
    """
    rename_map = {
        source: destination
        for source, destination in aliases.items()
        if source in frame.columns and destination not in frame.columns
    }
    return frame.rename(columns=rename_map)


# ==============================================================================
#  DATA ACQUISITION
# ==============================================================================

def download_real_ipl_data() -> None:
    """
    Download official Cricsheet IPL JSON data and convert to CSVs.

    Cricsheet.org provides free ball-by-ball JSON files for every IPL match.
    Each file is one match. We parse every file and extract:
      - Match-level info (teams, date, toss, winner) -> matches.csv
      - Ball-by-ball info (runs, wickets, extras)    -> deliveries.csv
    """
    print("[INFO] Downloading official Cricsheet IPL data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dir = os.path.join(DATA_DIR, "ipl_json")

    # Skip download if JSON files are already extracted
    json_available = (
        os.path.isdir(extract_dir) and
        any(name.endswith(".json") for name in os.listdir(extract_dir))
    )
    if not json_available:
        archive = os.path.join(DATA_DIR, "ipl_json.zip")
        try:
            urllib.request.urlretrieve(CRICSHEET_URL, archive)
        except Exception as error:
            raise RuntimeError(
                "Could not download Cricsheet data. "
                "Download ipl_json.zip manually from https://cricsheet.org/downloads/ "
                "and place it in data/."
            ) from error
        with zipfile.ZipFile(archive) as source:
            source.extractall(extract_dir)

    matches, deliveries = [], []
    for path in sorted(os.listdir(extract_dir)):
        if not path.endswith(".json"):
            continue
        with open(os.path.join(extract_dir, path), encoding="utf-8") as handle:
            game = json.load(handle)
        info    = game.get("info", {})
        teams   = info.get("teams", [])
        outcome = info.get("outcome", {})
        winner  = outcome.get("winner")
        # Skip abandoned / tied matches (no winner declared)
        if len(teams) != 2 or not winner:
            continue
        match_id = os.path.splitext(path)[0]
        dates = info.get("dates", [])
        matches.append({
            "id": match_id, "date": dates[0] if dates else None,
            "season": info.get("season"),
            "team1": teams[0], "team2": teams[1],
            "venue": info.get("venue", info.get("city", "Unknown venue")),
            "toss_winner": info.get("toss", {}).get("winner"),
            "toss_decision": info.get("toss", {}).get("decision"),
            "winner": winner, "dl_applied": 0,
        })
        for inning_no, inning in enumerate(game.get("innings", []), 1):
            batting = inning.get("team", "Unknown")
            bowling = teams[1] if batting == teams[0] else teams[0]
            for over in inning.get("overs", []):
                for ball_no, ball in enumerate(over.get("deliveries", []), 1):
                    runs = ball.get("runs", {})
                    extras = ball.get("extras", {})
                    wickets = ball.get("wickets", [])
                    first_wicket = wickets[0] if wickets else {}
                    deliveries.append({
                        "match_id": match_id, "inning": inning_no,
                        "batting_team": batting, "bowling_team": bowling,
                        "over": over.get("over", 0), "ball": ball_no,
                        "batsman": ball.get("batter"), "bowler": ball.get("bowler"),
                        "batsman_runs": runs.get("batter", 0),
                        "extra_runs": runs.get("extras", 0),
                        "total_runs": runs.get("total", 0),
                        "wide_runs": extras.get("wides", 0),
                        "noball_runs": extras.get("noballs", 0),
                        "bye_runs": extras.get("byes", 0),
                        "legbye_runs": extras.get("legbyes", 0),
                        "is_wicket": int(bool(wickets)),
                        "dismissal_kind": first_wicket.get("kind"),
                        "player_dismissed": first_wicket.get("player_out"),
                    })
    if not matches or not deliveries:
        raise ValueError("Cricsheet download contained no usable IPL matches.")
    pd.DataFrame(matches).to_csv(MATCHES_RAW_PATH, index=False)
    pd.DataFrame(deliveries).to_csv(DELIVERIES_RAW_PATH, index=False)
    with open(SOURCE_MARKER, "w", encoding="utf-8") as marker:
        marker.write("Official Cricsheet IPL JSON; https://cricsheet.org/downloads/ipl_json.zip\n")
    print(f"[OK] Real data converted: {len(matches):,} matches, {len(deliveries):,} deliveries.")


def create_demo_datasets() -> None:
    """Entry point: use existing data or download fresh from Cricsheet."""
    matches_exists    = os.path.exists(MATCHES_RAW_PATH)
    deliveries_exists = os.path.exists(DELIVERIES_RAW_PATH)
    if matches_exists and deliveries_exists:
        if os.path.exists(SOURCE_MARKER):
            print("[OK] Official Cricsheet IPL datasets found.")
        else:
            print("[INFO] Replacing with official Cricsheet data.")
            download_real_ipl_data()
        return
    if matches_exists != deliveries_exists:
        raise FileNotFoundError(
            "Both data/matches.csv and data/deliveries.csv are required. "
            "One exists, so no demo files were generated to avoid overwriting it."
        )
    download_real_ipl_data()


# ==============================================================================
#  STAGE 1: DATA CLEANING
# ==============================================================================

def clean_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    STAGE 1: Load and clean raw IPL datasets.

    Cleaning steps:
      1. Load raw CSVs into DataFrames
      2. Normalize column names (handle different CSV format variants)
      3. Validate all required columns are present
      4. Remove matches with no winner (abandoned, tied)
      5. Remove Duckworth-Lewis (D/L) rain-affected matches
         (D/L matches use an adjusted target, making results unreliable for ML)
      6. Standardize team names using TEAM_MAPPINGS
      7. Standardize venue names using VENUE_MAPPINGS
      8. Parse date column to proper datetime type
      9. Add/derive 'season' year column
     10. Add 'over_phase' column:
            Overs  0-5  -> Powerplay     (fielding restrictions)
            Overs  6-14 -> Middle Overs  (consolidation)
            Overs 15-20 -> Death Overs   (slog phase)
     11. Add 'is_legal_ball' column:
            True  if ball is NOT a wide or no-ball
            Used to correctly calculate strike rate and economy rate

    Returns:
        Tuple of (cleaned_matches, cleaned_deliveries) DataFrames
    """
    print("\n[1/5] Cleaning data")

    matches    = pd.read_csv(MATCHES_RAW_PATH)
    deliveries = pd.read_csv(DELIVERIES_RAW_PATH)

    # Normalize column names to handle different CSV format variants
    matches = normalise_columns(matches, {
        "match_id": "id", "match_date": "date",
        "match_winner": "winner", "venue_name": "venue",
    })
    deliveries = normalise_columns(deliveries, {
        "innings": "inning", "batter": "batsman", "batsman_name": "batsman",
        "bowler_name": "bowler", "runs_off_bat": "batsman_runs",
        "total_run": "total_runs", "extras": "extra_runs",
        "isWicketDelivery": "is_wicket",
    })

    require_columns(matches,    ["id", "date", "team1", "team2", "venue",
                                 "toss_winner", "toss_decision", "winner"], "matches.csv")
    require_columns(deliveries, ["match_id", "batting_team", "bowling_team",
                                 "batsman", "bowler", "batsman_runs"],       "deliveries.csv")

    # Remove matches with no winner and D/L affected matches
    matches = matches.dropna(subset=["winner"]).copy()
    if "dl_applied" in matches:
        matches = matches[matches["dl_applied"].fillna(0).eq(0)].copy()

    # Standardize team and venue names
    for col in ["team1", "team2", "toss_winner", "winner"]:
        matches[col] = matches[col].replace(TEAM_MAPPINGS)
    matches["venue"] = matches["venue"].replace(VENUE_MAPPINGS).fillna("Unknown venue")

    # Fix date formatting and sort chronologically
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Derive season year from date
    if "season" in matches:
        matches["season"] = pd.to_numeric(matches["season"], errors="coerce")
        matches["season"] = matches["season"].fillna(matches["date"].dt.year).astype(int)
    else:
        matches["season"] = matches["date"].dt.year.astype(int)

    # Clean deliveries: standardize teams, fix numeric columns
    for col in ["batting_team", "bowling_team"]:
        deliveries[col] = deliveries[col].replace(TEAM_MAPPINGS)
    for col in ["extra_runs", "wide_runs", "noball_runs",
                "bye_runs", "legbye_runs", "is_wicket"]:
        if col not in deliveries:
            deliveries[col] = 0
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce").fillna(0)
    deliveries["batsman_runs"] = pd.to_numeric(
        deliveries["batsman_runs"], errors="coerce").fillna(0)
    deliveries["total_runs"] = pd.to_numeric(
        deliveries.get("total_runs", deliveries["batsman_runs"] + deliveries["extra_runs"]),
        errors="coerce").fillna(0)

    # is_legal_ball: wides and no-balls don't count as balls faced by the batsman
    deliveries["is_legal_ball"] = (
        (deliveries["wide_runs"] == 0) & (deliveries["noball_runs"] == 0)
    )

    # over_phase: categorize overs into T20 game phases
    over_col = "over" if "over" in deliveries else "match_over"
    if over_col in deliveries:
        deliveries["over_phase"] = pd.cut(
            deliveries[over_col],
            bins=[-1, 5, 14, 20],
            labels=["Powerplay", "Middle Overs", "Death Overs"]
        )
    else:
        deliveries["over_phase"] = "Unknown"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    matches.to_csv(os.path.join(PROCESSED_DIR, "cleaned_matches.csv"), index=False)
    deliveries.to_csv(os.path.join(PROCESSED_DIR, "cleaned_deliveries.csv"), index=False)

    print(f"    Matches   : {len(matches):,} rows")
    print(f"    Deliveries: {len(deliveries):,} rows")
    return matches, deliveries


# ==============================================================================
#  STAGE 2: FEATURE ENGINEERING
# ==============================================================================

def _past_matches(df: pd.DataFrame, team: str, date: pd.Timestamp) -> pd.DataFrame:
    """
    Return all matches played by 'team' strictly BEFORE 'date'.
    The strict date filter prevents data leakage in feature computation.
    """
    return df[(df.date < date) & ((df.team1 == team) | (df.team2 == team))]


def feature_engineering_pipeline(matches: pd.DataFrame) -> pd.DataFrame:
    """
    STAGE 2: Build pre-match features for the Machine Learning model.

    For every match, compute features using ONLY historical data
    (matches that occurred BEFORE the current match date).
    This strict constraint prevents data leakage -- no future information
    is ever used to compute a feature.

    Features computed per match:
    -------------------------------------------------------------------------
      team1_form           - Team 1 win rate in last 5 matches (recent form)
      team2_form           - Team 2 win rate in last 5 matches
      form_difference      - team1_form minus team2_form (relative momentum)
      h2h_team1_win_rate   - Team 1's historical win rate vs Team 2 only
      team1_venue_win_rate - Team 1's win rate at this specific venue
      team2_venue_win_rate - Team 2's win rate at this specific venue
      toss_won_by_team1    - 1 if Team 1 won the toss, 0 otherwise
      toss_decision        - 'bat' or 'field' (toss winner's choice)
      target_team1_win     - 1 if Team 1 won this match (ML label / target)

    Default win rate = 0.5 when no historical data exists (neutral assumption).

    Returns:
        DataFrame with one row per match containing all features.
    """
    print("[2/5] Building pre-match features")

    records  = []
    win_rate = lambda df, team: (df.winner == team).mean() if len(df) else 0.5

    for row in matches.sort_values("date").itertuples(index=False):
        past1  = _past_matches(matches, row.team1, row.date).tail(5)
        past2  = _past_matches(matches, row.team2, row.date).tail(5)
        h2h    = matches[
            (matches.date < row.date) &
            (
                ((matches.team1 == row.team1) & (matches.team2 == row.team2)) |
                ((matches.team1 == row.team2) & (matches.team2 == row.team1))
            )
        ]
        venue1 = _past_matches(matches, row.team1, row.date)
        venue1 = venue1[venue1["venue"].eq(row.venue)]
        venue2 = _past_matches(matches, row.team2, row.date)
        venue2 = venue2[venue2["venue"].eq(row.venue)]

        records.append({
            "date"                : row.date,
            "team1"               : row.team1,
            "team2"               : row.team2,
            "venue"               : row.venue,
            "toss_decision"       : row.toss_decision,
            "toss_won_by_team1"   : int(row.toss_winner == row.team1),
            "team1_form"          : win_rate(past1, row.team1),
            "team2_form"          : win_rate(past2, row.team2),
            "form_difference"     : win_rate(past1, row.team1) - win_rate(past2, row.team2),
            "h2h_team1_win_rate"  : win_rate(h2h, row.team1),
            "team1_venue_win_rate": win_rate(venue1, row.team1),
            "team2_venue_win_rate": win_rate(venue2, row.team2),
            "target_team1_win"    : int(row.winner == row.team1),  # ML target label
        })

    features = pd.DataFrame(records)
    features.to_csv(os.path.join(PROCESSED_DIR, "match_features.csv"), index=False)
    print(f"    Features built for {len(features):,} matches")
    return features


# ==============================================================================
#  STAGE 3: MACHINE LEARNING MODEL TRAINING
# ==============================================================================

def model_training_pipeline(features: pd.DataFrame) -> Pipeline | None:
    """
    STAGE 3: Train a Random Forest Classifier to predict match winners.

    ALGORITHM CHOICE: Random Forest
    - Ensemble of 300 decision trees; majority vote = final prediction
    - Chosen because:
        * Handles both categorical (team names) and numerical (rates) features
        * Resistant to overfitting (averaging across many trees)
        * Produces win probabilities (not just Yes/No)
        * No feature scaling required
        * Built-in feature importance ranking

    TRAIN/TEST SPLIT: 80% / 20% CHRONOLOGICALLY
    - First 80% of matches (by date) -> training set
    - Last  20% of matches           -> test set
    - NOT a random split! Mimics real-world: train on past, predict future.

    PREPROCESSING PIPELINE:
    - Categorical features (team names, venue, toss decision)
        -> OneHotEncoder: converts text to binary columns
           Example: "Chennai Super Kings" -> [0, 0, 1, 0, ...]
    - Numerical features (win rates, form, h2h)
        -> PassThrough: used as-is (already in 0-1 range)

    EVALUATION METRICS:
    - Accuracy  : % of match winners predicted correctly
    - ROC-AUC   : How well model separates winners (0.5=random, 1.0=perfect)
    - Log Loss  : Penalizes confident wrong predictions (lower = better)

    Returns:
        Trained scikit-learn Pipeline object, or None if training was skipped.
    """
    print("[3/5] Training match-outcome model")

    if len(features) < 20 or features.target_team1_win.nunique() < 2:
        print("[SKIP] Not enough balanced data to train.")
        return None

    df    = features.sort_values("date").reset_index(drop=True)
    split = max(1, int(len(df) * 0.8))
    train = df.iloc[:split]
    test  = df.iloc[split:]

    if test.empty or train.target_team1_win.nunique() < 2:
        print("[SKIP] Training/test split does not have enough classes.")
        return None

    # Define which columns are categorical vs numerical
    categorical_features = ["team1", "team2", "venue", "toss_decision"]
    numerical_features   = [
        "toss_won_by_team1",
        "team1_form", "team2_form", "form_difference",
        "h2h_team1_win_rate",
        "team1_venue_win_rate", "team2_venue_win_rate",
    ]
    all_features = categorical_features + numerical_features

    # OneHotEncoder: converts "Chennai Super Kings" -> [0, 1, 0, ...]
    # handle_unknown="ignore" prevents crashes on new teams in test data
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # scikit-learn Pipeline: ensures same preprocessing at train AND predict time
    model = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("cat", encoder,       categorical_features),
            ("num", "passthrough", numerical_features),
        ])),
        ("classifier", RandomForestClassifier(
            n_estimators     = 300,       # 300 trees -> stable ensemble
            max_depth        = 8,         # Limit depth to prevent overfitting
            min_samples_leaf = 2,         # Minimum 2 samples per leaf node
            random_state     = 42,        # Fixed seed for reproducibility
            class_weight     = "balanced",# Compensate for unequal class counts
        )),
    ])

    model.fit(train[all_features], train.target_team1_win)

    probabilities = model.predict_proba(test[all_features])[:, 1]
    predictions   = model.predict(test[all_features])

    print(f"\n    -- Model Evaluation ({len(test)} test matches) --")
    print(f"    Accuracy : {accuracy_score(test.target_team1_win, predictions):.2%}")
    if test.target_team1_win.nunique() == 2:
        print(f"    ROC-AUC  : {roc_auc_score(test.target_team1_win, probabilities):.3f}")
        print(f"    Log Loss : {log_loss(test.target_team1_win, probabilities, labels=[0,1]):.3f}")
    print()
    print(classification_report(test.target_team1_win, predictions, zero_division=0))

    # Save model to disk using joblib (efficient for numpy-heavy objects)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_EXPORT_PATH)
    print(f"    Model saved: {MODEL_EXPORT_PATH}")
    return model


# ==============================================================================
#  STAGE 4: DASHBOARD EXPORT
# ==============================================================================

def export_dashboard_aggregations(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    features: pd.DataFrame | None = None
) -> None:
    """
    STAGE 4: Compute all analysis tables and export to CSV + Excel.

    8 output tables:
    - dashboard_overview         : Total stats, top team/player
    - dashboard_team_performance : Win counts and win% per team
    - dashboard_batsman_stats    : Runs, strike rate, 4s, 6s per batsman
    - dashboard_bowler_stats     : Wickets and economy rate per bowler
    - dashboard_toss_analysis    : Does toss winner = match winner?
    - dashboard_venue_analysis   : Matches per venue, average 1st innings score
    - dashboard_season_analysis  : Season champion, runs, wickets per year
    - match_features             : ML feature table (if provided)

    All tables are also written to IPL_PowerBI_Data.xlsx
    (one sheet per table, ready for Power BI or Tableau import).
    """
    print("[4/5] Exporting dashboard tables")

    # Bowler wickets: only these types count for the bowler's wicket tally
    # (run-outs, obstructing the field, handling the ball -> NOT bowler's wicket)
    bowler_wicket_types = ["bowled", "caught", "lbw", "stumped",
                           "caught and bowled", "hit wicket"]
    deliveries["is_bowler_wicket"] = deliveries.get(
        "dismissal_kind",
        pd.Series(index=deliveries.index, dtype="object")
    ).isin(bowler_wicket_types)

    # Batsman stats: runs, balls, boundary count, strike rate
    batsmen = deliveries.groupby("batsman", as_index=False).agg(
        total_runs  = ("batsman_runs",  "sum"),
        balls_faced = ("is_legal_ball", "sum"),
        fours       = ("batsman_runs",  lambda x: x.eq(4).sum()),
        sixes       = ("batsman_runs",  lambda x: x.eq(6).sum()),
    )
    batsmen["strike_rate"] = 100 * batsmen.total_runs / batsmen.balls_faced.clip(lower=1)

    # Bowler stats: wickets, economy rate
    # Economy = (runs/balls) * 6; exclude byes/legbyes (not bowler's fault)
    bowling_runs = deliveries.total_runs - deliveries.bye_runs - deliveries.legbye_runs
    deliveries["bowling_runs_conceded"] = bowling_runs.clip(lower=0)
    bowlers = deliveries.groupby("bowler", as_index=False).agg(
        wickets       = ("is_bowler_wicket",     "sum"),
        runs_conceded = ("bowling_runs_conceded", "sum"),
        legal_balls   = ("is_legal_ball",         "sum"),
    )
    bowlers["economy_rate"] = 6 * bowlers.runs_conceded / bowlers.legal_balls.clip(lower=1)

    # Team performance
    appearances = pd.concat([
        matches[["team1"]].rename(columns={"team1": "team"}),
        matches[["team2"]].rename(columns={"team2": "team"}),
    ]).value_counts("team").rename("matches_played")
    wins = matches.winner.value_counts().rename("wins")
    team = pd.concat([appearances, wins], axis=1).fillna(0).reset_index()
    team = team.rename(columns={"index": "team"})
    team["win_percentage"] = 100 * team.wins / team.matches_played

    # Toss analysis: does winning the toss mean winning the match?
    toss = matches.assign(
        toss_match_same=matches.toss_winner.eq(matches.winner)
    ).groupby("toss_match_same", as_index=False).size().rename(columns={"size": "matches"})

    # Venue analysis with average first innings score
    venue = matches.groupby("venue", as_index=False).agg(
        matches   = ("id",     "count"),
        most_wins = ("winner", lambda x: x.value_counts().index[0]),
    )
    innings_scores = deliveries.groupby(["match_id", "inning"], as_index=False).total_runs.sum()
    first_scores = (
        innings_scores[innings_scores.inning.eq(1)]
        .groupby("match_id", as_index=False).total_runs.sum()
        .rename(columns={"total_runs": "first_innings_score"})
    )
    venue = venue.merge(
        matches[["id", "venue"]]
        .merge(first_scores, left_on="id", right_on="match_id", how="left")
        .groupby("venue", as_index=False).first_innings_score.mean(),
        on="venue", how="left"
    )

    # Season analysis: champion, total runs, total wickets per season
    season = matches.groupby("season", as_index=False).agg(
        matches  = ("id",     "count"),
        champion = ("winner", lambda x: x.value_counts().index[0]),
    )
    season_runs = (
        deliveries.merge(matches[["id", "season"]], left_on="match_id", right_on="id", how="inner")
        .groupby("season", as_index=False)
        .agg(total_runs=("total_runs", "sum"), total_wickets=("is_wicket", "sum"))
    )
    season = season.merge(season_runs, on="season", how="left")

    # Overview KPIs
    overview = pd.DataFrame([{
        "total_matches"       : len(matches),
        "total_runs"          : int(deliveries.total_runs.sum()),
        "total_wickets"       : int(deliveries.is_wicket.sum()),
        "most_successful_team": wins.index[0],
        "highest_run_scorer"  : batsmen.loc[batsmen.total_runs.idxmax(), "batsman"],
        "highest_wicket_taker": bowlers.loc[bowlers.wickets.idxmax(), "bowler"],
    }])

    tables = {
        "dashboard_overview"         : overview,
        "dashboard_team_performance" : team.sort_values("wins", ascending=False),
        "dashboard_batsman_stats"    : batsmen.sort_values("total_runs", ascending=False),
        "dashboard_bowler_stats"     : bowlers.sort_values("wickets", ascending=False),
        "dashboard_toss_analysis"    : toss,
        "dashboard_venue_analysis"   : venue,
        "dashboard_season_analysis"  : season.sort_values("season"),
        "cleaned_matches"            : matches,
        "cleaned_deliveries"         : deliveries,
    }
    if features is not None:
        tables["match_features"] = features

    for name, table in tables.items():
        table.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)

    # Sheet names capped at 31 characters (Excel limitation)
    with pd.ExcelWriter(POWERBI_WORKBOOK, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)

    print(f"    {len(tables)} tables exported to: {PROCESSED_DIR}/")
    print(f"    Power BI workbook: {POWERBI_WORKBOOK}")


# ==============================================================================
#  STAGE 5: MATCH PREDICTION DEMO
# ==============================================================================

def predict_match_scenario(details: dict) -> None:
    """
    STAGE 5: Load the saved model and predict win probabilities.

    The model returns a probability (0 to 1) that Team 1 wins.
    We show it as a percentage for each team.

    Args:
        details: Dictionary with pre-match feature values.
    """
    if not os.path.exists(MODEL_EXPORT_PATH):
        print("[SKIP] No trained model found.")
        return
    loaded_model = joblib.load(MODEL_EXPORT_PATH)
    probability  = loaded_model.predict_proba(pd.DataFrame([details]))[0, 1]
    print(f"\n    Match Prediction:")
    print(f"    {details['team1']:<35} -> {probability:.1%} win probability")
    print(f"    {details['team2']:<35} -> {1 - probability:.1%} win probability")


# ==============================================================================
#  MAIN ENTRY POINT — runs all 5 stages in sequence
# ==============================================================================

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  IPL Analytics & Machine Learning Pipeline")
        print("=" * 60)

        create_demo_datasets()                                           # Stage 0
        cleaned_matches, cleaned_deliveries = clean_pipeline()          # Stage 1
        feature_data = feature_engineering_pipeline(cleaned_matches)    # Stage 2
        model_training_pipeline(feature_data)                           # Stage 3
        export_dashboard_aggregations(                                   # Stage 4
            cleaned_matches, cleaned_deliveries, feature_data)

        # Stage 5: Demo prediction for RCB vs CSK at Chinnaswamy Stadium
        predict_match_scenario({
            "team1"               : "Royal Challengers Bengaluru",
            "team2"               : "Chennai Super Kings",
            "venue"               : "M. Chinnaswamy Stadium",
            "toss_decision"       : "field",
            "toss_won_by_team1"   : 1,     # RCB won the toss
            "team1_form"          : 0.8,   # RCB won 4 of last 5 matches
            "team2_form"          : 0.6,   # CSK won 3 of last 5 matches
            "form_difference"     : 0.2,
            "h2h_team1_win_rate"  : 0.45,  # CSK has slight historical edge
            "team1_venue_win_rate": 0.6,   # RCB strong at home (Chinnaswamy)
            "team2_venue_win_rate": 0.55,
        })

        print("\n" + "=" * 60)
        print("  [SUCCESS] All outputs ready!")
        print(f"  Processed data: data/processed/")
        print(f"  Power BI file : {POWERBI_WORKBOOK}")
        print(f"  Trained model : {MODEL_EXPORT_PATH}")
        print("=" * 60)

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)
