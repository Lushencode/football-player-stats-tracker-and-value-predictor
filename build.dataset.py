import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading players...")
players = pd.read_csv("data/archive/players.csv")

print("Loading appearances...")
appearances = pd.read_csv("data/archive/appearances.csv")



print("Loading games...")
games = pd.read_csv("data/archive/games.csv")



print("Loading valuations...")
valuations = pd.read_csv("data/archive/player_valuations.csv")


# ==========================================
# 2. CONVERT DATES
# ==========================================

players["date_of_birth"] = pd.to_datetime(
    players["date_of_birth"],
    errors="coerce"
)

appearances["date"] = pd.to_datetime(
    appearances["date"],
    errors="coerce"
)

games["date"] = pd.to_datetime(
    games["date"],
    errors="coerce"
)

valuations["date"] = pd.to_datetime(
    valuations["date"],
    errors="coerce"
)


# ==========================================
# 3. CHECK DATES
# ==========================================

print("\nPlayers:")
print(players["date_of_birth"].min())
print(players["date_of_birth"].max())

print("\nAppearances:")
print(appearances["date"].min())
print(appearances["date"].max())

print("\nGames:")
print(games["date"].min())
print(games["date"].max())

print("\nValuations:")
print(valuations["date"].min())
print(valuations["date"].max())

# ==========================================
# 4. JOIN APPEARANCES WITH GAMES
# ==========================================

print("\nJoining appearances with games...")

appearance_data = appearances.merge(
    games[["game_id", "season"]],
    on="game_id",
    how="left"
)

print("Joined shape:", appearance_data.shape)

print("\nExample:")
print(
    appearance_data[
        [
            "player_id",
            "player_name",
            "game_id",
            "season",
            "goals",
            "assists",
            "minutes_played"
        ]
    ].head()
)

print("\nCalculating clean sheets...")

# Add game information to each player appearance
appearances_with_games = appearances.merge(
    games[
        [
            "game_id",
            "home_club_id",
            "away_club_id",
            "home_club_goals",
            "away_club_goals",
            "season"
        ]
    ],
    on="game_id",
    how="left"
)

# Add player position
appearances_with_games = appearances_with_games.merge(
    players[
        [
            "player_id",
            "position"
        ]
    ],
    on="player_id",
    how="left"
)

# Positions that can receive clean-sheet credit
eligible_positions = [
    "Goalkeeper",
    "Defender",
    "Midfielder"
]

# Check whether the player's club kept a clean sheet
appearances_with_games["team_clean_sheet"] = (
    (
        (appearances_with_games["player_club_id"] == appearances_with_games["home_club_id"]) &
        (appearances_with_games["away_club_goals"] == 0)
    )
    |
    (
        (appearances_with_games["player_club_id"] == appearances_with_games["away_club_id"]) &
        (appearances_with_games["home_club_goals"] == 0)
    )
)

# Player must:
# 1. Be a goalkeeper, defender or midfielder
# 2. Play MORE than half of the match
# 3. Be on a team that kept a clean sheet
#
# More than half of a 90-minute match = more than 45 minutes.
#
# We use > 45 rather than >= 45 because you specifically said
# "more than half".

appearances_with_games["clean_sheet"] = np.where(
    (
        appearances_with_games["position"].isin(eligible_positions)
    )
    &
    (
        appearances_with_games["minutes_played"] > 45
    )
    &
    (
        appearances_with_games["team_clean_sheet"]
    ),
    1,
    0
)

print("Clean sheets calculated!")

clean_sheets_per_season = (
    appearances_with_games
    .groupby(
        ["player_id", "season"],
        as_index=False
    )["clean_sheet"]
    .sum()
    .rename(
        columns={
            "clean_sheet": "clean_sheets"
        }
    )
)

print("\nClean sheets per season:")
print(clean_sheets_per_season.head(20))


print("\nClean sheets per season:")
print(clean_sheets_per_season.head(20))
# ==========================================
# 5. CREATE PLAYER-SEASON STATISTICS
# ==========================================



print("\nCreating player-season statistics...")

player_season = (
    appearance_data
    .groupby(["player_id", "season"])
    .agg(
        appearances=("game_id", "nunique"),
        minutes_played=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum")
    )
    .reset_index()
)

print("\nPlayer-season shape:")
print(player_season.shape)

print("\nPlayer-season example:")
print(player_season.head(10))

#adding clean sheets
print("\nAdding clean sheets...")

player_season = player_season.merge(
    clean_sheets_per_season,
    on=["player_id", "season"],
    how="left"
)

player_season["clean_sheets"] = (
    player_season["clean_sheets"].fillna(0)
)

print("\nPlayer-season with clean sheets:")
print(
    player_season[
        [
            "player_id",
            "season",
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "clean_sheets"
        ]
    ].head(20)
)

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================

player_season["goal_contributions"] = (
    player_season["goals"] +
    player_season["assists"]
)

# Only calculate per-90 statistics when
# the player has at least 450 minutes.

minimum_minutes = 450

valid_minutes = (
    player_season["minutes_played"] >= minimum_minutes
)

player_season["goals_per_90"] = pd.NA
player_season["assists_per_90"] = pd.NA
player_season["goal_contributions_per_90"] = pd.NA

player_season.loc[valid_minutes, "goals_per_90"] = (
    player_season.loc[valid_minutes, "goals"] /
    player_season.loc[valid_minutes, "minutes_played"] *
    90
)

player_season.loc[valid_minutes, "assists_per_90"] = (
    player_season.loc[valid_minutes, "assists"] /
    player_season.loc[valid_minutes, "minutes_played"] *
    90
)

player_season.loc[valid_minutes, "goal_contributions_per_90"] = (
    player_season.loc[valid_minutes, "goal_contributions"] /
    player_season.loc[valid_minutes, "minutes_played"] *
    90
)

print("\nMINUTES DISTRIBUTION")

print(
    player_season["minutes_played"].describe()
)

print("\nPLAYER-SEASONS WITH AT LEAST 450 MINUTES:")

print(
    (player_season["minutes_played"] >= 450).sum()
)

print("\nPLAYER-SEASONS WITH AT LEAST 900 MINUTES:")

print(
    (player_season["minutes_played"] >= 900).sum()
)

print("\nCORRECTED PER-90 STATISTICS")

print(
    player_season[
        [
            "goals_per_90",
            "assists_per_90",
            "goal_contributions_per_90"
        ]
    ].describe()
)

# ==========================================
# 7. ADD PLAYER INFORMATION
# ==========================================

player_info = players[
    [
        "player_id",
        "name",
        "date_of_birth",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "international_caps",
        "international_goals",
        "image_url"
    ]
].copy()

player_season = player_season.merge(
    player_info,
    on="player_id",
    how="left"
)

# ==========================================
# 8. CALCULATE AGE
# ==========================================

player_season["season_start_date"] = pd.to_datetime(
    player_season["season"].astype(str) + "-07-01",
    errors="coerce"
)

player_season["age"] = (
    player_season["season_start_date"] -
    player_season["date_of_birth"]
).dt.days / 365.25


print("\nFINAL PLAYER-SEASON DATA")

print(player_season.shape)

print(
    player_season[
        [
            "player_id",
            "name",
            "season",
            "clean_sheets",
            "age",
            "position",
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "goal_contributions",
            "goals_per_90",
            "assists_per_90",
            "goal_contributions_per_90"
        ]
    ].head(20)
)

player_season.to_csv(
    "data/player_season.csv",
    index=False
)

# ==========================================
# 9. INVESTIGATE PLAYER VALUATIONS
# ==========================================

print("\nVALUATION DATE RANGE")
print(valuations["date"].min())
print(valuations["date"].max())

print("\nVALUATIONS PER PLAYER")

valuation_counts = valuations.groupby("player_id").size()

print(valuation_counts.describe())

print("\nMOST VALUED PLAYERS")

print(
    valuations[
        ["player_id", "date", "market_value_in_eur", "current_club_name"]
    ]
    .sort_values("market_value_in_eur", ascending=False)
    .head(20)
)

# ==========================================
# 10. VALUATIONS BY YEAR
# ==========================================

valuations["year"] = valuations["date"].dt.year

print("\nVALUATIONS BY YEAR")

print(
    valuations["year"]
    .value_counts()
    .sort_index()
)

# ==========================================
# 11. VALUATION EXAMPLE FOR ONE PLAYER
# ==========================================

example_player = 342229

print("\nVALUATION HISTORY FOR PLAYER", example_player)

print(
    valuations[
        valuations["player_id"] == example_player
    ][
        [
            "player_id",
            "date",
            "market_value_in_eur",
            "current_club_name"
        ]
    ]
    .sort_values("date")
    .to_string(index=False)
)

print("\nPLAYER-SEASON DATE CHECK")

print(
    player_season[
        [
            "player_id",
            "name",
            "season",
            "age",
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "goal_contributions",
            "goals_per_90",
            "assists_per_90"
        ]
    ].head(20).to_string(index=False)
)

print("\nPLAYER-SEASON STATISTICS")

print(
    player_season[
        [
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "goal_contributions",
            "goals_per_90",
            "assists_per_90",
            "goal_contributions_per_90"
        ]
    ].describe()
)
# ============================================================
# STEP 4.1: PREPARE VALUATION DATA
# ============================================================

print("\nPreparing valuation data...")

# Make sure valuation dates are datetime objects
valuations["date"] = pd.to_datetime(valuations["date"])

# Extract the year from each valuation date
valuations["year"] = valuations["date"].dt.year

print("\nValuation years:")
print(valuations["year"].min(), "to", valuations["year"].max())

# ============================================================
# STEP 4.2: GET ONE VALUATION PER PLAYER PER YEAR
# ============================================================

print("\nCreating yearly player valuations...")

yearly_valuations = (
    valuations
    .sort_values(["player_id", "date"])
    .groupby(["player_id", "year"], as_index=False)
    .last()
)

print("\nYearly valuations shape:")
print(yearly_valuations.shape)

print("\nExample yearly valuations:")
print(
    yearly_valuations[
        ["player_id", "year", "market_value_in_eur", "current_club_name"]
    ].head(20)
)
#historic value
print("\nCreating historical value trend...")

yearly_valuations = yearly_valuations.sort_values(
    ["player_id", "year"]
)

yearly_valuations["previous_market_value"] = (
    yearly_valuations
    .groupby("player_id")["market_value_in_eur"]
    .shift(1)
)

yearly_valuations["value_change"] = (
    yearly_valuations["market_value_in_eur"]
    - yearly_valuations["previous_market_value"]
)

yearly_valuations["value_change_percent"] = (
    yearly_valuations["value_change"]
    / yearly_valuations["previous_market_value"].replace(0, pd.NA)
)

print("\nValue trend example:")
print(
    yearly_valuations[
        [
            "player_id",
            "year",
            "market_value_in_eur",
            "previous_market_value",
            "value_change_percent"
        ]
    ].head(20)
)


# ============================================================
# STEP 4.3: ADD CURRENT MARKET VALUE TO PLAYER-SEASON DATA
# ============================================================

print("\nAdding current market value...")

current_valuations = yearly_valuations[
    [
        "player_id",
        "year",
        "market_value_in_eur",
        "current_club_name",
        "previous_market_value",
        "value_change_percent"
    ]
].rename(
    columns={
        "year": "season",
        "market_value_in_eur": "current_market_value"
    }
)

dataset = player_season.merge(
    current_valuations,
    on=["player_id", "season"],
    how="left"
)

print("\nDataset shape:")
print(dataset.shape)

print("\nCurrent market value availability:")
print(dataset["current_market_value"].notna().sum(), "available")
print(dataset["current_market_value"].isna().sum(), "missing")

print("\nExample:")
print(
    dataset[
        [
            "player_id",
            "name",
            "season",
            "age",
            "current_market_value",
            "current_club_name"
        ]
    ].head(20)
)

# ============================================================
# STEP 4.4: CREATE FUTURE MARKET VALUE TARGETS
# ============================================================

print("\nCreating future market value targets...")

future_valuations = yearly_valuations[
    [
        "player_id",
        "year",
        "market_value_in_eur"
    ]
].rename(
    columns={
        "year": "future_season",
        "market_value_in_eur": "future_market_value"
    }
)

# Create multiple future horizons
future_datasets = []

for years_ahead in [1, 2, 3, 4, 5]:

    temp = dataset.copy()

    # Calculate the season we want to predict
    temp["future_season"] = temp["season"] + years_ahead

    # Match the future valuation
    temp = temp.merge(
        future_valuations,
        on=["player_id", "future_season"],
        how="left"
    )

    # Record how far into the future we're predicting
    temp["years_ahead"] = years_ahead

    future_datasets.append(temp)

# Combine all prediction horizons
future_dataset = pd.concat(
    future_datasets,
    ignore_index=True
)

print("\nFuture dataset shape:")
print(future_dataset.shape)

print("\nFuture dataset example:")
print(
    future_dataset[
        [
            "player_id",
            "name",
            "season",
            "age",
            "goals",
            "assists",
            "current_market_value",
            "previous_market_value",
            "value_change_percent",
            "years_ahead",
            "future_season",
            "future_market_value"
        ]
    ].head(20)
)


# ============================================================
# STEP 4.5: REMOVE MISSING FUTURE VALUES
# ============================================================

print("\nChecking future market values...")

print(
    "Rows before removing missing future values:",
    len(future_dataset)
)

future_dataset = future_dataset.dropna(
    subset=["current_market_value", "future_market_value"]
)

print(
    "Rows after removing missing future values:",
    len(future_dataset)
)

print("\nMissing values:")
print(
    future_dataset[
        [
            "current_market_value",
            "future_market_value"
        ]
    ].isna().sum()
)

# ============================================================
# STEP 4.6: SAVE FUTURE-VALUE DATASET
# ============================================================

output_file = "player_future_value_dataset.csv"

future_dataset.to_csv(
    output_file,
    index=False
)

print("\nDataset saved successfully!")
print(output_file)

print("\nFINAL DATASET SHAPE:")
print(future_dataset.shape)

print("\nFINAL COLUMNS:")
print(future_dataset.columns.tolist())
print("\nFINAL DATASET EXAMPLE:")
print(player_season.columns.tolist())