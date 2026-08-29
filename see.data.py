import pandas as pd

# =========================
# LOAD PLAYER DATA
# =========================

players = pd.read_csv("data/archive/players.csv")

print("\nPLAYER DATA")
print("Shape:", players.shape)
print(players.head())


# =========================
# LOAD APPEARANCE DATA
# =========================

appearances = pd.read_csv("data/archive/appearances.csv")

print("\nAPPEARANCE DATA")
print("Shape:", appearances.shape)
print(appearances.head())


# =========================
# LOAD PLAYER VALUATIONS
# =========================

valuations = pd.read_csv("data/archive/player_valuations.csv")

print("\nVALUATION DATA")
print("Shape:", valuations.shape)
print(valuations.head())

print("\nPLAYER INFORMATION")
print(players.info())

print("\nAPPEARANCE INFORMATION")
print(appearances.info())

print("\nVALUATION INFORMATION")
print(valuations.info())

print("\nMISSING VALUES - PLAYERS")
print(players.isnull().sum())

print("\nMISSING VALUES - APPEARANCES")
print(appearances.isnull().sum())

print("\nMISSING VALUES - VALUATIONS")
print(valuations.isnull().sum())

games = pd.read_csv("data/archive/games.csv")

print("\nGAMES DATA")
print("Shape:", games.shape)
print(games.head())
print(games.columns.tolist())

print("\nSEASONS")
print(games["season"].value_counts().sort_index())

print("\nPLAYER POSITIONS")
print(players["position"].value_counts())

print("\nPLAYER SUB-POSITIONS")
print(players["sub_position"].value_counts().head(20))

print("\nMARKET VALUE STATISTICS")

print(players["market_value_in_eur"].describe())

print("\nHIGHEST MARKET VALUES")

print(
    players[
        ["name", "market_value_in_eur", "highest_market_value_in_eur"]
    ]
    .sort_values("market_value_in_eur", ascending=False)
    .head(20)
)
games["season"].value_counts().sort_index()
players["position"].value_counts()
players[
    ["name", "market_value_in_eur", "highest_market_value_in_eur"]
].sort_values(
    "market_value_in_eur",
    ascending=False
).head(20)