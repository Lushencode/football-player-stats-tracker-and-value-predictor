import pandas as pd
import joblib

print("Loading dataset...")
dataset = pd.read_csv("player_future_value_dataset.csv")

print("Loading trained model...")
model = joblib.load("player_value_model.pkl")

print("Everything loaded successfully!")


player_name = input("Enter player name: ")
years_ahead = int(input("How many years into the future? "))

player_data = dataset[
    dataset["name"].str.lower() == player_name.lower()
]


if player_data.empty:
    print("Player not found.")
    exit()
else:
    print(f"Found {len(player_data)} records for {player_name}")

    player = player_data.sort_values("season").iloc[-1]
    prediction_data = pd.DataFrame([{
        "age": player["age"],
        "appearances": player["appearances"],
        "minutes_played": player["minutes_played"],
        "goals": player["goals"],
        "assists": player["assists"],
        "yellow_cards": player["yellow_cards"],
        "red_cards": player["red_cards"],
        "clean_sheets": player["clean_sheets"],
        "goal_contributions": player["goal_contributions"],
        "goals_per_90": player["goals_per_90"],
        "assists_per_90": player["assists_per_90"],
        "goal_contributions_per_90": player["goal_contributions_per_90"],
        "height_in_cm": player["height_in_cm"],
        "international_caps": player["international_caps"],
        "international_goals": player["international_goals"],
        "current_market_value": player["current_market_value"],
        "previous_market_value": player["previous_market_value"],
        "value_change_percent": player["value_change_percent"],
        "years_ahead": years_ahead
    }])

predicted_ratio = model.predict(prediction_data)[0]

prediction = (
    predicted_ratio
    * player["current_market_value"]
)
print()
print("=" * 40)
print("       ⚽ PLAYER VALUE PREDICTOR")
print("=" * 40)

print()
print(f"Player found: {player['name']}")

if "current_club_name" in player.index:
    print(f"Current club: {player['current_club_name']}")

print()
print(f"Current age: {round(float(player['age']))}")
print(f"Current value: €{int(player['current_market_value']):,}")

print()
print(f"Prediction: {years_ahead} years into the future")

print()
print("=" * 40)
print("             PREDICTION")
print("=" * 40)

print()
print(f"Player: {player['name']}")
print(f"Current age: {round(float(player['age']))}")
print(f"Future age: {round(float(player['age'])) + years_ahead}")

print()
print(f"Current value: €{int(player['current_market_value']):,}")
print(f"Predicted value: €{int(prediction):,}")

print()
print("=" * 40)