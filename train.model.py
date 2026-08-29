import pandas as pd
import numpy as np


print("Loading dataset...")

data = pd.read_csv("player_future_value_dataset.csv")

print("Dataset loaded!")
print("Shape:", data.shape)
print()
print("Columns:")
print(data.columns.tolist())

# Features used by the model
features = [
    "age",
    "appearances",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "clean_sheets",
    "goal_contributions",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions_per_90",
    "height_in_cm",
    "international_caps",
    "international_goals",
    "current_market_value",
    "previous_market_value",
    "value_change_percent",
    "years_ahead"
]

# Calculate how the player's value changes in the future
data["future_value_ratio"] = (
    data["future_market_value"]
    / data["current_market_value"]
)

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=["future_value_ratio"])

target = "future_value_ratio"

X = data[features]
y = data[target]



print()
print("Features selected:")
print(features)

print()
print("Target:")
print(target)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)

print()
print("Splitting data by time...")

# Use older seasons for training
train_data = data[data["season"] <= 2022]

# Use newer seasons for testing
test_data = data[data["season"] > 2022]

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training seasons: 2012 to 2022")
print("Testing seasons: 2023 onwards")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

from sklearn.ensemble import RandomForestRegressor

print()
print("Creating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Model created!")

print()
print("Training model...")
print("This may take a while...")

model.fit(X_train, y_train)

print("Model training complete!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

print()
print("Testing model...")

predicted_ratios = model.predict(X_test)

# Convert predicted ratios back into actual market values
predicted_values = (
    predicted_ratios
    * test_data["current_market_value"].values
)

# Actual future market values
actual_values = test_data["future_market_value"].values

mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
r2 = r2_score(actual_values, predicted_values)

print()
print("MODEL PERFORMANCE")
print("-----------------")
print(f"MAE: €{mae:,.2f}")
print(f"RMSE: €{rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

print("\nTARGET CHECK")
print("-----------------")
print("Target minimum:", data["future_value_ratio"].min())
print("Target maximum:", data["future_value_ratio"].max())
print("Target mean:", data["future_value_ratio"].mean())
print("Target median:", data["future_value_ratio"].median())
print("Missing target values:", data["future_value_ratio"].isna().sum())

import joblib

print()
print("Saving model...")

joblib.dump(model, "player_value_model.pkl")

print("Model saved successfully!")

print()
print("TESTING PLAYER PREDICTION")

player_name = input("Enter player name: ")
years_ahead = int(input("How many years into the future? "))

player_data = data[
    data["name"].str.lower() == player_name.lower()
]
player_data = player_data.sort_values("season")

if player_data.empty:
    print("Player not found.")

else:
    player = player_data.iloc[-1]

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
print("PREDICTION")
print("-----------------")
print(f"Player: {player['name']}")
print(f"Current market value: €{player['current_market_value']:,.0f}")
print(f"Predicted value in {years_ahead} year(s): €{prediction:,.0f}")