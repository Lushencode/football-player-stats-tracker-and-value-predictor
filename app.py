import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=" LG Football Player Stats Tracker and Value Predictor",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("player_value_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("player_future_value_dataset.csv")


model = load_model()
data = load_data()

# ============================================================
# FEATURES
# ============================================================

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

# ============================================================
# TITLE
# ============================================================

st.title(" LG Football Player Stats Tracker and Value Predictor")

st.write(
    "Analyse player statistics and predict future market value."
)

# ============================================================
# PLAYER SEARCH
# ============================================================

player_names = sorted(
    data["name"].dropna().unique()
)

player_name = st.selectbox(
    "🔎 Search for a player",
    player_names
)

player_data = data[
    data["name"] == player_name
].copy()

player_data = player_data.sort_values("season")

# ============================================================
# PLAYER HEADER
# ============================================================

player = player_data.iloc[-1]

# ============================================================
# PLAYER HEADER
# ============================================================

header_col1, header_col2 = st.columns([1, 5])

with header_col1:

    image_url = player["image_url"]

    if pd.notna(image_url) and image_url != "":
        st.image(
            image_url,
            width=100
        )
    else:
        st.write("👤")

with header_col2:

    st.subheader(player_name)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Position**")
    st.write(player["position"])

with col2:
    st.write("**Age**")
    st.write(f"{round(player['age'])}")

with col3:
    st.write("**Club**")
    st.write(player["current_club_name"])

with col4:
    st.write("**Current Value**")

    if pd.notna(player["current_market_value"]):
        st.write(
            f"€{player['current_market_value']:,.0f}"
        )
    else:
        st.write("N/A")

# ============================================================
# MAIN MENU
# ============================================================

st.divider()

option = st.radio(
    "Choose an option:",
    [
        "Predict Future Market Value",
        "📊 View Past Season Stats"
    ],
    horizontal=True
)

# ============================================================
# PREDICTION
# ============================================================

if option == "Predict Future Market Value":

    st.subheader("Future Market Value Prediction")

    years_ahead = st.selectbox(
        "How many years into the future?",
        [1, 2, 3, 4, 5,6,7,8,9,10]
    )

    st.write("")

    predict_button = st.button(
        "🤖 Predict Future Market Value",
        type="primary",
        use_container_width=True
    )

    if predict_button:

        current_value = player["current_market_value"]

        # Check current market value
        if pd.isna(current_value) or current_value <= 0:

            st.error(
                "This player does not have a valid current market value."
            )

        else:

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

                "goal_contributions_per_90":
                    player["goal_contributions_per_90"],

                "height_in_cm": player["height_in_cm"],

                "international_caps":
                    player["international_caps"],

                "international_goals":
                    player["international_goals"],

                "current_market_value":
                    player["current_market_value"],

                "previous_market_value":
                    player["previous_market_value"],

                "value_change_percent":
                    player["value_change_percent"],

                "years_ahead":
                    years_ahead
            }])

            # Check missing model inputs
            if prediction_data[features].isna().any().any():

                missing = prediction_data[
                    features
                ].columns[
                    prediction_data[features]
                    .isna()
                    .any()
                ].tolist()

                st.error(
                    "Not enough player data is available for prediction. "
                   # + ", ".join(missing)
                )

            else:

                # ============================================
                # RANDOM FOREST
                # ============================================

                predicted_ratio = model.predict(
                    prediction_data[features]
                )[0]

                predicted_value = (
                    predicted_ratio *
                    current_value
                )

                # ============================================
                # VALUE CHANGE
                # ============================================

                percentage_change = (
                    (predicted_value - current_value)
                    / current_value
                ) * 100

                # ============================================
                # RESULTS
                # ============================================

                st.divider()

                st.subheader("📈 Prediction Result")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Current Value",
                        f"€{current_value:,.0f}"
                    )

                with col2:

                    st.metric(
                        f"Predicted Value ({years_ahead} Year"
                        f"{'s' if years_ahead != 1 else ''})",
                        f"€{predicted_value:,.0f}"
                    )

                with col3:

                    st.metric(
                        "Expected Change",
                        f"{percentage_change:+.2f}%"
                    )

                # ============================================
                # INTERPRETATION
                # ============================================

                if percentage_change > 5:

                    st.success(
                        f"📈 The model predicts that "
                        f"{player_name}'s value could increase "
                        f"by approximately "
                        f"{percentage_change:.2f}%."
                    )

                elif percentage_change < -5:

                    st.error(
                        f"📉 The model predicts that "
                        f"{player_name}'s value could decrease "
                        f"by approximately "
                        f"{abs(percentage_change):.2f}%."
                    )

                else:

                    st.info(
                        f"➡️ The model predicts that "
                        f"{player_name}'s value will remain "
                        f"relatively stable."
                    )

# ============================================================
# PAST SEASON STATISTICS
# ============================================================

elif option == "📊 View Past Season Stats":

    st.subheader("📊 Past Season Statistics")

    available_seasons = sorted(
        player_data["season"]
        .dropna()
        .unique(),
        reverse=True
    )

    selected_season = st.selectbox(
        "Select a season",
        available_seasons
    )

    season_data = player_data[
        player_data["season"] == selected_season
    ]

    if not season_data.empty:

        season_player = season_data.iloc[0]

        st.divider()

        st.subheader(
            f"{player_name} — {int(selected_season)} Season"
        )

        # ====================================================
        # BASIC INFORMATION
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Age",
                f"{round(season_player['age'])}"
            )

        with col2:
            st.metric(
                "Position",
                season_player["position"]
            )

        with col3:
            st.metric(
                "Club",
                season_player["current_club_name"]
            )

        with col4:

            if pd.notna(
                season_player["current_market_value"]
            ):

                st.metric(
                    "Market Value",
                    f"€{season_player['current_market_value']:,.0f}"
                )

            else:

                st.metric(
                    "Market Value",
                    "N/A"
                )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.subheader("Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Appearances",
                int(season_player["appearances"])
            )

        with col2:
            st.metric(
                "Minutes",
                f"{int(season_player['minutes_played']):,}"
            )

        with col3:
            st.metric(
                "Goals",
                int(season_player["goals"])
            )

        with col4:
            st.metric(
                "Assists",
                int(season_player["assists"])
            )

        # ====================================================
        # ADDITIONAL STATS
        # ====================================================

        st.subheader("Additional Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Goal Contributions",
                int(season_player["goal_contributions"])
            )

        with col2:
            st.metric(
                "Clean Sheets",
                int(season_player["clean_sheets"])
            )

        with col3:
            st.metric(
                "Yellow Cards",
                int(season_player["yellow_cards"])
            )

        with col4:
            st.metric(
                "Red Cards",
                int(season_player["red_cards"])
            )

        # ====================================================
        # PER 90
        # ====================================================

        st.subheader("Per-90 Statistics")

        col1, col2, col3 = st.columns(3)

        goals_90 = season_player["goals_per_90"]
        assists_90 = season_player["assists_per_90"]
        contributions_90 = (
            season_player["goal_contributions_per_90"]
        )

        with col1:

            st.metric(
                "Goals per 90",
                "N/A"
                if pd.isna(goals_90)
                else f"{goals_90:.2f}"
            )

        with col2:

            st.metric(
                "Assists per 90",
                "N/A"
                if pd.isna(assists_90)
                else f"{assists_90:.2f}"
            )

        with col3:

            st.metric(
                "Goal Contributions per 90",
                "N/A"
                if pd.isna(contributions_90)
                else f"{contributions_90:.2f}"
            )

        # ====================================================
        # INTERNATIONAL
        # ====================================================

        st.subheader("International Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "International Caps",
                int(season_player["international_caps"]) if pd.notna(season_player["international_caps"]) else 0
            )

        with col2:
            st.metric(
                "International Goals",
                int(season_player["international_goals"])  if pd.notna(season_player["international_goals"]) else 0
            
            )

        # ====================================================
        # VALUE HISTORY
        # ====================================================

        st.subheader("Market Value Information")

        col1, col2 = st.columns(2)

        previous_value = season_player[
            "previous_market_value"
        ]

        value_change = season_player[
            "value_change_percent"
        ]

        with col1:

            if pd.notna(previous_value):

                st.metric(
                    "Previous Market Value",
                    f"€{previous_value:,.0f}"
                )

            else:

                st.metric(
                    "Previous Market Value",
                    "N/A"
                )

        with col2:

            if pd.notna(value_change):

                st.metric(
                    "Value Change",
                    f"{value_change * 100:+.2f}%"
                )

            else:

                st.metric(
                    "Value Change",
                    "N/A"
                )

        # ====================================================
        # COMPLETE HISTORY
        # ====================================================

        st.subheader("📋 Player Season History")

        history_columns = [
            "season",
            "age",
            "current_club_name",
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "clean_sheets",
            "goal_contributions",
            "current_market_value"
        ]

        history = player_data[
            history_columns
        ].drop_duplicates(
            subset=["season"]
        ).sort_values(
            "season",
            ascending=False
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

