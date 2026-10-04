from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ==============================
# LOAD CROP DATASET
# ==============================

crop_data = pd.read_csv(
    "datasets/crop_recommendation.csv"
)

# Input features
X = crop_data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target
y = crop_data["label"]

# Train crop recommendation model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==============================
# LOAD INVESTMENT DATASET
# ==============================

investment_data = pd.read_csv(
    "datasets/crop_investment.csv"
)


# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # Farmer input
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])

        temperature = float(
            request.form["temperature"]
        )

        humidity = float(
            request.form["humidity"]
        )

        ph = float(
            request.form["ph"]
        )

        rainfall = float(
            request.form["rainfall"]
        )

        land_area = float(
            request.form["land_area"]
        )

        # ==============================
        # CROP RECOMMENDATION
        # ==============================

        input_data = [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]]

        recommended_crop = model.predict(
            input_data
        )[0]

        # ==============================
        # INVESTMENT DETAILS
        # ==============================

        crop_info = investment_data[
            investment_data["crop"].str.lower()
            == recommended_crop.lower()
        ]

        if not crop_info.empty:

            crop_info = crop_info.iloc[0]

            cost_per_acre = float(
                crop_info["total_cost_per_acre"]
            )

            yield_per_acre = float(
                crop_info[
                    "expected_yield_quintal_per_acre"
                ]
            )

            total_investment = (
                cost_per_acre * land_area
            )

            expected_production = (
                yield_per_acre * land_area
            )

            # ==============================
            # PROFIT CALCULATION
            # ==============================

            # Demo market price
            market_data = pd.read_csv(
                "datasets/market_prices.csv"
            )

            market_info = market_data[
                market_data["commodity"].str.lower()
                == recommended_crop.lower()
            ]

            if not market_info.empty:

                market_price = float(
                    market_info["modal_price"].mean()
                )

            else:
                market_price = 0

            expected_revenue = (
                expected_production
                * market_price
            )

            expected_profit = (
                expected_revenue
                - total_investment
            )

            if total_investment > 0:

                profit_margin = (
                    expected_profit
                    / total_investment
                ) * 100

            else:

                profit_margin = 0

            # ==============================
            # FINAL RESULT
            # ==============================

            result = {
                "crop": recommended_crop,
                "land_area": land_area,
                "investment": total_investment,
                "production": expected_production,
                "market_price": market_price,
                "revenue": expected_revenue,
                "profit": expected_profit,
                "margin": profit_margin
            }

    return render_template(
        "index.html",
        result=result
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":

    print("--------------------------------")
    print("CROP RECOMMENDATION +")
    print("INVESTMENT & PROFIT ANALYSIS")
    print("--------------------------------")
    print("Open browser:")
    print("http://127.0.0.1:5000")
    print("--------------------------------")

    app.run(host="0.0.0.0", port=5000, debug=True)