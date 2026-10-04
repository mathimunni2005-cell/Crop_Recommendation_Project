from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

app = Flask(__name__)

# ==========================================================
# LOAD CROP DATASET
# ==========================================================

crop_data = pd.read_csv(
    "datasets/crop_recommendation.csv"
)

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = crop_data[features]
y = crop_data["label"]

# ==========================================================
# TRAIN RANDOM FOREST MODEL
# ==========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# ==========================================================
# SHAP EXPLAINER
# ==========================================================

explainer = shap.TreeExplainer(model)

# ==========================================================
# LOAD INVESTMENT DATASET
# ==========================================================

investment_data = pd.read_csv(
    "datasets/crop_investment.csv"
)

# ==========================================================
# LOAD MARKET PRICE DATASET
# ==========================================================

market_data = pd.read_csv(
    "datasets/market_prices.csv"
)

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        # ==================================================
        # FARMER INPUT
        # ==================================================

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

        # ==================================================
        # CREATE INPUT DATAFRAME
        # ==================================================

        input_df = pd.DataFrame(
            [[
                N,
                P,
                K,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=features
        )

        # ==================================================
        # CROP PREDICTION
        # ==================================================

        recommended_crop = model.predict(
            input_df
        )[0]

        # ==================================================
        # TOP 3 CROP SELECTION
        # ==================================================

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = model.classes_

        probability_data = list(
            zip(classes, probabilities)
        )

        probability_data.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_crops = []

        for crop, probability in probability_data[:3]:

            top_crops.append({
                "crop": crop,
                "probability": round(
                    probability * 100,
                    2
                )
            })

        # ==================================================
        # SHAP EXPLANATION
        # ==================================================

        shap_values = explainer.shap_values(
            input_df
        )

        predicted_class_index = list(
            model.classes_
        ).index(recommended_crop)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            crop_shap_values = np.array(
                shap_values[predicted_class_index][0]
            )

        else:

            shap_array = np.array(
                shap_values
            )

            if shap_array.ndim == 3:

                crop_shap_values = shap_array[
                    0,
                    :,
                    predicted_class_index
                ]

            elif shap_array.ndim == 2:

                crop_shap_values = shap_array[0]

            else:

                crop_shap_values = shap_array

        # ==================================================
        # CREATE SHAP FEATURE DATA
        # ==================================================

        shap_data = []

        for feature, value, shap_value in zip(
            features,
            input_df.iloc[0],
            crop_shap_values
        ):

            shap_data.append({
                "feature": feature,
                "value": round(
                    float(value),
                    2
                ),
                "shap": round(
                    float(shap_value),
                    4
                ),
                "abs_shap": abs(
                    float(shap_value)
                )
            })

        # Sort by importance
        shap_data.sort(
            key=lambda x: x["abs_shap"],
            reverse=True
        )

        # ==================================================
        # CREATE HUMAN READABLE EXPLANATION
        # ==================================================

        top_explanations = shap_data[:4]

        explanation_points = []

        for item in top_explanations:

            if item["shap"] > 0:

                explanation_points.append(
                    f'{item["feature"]} ({item["value"]}) '
                    f'supported the recommendation'
                )

            elif item["shap"] < 0:

                explanation_points.append(
                    f'{item["feature"]} ({item["value"]}) '
                    f'had a lower influence on the recommendation'
                )

            else:

                explanation_points.append(
                    f'{item["feature"]} ({item["value"]}) '
                    f'had very little influence'
                )

        # ==================================================
        # INVESTMENT DETAILS
        # ==================================================

        crop_info = investment_data[
            investment_data["crop"].str.lower()
            == recommended_crop.lower()
        ]

        if not crop_info.empty:

            crop_info = crop_info.iloc[0]

            cost_per_acre = float(
                crop_info[
                    "total_cost_per_acre"
                ]
            )

            yield_per_acre = float(
                crop_info[
                    "expected_yield_quintal_per_acre"
                ]
            )

            total_investment = (
                cost_per_acre
                * land_area
            )

            expected_production = (
                yield_per_acre
                * land_area
            )

        else:

            cost_per_acre = 0
            yield_per_acre = 0
            total_investment = 0
            expected_production = 0

        # ==================================================
        # MARKET PRICE
        # ==================================================

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

        # ==================================================
        # PROFIT CALCULATION
        # ==================================================

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

        # ==================================================
        # FINAL RESULT
        # ==================================================

        result = {

            "crop": recommended_crop,

            "confidence": round(
                probabilities[
                    list(classes).index(
                        recommended_crop
                    )
                ] * 100,
                2
            ),

            "top_crops": top_crops,

            "shap_data": shap_data,

            "explanation_points":
                explanation_points,

            "land_area":
                land_area,

            "investment":
                total_investment,

            "production":
                expected_production,

            "market_price":
                market_price,

            "revenue":
                expected_revenue,

            "profit":
                expected_profit,

            "margin":
                profit_margin
        }

    return render_template(
        "index.html",
        result=result
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    print("--------------------------------")
    print("CROP RECOMMENDATION SYSTEM")
    print("SHAP EXPLAINABLE AI")
    print("INVESTMENT & PROFIT ANALYSIS")
    print("--------------------------------")

    print(
        "Open browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("--------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )