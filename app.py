from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==========================================================
# LOAD TRAINED LSTM MODEL
# ==========================================================

model = load_model(
    "models/crop_lstm.keras"
)

# ==========================================================
# LOAD SCALER
# ==========================================================

scaler = joblib.load(
    "models/scaler.pkl"
)

# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# ==========================================================
# FEATURES
# ==========================================================

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

SEQUENCE_LENGTH = 3

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
        # SCALE INPUT
        # ==================================================

        input_scaled = scaler.transform(
            input_df
        )

        # ==================================================
        # CREATE LSTM SEQUENCE
        # ==================================================
        #
        # The trained model expects:
        # (samples, 3 time steps, 7 features)
        #
        # For a single farmer input, we repeat the
        # current conditions for the 3 required steps.
        #

        sequence = np.repeat(
            input_scaled[:, np.newaxis, :],
            SEQUENCE_LENGTH,
            axis=1
        )

        # ==================================================
        # LSTM PREDICTION
        # ==================================================

        probabilities = model.predict(
            sequence,
            verbose=0
        )[0]

        # ==================================================
        # GET PREDICTED CROP
        # ==================================================

        predicted_index = np.argmax(
            probabilities
        )

        recommended_crop = (
            label_encoder.inverse_transform(
                [predicted_index]
            )[0]
        )

        # ==================================================
        # TOP 3 CROPS
        # ==================================================

        probability_data = []

        for index, probability in enumerate(
            probabilities
        ):

            crop_name = (
                label_encoder.inverse_transform(
                    [index]
                )[0]
            )

            probability_data.append(
                (
                    crop_name,
                    float(probability)
                )
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
        # LSTM FEATURE EXPLANATION
        # ==================================================
        #
        # This is a simple human-readable explanation.
        # Tree SHAP is NOT used because the model is LSTM.
        #

        feature_values = input_df.iloc[0].to_dict()

        # Crop-specific approximate comparison
        crop_rows = crop_data = pd.read_csv(
            "datasets/crop_recommendation.csv"
        )

        crop_reference = crop_rows[
            crop_rows["label"].str.lower()
            == recommended_crop.lower()
        ]

        explanation_data = []

        if not crop_reference.empty:

            reference_mean = (
                crop_reference[features]
                .mean()
            )

            for feature in features:

                user_value = float(
                    feature_values[feature]
                )

                reference_value = float(
                    reference_mean[feature]
                )

                difference = abs(
                    user_value
                    - reference_value
                )

                explanation_data.append({

                    "feature": feature,

                    "value": round(
                        user_value,
                        2
                    ),

                    "reference": round(
                        reference_value,
                        2
                    ),

                    "difference": round(
                        difference,
                        2
                    )

                })

        else:

            for feature in features:

                explanation_data.append({

                    "feature": feature,

                    "value": round(
                        float(
                            feature_values[feature]
                        ),
                        2
                    ),

                    "reference": 0,

                    "difference": 0

                })

        # ==================================================
        # SORT FEATURES
        # ==================================================

        explanation_data.sort(
            key=lambda x: x["difference"]
        )

        # ==================================================
        # HUMAN READABLE EXPLANATION
        # ==================================================

        explanation_points = []

        for item in explanation_data[:4]:

            explanation_points.append(

                f'{item["feature"]} value '
                f'{item["value"]} is close to the '
                f'recommended {recommended_crop} '
                f'conditions.'

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
        # REVENUE
        # ==================================================

        expected_revenue = (
            expected_production
            * market_price
        )

        # ==================================================
        # PROFIT
        # ==================================================

        expected_profit = (
            expected_revenue
            - total_investment
        )

        # ==================================================
        # PROFIT MARGIN
        # ==================================================

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

            "crop":
                recommended_crop,

            "confidence":
                round(
                    probabilities[
                        predicted_index
                    ] * 100,
                    2
                ),

            "top_crops":
                top_crops,

            "shap_data":
                explanation_data,

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

    print(
        "LSTM CROP RECOMMENDATION SYSTEM"
    )

    print(
        "INVESTMENT & PROFIT ANALYSIS"
    )

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