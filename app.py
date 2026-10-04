import os

# ============================================================
# RENDER / TENSORFLOW RESOURCE SETTINGS
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/crop_lstm.keras"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# LOAD DATASETS
# ============================================================

investment_data = pd.read_csv(
    "datasets/crop_investment.csv"
)

market_data = pd.read_csv(
    "datasets/market_prices.csv"
)

recommendation_data = pd.read_csv(
    "datasets/crop_recommendation.csv"
)


# ============================================================
# LAZY LOAD AI MODEL
# ============================================================

model = None
scaler = None
label_encoder = None


def load_ai_model():

    global model
    global scaler
    global label_encoder

    if model is None:

        print("Loading TensorFlow LSTM model...")

        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        from tensorflow.keras.models import load_model

        model = load_model(
            MODEL_PATH,
            compile=False
        )

        scaler = joblib.load(
            SCALER_PATH
        )

        label_encoder = joblib.load(
            ENCODER_PATH
        )

        print("LSTM model loaded successfully.")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            load_ai_model()


            # ------------------------------------------------
            # GET USER INPUT
            # ------------------------------------------------

            N = float(
                request.form["N"]
            )

            P = float(
                request.form["P"]
            )

            K = float(
                request.form["K"]
            )

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


            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

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


            # ------------------------------------------------
            # SCALE INPUT
            # ------------------------------------------------

            input_scaled = scaler.transform(
                input_df
            )


            # ------------------------------------------------
            # CREATE LSTM SEQUENCE
            # ------------------------------------------------

            sequence = np.repeat(
                input_scaled[:, np.newaxis, :],
                SEQUENCE_LENGTH,
                axis=1
            )


            # ------------------------------------------------
            # LSTM PREDICTION
            # ------------------------------------------------

            probabilities = model.predict(
                sequence,
                verbose=0
            )[0]


            # ------------------------------------------------
            # TOP 3 CROPS
            # ------------------------------------------------

            top_indices = np.argsort(
                probabilities
            )[::-1][:3]

            top_crops = []

            for index in top_indices:

                crop_name = label_encoder.inverse_transform(
                    [index]
                )[0]

                confidence = float(
                    probabilities[index] * 100
                )

                top_crops.append({
                    "crop": crop_name,
                    "confidence": round(
                        confidence,
                        2
                    )
                })


            # ------------------------------------------------
            # RECOMMENDED CROP
            # ------------------------------------------------

            recommended_crop = top_crops[0]["crop"]

            recommended_confidence = (
                top_crops[0]["confidence"]
            )


            # =================================================
            # CROP CONDITION ANALYSIS
            # =================================================

            crop_rows = recommendation_data[
                recommendation_data["crop"].str.lower()
                == recommended_crop.lower()
            ]

            explanation_data = []
            explanation_points = []


            if not crop_rows.empty:

                reference = crop_rows[
                    features
                ].mean()


                for feature in features:

                    actual_value = float(
                        input_df.iloc[0][feature]
                    )

                    reference_value = float(
                        reference[feature]
                    )

                    difference = abs(
                        actual_value -
                        reference_value
                    )


                    explanation_data.append({

                        "feature": feature,

                        "value": round(
                            actual_value,
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


                    explanation_points.append(
                        f"{feature} = "
                        f"{actual_value:.2f}, "
                        f"reference = "
                        f"{reference_value:.2f}"
                    )


            # =================================================
            # INVESTMENT ANALYSIS
            # =================================================

            investment_row = investment_data[
                investment_data["crop"].str.lower()
                == recommended_crop.lower()
            ]


            if not investment_row.empty:

                investment_row = (
                    investment_row.iloc[0]
                )

                cost_per_acre = float(
                    investment_row[
                        "total_cost_per_acre"
                    ]
                )

                yield_per_acre = float(
                    investment_row[
                        "expected_yield_quintal_per_acre"
                    ]
                )

            else:

                cost_per_acre = 0

                yield_per_acre = 0


            # =================================================
            # TOTAL INVESTMENT
            # =================================================

            total_investment = (
                cost_per_acre *
                land_area
            )


            # =================================================
            # EXPECTED PRODUCTION
            # =================================================

            expected_production = (
                yield_per_acre *
                land_area
            )


            # =================================================
            # MARKET PRICE
            # =================================================

            market_rows = market_data[
                market_data["commodity"].str.lower()
                == recommended_crop.lower()
            ]


            if not market_rows.empty:

                market_price = float(
                    market_rows[
                        "modal_price"
                    ].mean()
                )

            else:

                market_price = 0


            # =================================================
            # EXPECTED REVENUE
            # =================================================

            expected_revenue = (
                expected_production *
                market_price
            )


            # =================================================
            # EXPECTED PROFIT
            # =================================================

            expected_profit = (
                expected_revenue -
                total_investment
            )


            # =================================================
            # PROFIT MARGIN
            # =================================================

            if total_investment > 0:

                profit_margin = (
                    expected_profit /
                    total_investment
                ) * 100

            else:

                profit_margin = 0


            # =================================================
            # FINAL RESULT
            # =================================================

            result = {

                "crop":
                    recommended_crop,

                "confidence":
                    recommended_confidence,

                "top_crops":
                    top_crops,

                # Kept as shap_data because
                # your existing HTML uses this name.
                # It is NOT SHAP data.
                "shap_data":
                    explanation_data,

                "explanation_points":
                    explanation_points,

                "land_area":
                    round(
                        land_area,
                        2
                    ),

                "investment":
                    round(
                        total_investment,
                        2
                    ),

                "production":
                    round(
                        expected_production,
                        2
                    ),

                "market_price":
                    round(
                        market_price,
                        2
                    ),

                "revenue":
                    round(
                        expected_revenue,
                        2
                    ),

                "profit":
                    round(
                        expected_profit,
                        2
                    ),

                "margin":
                    round(
                        profit_margin,
                        2
                    )
            }


        except Exception as e:

            print(
                "Prediction error:",
                repr(e)
            )

            error = str(e)


    # =========================================================
    # SEND RESULT TO HTML
    # =========================================================

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )