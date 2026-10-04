import os

from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)


# =========================================================
# FILE PATHS
# =========================================================

WEIGHTS_PATH = "models/lstm_weights.npz"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# LOAD DATASETS
# =========================================================

investment_data = pd.read_csv(
    "datasets/crop_investment.csv"
)

market_data = pd.read_csv(
    "datasets/market_prices.csv"
)

recommendation_data = pd.read_csv(
    "datasets/crop_recommendation.csv"
)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

investment_data.columns = (
    investment_data.columns
    .astype(str)
    .str.strip()
)

market_data.columns = (
    market_data.columns
    .astype(str)
    .str.strip()
)

recommendation_data.columns = (
    recommendation_data.columns
    .astype(str)
    .str.strip()
)


# =========================================================
# FIND CROP COLUMN
# =========================================================

def find_crop_column(dataframe):

    possible_columns = [
        "crop",
        "Crop",
        "label",
        "Label",
        "CROP"
    ]

    for column in possible_columns:

        if column in dataframe.columns:
            return column

    return None


recommendation_crop_column = find_crop_column(
    recommendation_data
)

investment_crop_column = find_crop_column(
    investment_data
)

market_crop_column = None

for column in [
    "commodity",
    "Commodity",
    "crop",
    "Crop"
]:

    if column in market_data.columns:
        market_crop_column = column
        break


# =========================================================
# CHECK DATASET COLUMNS
# =========================================================

print("\n====================================")
print("DATASET INFORMATION")
print("====================================")

print(
    "Recommendation columns:",
    list(recommendation_data.columns)
)

print(
    "Investment columns:",
    list(investment_data.columns)
)

print(
    "Market columns:",
    list(market_data.columns)
)

print(
    "Recommendation crop column:",
    recommendation_crop_column
)

print(
    "Investment crop column:",
    investment_crop_column
)

print(
    "Market crop column:",
    market_crop_column
)

print("====================================\n")


# =========================================================
# LOAD LSTM WEIGHTS
# =========================================================

weights = np.load(WEIGHTS_PATH)

lstm_kernel = weights["lstm_kernel"]

lstm_recurrent_kernel = (
    weights["lstm_recurrent_kernel"]
)

lstm_bias = weights["lstm_bias"]

dense1_kernel = weights["dense1_kernel"]

dense1_bias = weights["dense1_bias"]

dense2_kernel = weights["dense2_kernel"]

dense2_bias = weights["dense2_bias"]


# =========================================================
# LOAD SCALER AND LABEL ENCODER
# =========================================================

scaler = joblib.load(
    SCALER_PATH
)

label_encoder = joblib.load(
    ENCODER_PATH
)


# =========================================================
# ACTIVATION FUNCTIONS
# =========================================================

def sigmoid(x):

    x = np.clip(
        x,
        -50,
        50
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


def relu(x):

    return np.maximum(
        0,
        x
    )


# =========================================================
# LSTM PREDICTION
# =========================================================

def lstm_predict(sequence):

    hidden_size = (
        lstm_recurrent_kernel.shape[0]
    )

    h = np.zeros(
        hidden_size,
        dtype=np.float32
    )

    c = np.zeros(
        hidden_size,
        dtype=np.float32
    )

    for timestep in sequence:

        z = (
            np.dot(
                timestep,
                lstm_kernel
            )
            +
            np.dot(
                h,
                lstm_recurrent_kernel
            )
            +
            lstm_bias
        )

        # Keras LSTM gate order:
        # input
        # forget
        # cell
        # output

        i = sigmoid(
            z[
                :hidden_size
            ]
        )

        f = sigmoid(
            z[
                hidden_size:
                hidden_size * 2
            ]
        )

        g = np.tanh(
            z[
                hidden_size * 2:
                hidden_size * 3
            ]
        )

        o = sigmoid(
            z[
                hidden_size * 3:
            ]
        )

        c = (
            f * c
            +
            i * g
        )

        h = (
            o
            *
            np.tanh(c)
        )


    # Dense layer 1

    dense1 = (
        np.dot(
            h,
            dense1_kernel
        )
        +
        dense1_bias
    )

    dense1 = relu(
        dense1
    )


    # Dense layer 2

    dense2 = (
        np.dot(
            dense1,
            dense2_kernel
        )
        +
        dense2_bias
    )


    # Final probabilities

    probabilities = softmax(
        dense2
    )

    return probabilities


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    result = None

    error = None


    # =====================================================
    # POST
    # =====================================================

    if request.method == "POST":

        try:

            # =============================================
            # GET INPUT VALUES
            # =============================================

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


            # =============================================
            # CREATE INPUT DATAFRAME
            # =============================================

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


            # =============================================
            # SCALE INPUT
            # =============================================

            input_scaled = scaler.transform(
                input_df
            )


            # =============================================
            # CREATE LSTM SEQUENCE
            # =============================================

            sequence = np.repeat(
                input_scaled[
                    :,
                    np.newaxis,
                    :
                ],
                SEQUENCE_LENGTH,
                axis=1
            )


            # =============================================
            # LSTM PREDICTION
            # =============================================

            probabilities = lstm_predict(
                sequence[0]
            )


            print(
                "\n===================================="
            )

            print(
                "ALL CROP PREDICTIONS"
            )

            print(
                "===================================="
            )


            # =============================================
            # ALL CROP PREDICTIONS
            # =============================================

            top_indices = np.argsort(
                probabilities
            )[::-1]


            top_crops = []


            for index in top_indices:

                index = int(index)


                crop_name = str(
                    label_encoder.inverse_transform(
                        [index]
                    )[0]
                )


                confidence = float(
                    probabilities[index] * 100
                )


                if not np.isfinite(
                    confidence
                ):

                    confidence = 0.0


                confidence = round(
                    confidence,
                    2
                )


                print(
                    crop_name,
                    "=>",
                    confidence,
                    "%"
                )


                top_crops.append({

                    "crop": crop_name,

                    "confidence": confidence

                })


            print(
                "====================================\n"
            )


            # =============================================
            # CHECK PREDICTION
            # =============================================

            if len(top_crops) == 0:

                raise ValueError(
                    "No crop predictions were generated."
                )


            # =============================================
            # RECOMMENDED CROP
            # =============================================

            recommended_crop = (
                top_crops[0]["crop"]
            )

            recommended_confidence = (
                top_crops[0]["confidence"]
            )


            # =============================================
            # FEATURE EXPLANATION
            # =============================================

            explanation_data = []

            explanation_points = []


            # IMPORTANT:
            # Use detected crop column instead of
            # directly using recommendation_data["crop"]

            if recommendation_crop_column is not None:

                crop_rows = (
                    recommendation_data[
                        recommendation_data[
                            recommendation_crop_column
                        ]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        recommended_crop
                        .strip()
                        .lower()
                    ]
                )


                if not crop_rows.empty:

                    reference = (
                        crop_rows[
                            features
                        ].mean()
                    )


                    for feature in features:

                        actual_value = float(
                            input_df.iloc[
                                0
                            ][feature]
                        )


                        reference_value = float(
                            reference[feature]
                        )


                        difference = abs(
                            actual_value
                            -
                            reference_value
                        )


                        explanation_data.append({

                            "feature":
                                feature,

                            "value":
                                round(
                                    actual_value,
                                    2
                                ),

                            "reference":
                                round(
                                    reference_value,
                                    2
                                ),

                            "difference":
                                round(
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


            # =============================================
            # INVESTMENT
            # =============================================

            investment_row = pd.DataFrame()


            if investment_crop_column is not None:

                investment_row = (
                    investment_data[
                        investment_data[
                            investment_crop_column
                        ]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        recommended_crop
                        .strip()
                        .lower()
                    ]
                )


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

                cost_per_acre = 0.0

                yield_per_acre = 0.0


            # =============================================
            # INVESTMENT CALCULATION
            # =============================================

            total_investment = (
                cost_per_acre
                *
                land_area
            )


            expected_production = (
                yield_per_acre
                *
                land_area
            )


            # =============================================
            # MARKET PRICE
            # =============================================

            market_rows = pd.DataFrame()


            if market_crop_column is not None:

                market_rows = (
                    market_data[
                        market_data[
                            market_crop_column
                        ]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        ==
                        recommended_crop
                        .strip()
                        .lower()
                    ]
                )


            if not market_rows.empty:

                market_price = float(
                    market_rows[
                        "modal_price"
                    ].mean()
                )

            else:

                market_price = 0.0


            # =============================================
            # PROFIT
            # =============================================

            expected_revenue = (
                expected_production
                *
                market_price
            )


            expected_profit = (
                expected_revenue
                -
                total_investment
            )


            # =============================================
            # PROFIT MARGIN
            # =============================================

            if total_investment > 0:

                profit_margin = (
                    expected_profit
                    /
                    total_investment
                ) * 100

            else:

                profit_margin = 0.0


            # =============================================
            # FINAL RESULT
            # =============================================

            result = {

                "crop":
                    recommended_crop,

                "confidence":
                    recommended_confidence,

                "top_crops":
                    top_crops,

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


            # =============================================
            # PRINT RESULT
            # =============================================

            print(
                "\n===================================="
            )

            print(
                "RESULT SENT TO HTML"
            )

            print(
                "===================================="
            )

            print(
                "Recommended Crop:",
                result["crop"]
            )

            print(
                "Recommended Confidence:",
                result["confidence"],
                "%"
            )

            print(
                "====================================\n"
            )


        except Exception as e:

            print(
                "\nPrediction Error:"
            )

            print(
                repr(e)
            )

            print(
                "====================================\n"
            )

            error = str(e)


    return render_template(
        "index.html",
        result=result,
        error=error
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )