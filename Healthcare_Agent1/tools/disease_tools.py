import os
import joblib
import pandas as pd


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model"
    )
)


# ============================================================
# DISEASE MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "breast_cancer": {
        "folder": "breast_cancer",
        "model": "best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "chronic_kidney": {
        "folder": "chronic_kidney",
        "model": "ckd_random_forest.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "diabetes": {
        "folder": "diabetes",
        "model": "QD_diabetes.pkl",
        "scaler": "diabetes_scaler.pkl",
        "columns": "diabetes_columns.pkl"
    },

    "heart_disease": {
        "folder": "heart_disease",
        "model": "KNN_heart.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "liver_disease": {
        "folder": "liver_disease",
        "model": "liver_logistic_model.pkl",
        "scaler": "liver_scaler.pkl",
        "columns": "liver_columns.pkl"
    },

    "lung_cancer": {
        "folder": "lung_cancer",
        "model": "lung_cancer_best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "stroke": {
        "folder": "stroke",
        "model": "stroke_model.pkl",
        "scaler": "stroke_scaler.pkl",
        "columns": "stroke_columns.pkl"
    }
}


# ============================================================
# LOAD DISEASE FILES
# ============================================================

def load_disease_files(disease):

    if disease not in MODEL_CONFIG:

        raise ValueError(
            f"Unknown disease: {disease}"
        )

    config = MODEL_CONFIG[disease]

    folder = os.path.join(
        BASE_DIR,
        config["folder"]
    )

    model_path = os.path.join(
        folder,
        config["model"]
    )

    scaler_path = os.path.join(
        folder,
        config["scaler"]
    )

    columns_path = os.path.join(
        folder,
        config["columns"]
    )

    model = joblib.load(model_path)

    scaler = joblib.load(scaler_path)

    columns = joblib.load(columns_path)

    return model, scaler, columns


# ============================================================
# GENERIC PREDICTION FUNCTION
# ============================================================

def predict_disease(
    disease,
    input_data
):

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, scaler, columns = load_disease_files(
        disease
    )


    # --------------------------------------------------------
    # Convert input to DataFrame
    # --------------------------------------------------------

    if isinstance(input_data, dict):

        input_df = pd.DataFrame(
            [input_data]
        )

    elif isinstance(input_data, pd.DataFrame):

        input_df = input_data.copy()

    else:

        raise TypeError(
            "input_data must be dictionary or DataFrame"
        )


    # --------------------------------------------------------
    # Make sure all required columns exist
    # --------------------------------------------------------

    for column in columns:

        if column not in input_df.columns:

            input_df[column] = 0


    # --------------------------------------------------------
    # Remove unwanted columns
    # --------------------------------------------------------

    input_df = input_df[
        columns
    ]


    # ========================================================
    # LUNG CANCER
    # ========================================================
    #
    # Lung cancer model is already a Pipeline.
    # Therefore, DO NOT manually apply scaler.
    # ========================================================

    if disease == "lung_cancer":

        prediction = model.predict(
            input_df
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_df
            )[0]

        else:

            probabilities = None


    # ========================================================
    # OTHER MODELS
    # ========================================================

    else:

        input_scaled = scaler.transform(
            input_df
        )

        prediction = model.predict(
            input_scaled
        )[0]

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

        else:

            probabilities = None


    # ========================================================
    # CREATE RESULT
    # ========================================================

    result = {

        "disease": disease,

        "prediction": prediction,

        "probabilities": probabilities,

        "model": type(model).__name__,

        "features": columns
    }


    return result


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_NAMES = {

    "breast_cancer":
        "Breast Cancer",

    "chronic_kidney":
        "Chronic Kidney Disease",

    "diabetes":
        "Diabetes",

    "heart_disease":
        "Heart Disease",

    "liver_disease":
        "Liver Disease",

    "lung_cancer":
        "Lung Cancer",

    "stroke":
        "Stroke"
}


# ============================================================
# GET DISEASE NAME
# ============================================================

def get_disease_name(
    disease
):

    return DISEASE_NAMES.get(
        disease,
        disease
    )