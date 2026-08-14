import os
import joblib

# ============================================================
# BASE MODEL DIRECTORY
# ============================================================

BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "model"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {

    "Breast Cancer": {
        "folder": "breast_cancer",
        "model": "best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "Chronic Kidney Disease": {
        "folder": "chronic_kidney",
        "model": "ckd_random_forest.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "Diabetes": {
        "folder": "diabetes",
        "model": "QD_diabetes.pkl",
        "scaler": "diabetes_scaler.pkl",
        "columns": "diabetes_columns.pkl"
    },

    "Heart Disease": {
        "folder": "heart_disease",
        "model": "KNN_heart.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "Liver Disease": {
        "folder": "liver_disease",
        "model": "liver_logistic_model.pkl",
        "scaler": "liver_scaler.pkl",
        "columns": "liver_columns.pkl"
    },

    "Lung Cancer": {
        "folder": "lung_cancer",
        "model": "lung_cancer_best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl"
    },

    "Stroke": {
        "folder": "stroke",
        "model": "stroke_model.pkl",
        "scaler": "stroke_scaler.pkl",
        "columns": "stroke_columns.pkl"
    }
}


# ============================================================
# TEST SYMPTOM ROUTER
# ============================================================

print("\n" + "=" * 60)
print("TESTING SYMPTOM ROUTER")
print("=" * 60)

router_path = os.path.join(
    BASE_DIR,
    "symptom_router.pkl"
)

try:

    symptom_router = joblib.load(router_path)

    print("✅ symptom_router.pkl loaded successfully")
    print("Router type:", type(symptom_router))

except Exception as e:

    print("❌ Failed to load symptom_router.pkl")
    print("Error:", e)

    symptom_router = None


# ============================================================
# TEST ALL DISEASE MODELS
# ============================================================

print("\n" + "=" * 60)
print("TESTING ALL DISEASE MODELS")
print("=" * 60)


successful_models = 0
failed_models = 0


for disease, config in MODEL_CONFIG.items():

    print("\n" + "-" * 60)
    print(f"Testing: {disease}")
    print("-" * 60)

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


    # --------------------------------------------------------
    # Check Model
    # --------------------------------------------------------

    try:

        model = joblib.load(model_path)

        print("✅ Model loaded:")
        print("   ", config["model"])

    except Exception as e:

        print("❌ Model loading failed")
        print("Error:", e)

        failed_models += 1
        continue


    # --------------------------------------------------------
    # Check Scaler
    # --------------------------------------------------------

    try:

        scaler = joblib.load(scaler_path)

        print("✅ Scaler loaded:")
        print("   ", config["scaler"])

    except Exception as e:

        print("❌ Scaler loading failed")
        print("Error:", e)

        failed_models += 1
        continue


    # --------------------------------------------------------
    # Check Columns
    # --------------------------------------------------------

    try:

        columns = joblib.load(columns_path)

        print("✅ Columns loaded:")
        print("   ", config["columns"])

        print("   Number of features:", len(columns))

    except Exception as e:

        print("❌ Columns loading failed")
        print("Error:", e)

        failed_models += 1
        continue


    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    print("\nModel Type:")
    print("   ", type(model).__name__)

    print("Scaler Type:")
    print("   ", type(scaler).__name__)

    print("Features:")
    print("   ", columns)


    successful_models += 1


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULT")
print("=" * 60)

print(
    f"✅ Successful disease models: "
    f"{successful_models}/7"
)

print(
    f"❌ Failed disease models: "
    f"{failed_models}/7"
)


if symptom_router is not None:

    print(
        "✅ Symptom Router: Loaded"
    )

else:

    print(
        "❌ Symptom Router: Failed"
    )


print("=" * 60)


# ============================================================
# SYSTEM STATUS
# ============================================================

if successful_models == 7 and symptom_router is not None:

    print("\n🎉 ALL MODELS ARE READY!")

    print(
        "You can now proceed to "
        "disease_tools.py and orchestrator.py."
    )

else:

    print(
        "\n⚠️ Some files/models have problems."
    )

    print(
        "Check the error messages above."
    )