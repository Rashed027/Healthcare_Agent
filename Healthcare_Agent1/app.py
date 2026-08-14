import streamlit as st
import pandas as pd
import joblib
import os
import re
import numpy as np

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Healthcare Agent 🩺",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ============================================================
# HEADER
# ============================================================
st.title("🏥 Healthcare Agent 🩺")
st.subheader("🤖 AI-Powered Multi-Disease Prediction System")

st.write(
    "This application performs educational disease screening using "
    "symptoms and disease-specific machine-learning models. "
    "It is not a medical diagnosis system."
)

# ============================================================
# DISEASE CONFIG
# ============================================================
DISEASES = {
    "breast_cancer": {
        "name": "Breast Cancer",
        "model": "best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl",
    },
    "chronic_kidney": {
        "name": "Chronic Kidney Disease",
        "model": "ckd_random_forest.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl",
    },
    "diabetes": {
        "name": "Diabetes",
        "model": "QD_diabetes.pkl",
        "scaler": "diabetes_scaler.pkl",
        "columns": "diabetes_columns.pkl",
    },
    "heart_disease": {
        "name": "Heart Disease",
        "model": "KNN_heart.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl",
    },
    "liver_disease": {
        "name": "Liver Disease",
        "model": "liver_logistic_model.pkl",
        "scaler": "liver_scaler.pkl",
        "columns": "liver_columns.pkl",
    },
    "lung_cancer": {
        "name": "Lung Cancer",
        "model": "lung_cancer_best_model.pkl",
        "scaler": "scaler.pkl",
        "columns": "columns.pkl",
    },
    "stroke": {
        "name": "Stroke",
        "model": "stroke_model.pkl",
        "scaler": "stroke_scaler.pkl",
        "columns": "stroke_columns.pkl",
    },
}

# ============================================================
# DISEASE CARDS
# ============================================================
st.markdown("### 🧬 Supported Diseases")
DISEASE_ICONS = {
    "breast_cancer": "🎗️",
    "chronic_kidney": "🫘",
    "diabetes": "🩸",
    "heart_disease": "❤️",
    "liver_disease": "🫀",
    "lung_cancer": "🫁",
    "stroke": "🧠",
}
cards = list(DISEASES.items())
for start in range(0, len(cards), 3):
    cols = st.columns(3)
    for i, (_, cfg) in enumerate(cards[start:start + 3]):
        with cols[i]:
            st.info(
                f"{DISEASE_ICONS.get(_, '🩺')} **{cfg['name']}**")

st.divider()

# ============================================================
# REPORT GUIDE
# ============================================================
REPORTS = {
    "breast_cancer": [
        "Clinical breast examination",
        "Mammography or breast ultrasound when clinically appropriate",
        "Pathology/biopsy report when recommended by a clinician",
        "Tumor measurement/pathology features required by the model",
    ],
    "chronic_kidney": [
        "Blood pressure measurement",
        "Kidney Function Test (KFT/RFT)",
        "Serum creatinine",
        "Blood urea",
        "Serum electrolytes",
        "Urine routine examination",
        "CBC",
    ],
    "diabetes": [
        "Fasting or random blood glucose",
        "HbA1c when clinically appropriate",
        "Pregnancy history where applicable",
        "Height and weight for BMI",
        "Insulin laboratory value if required by the trained model",
    ],
    "heart_disease": [
        "Blood pressure measurement",
        "Lipid profile",
        "ECG",
        "Heart-rate measurement",
        "Clinical cardiac assessment",
        "Exercise/stress testing when clinically indicated",
    ],
    "liver_disease": [
        "Liver Function Test (LFT)",
        "Total bilirubin",
        "Direct bilirubin",
        "ALT",
        "AST",
        "Alkaline phosphatase",
        "Total protein",
        "Albumin",
        "Albumin/Globulin ratio",
    ],
    "lung_cancer": [
        "Clinical chest evaluation",
        "Chest X-ray when clinically appropriate",
        "CT scan when clinically indicated",
        "Smoking history",
        "Relevant clinical evaluation",
    ],
    "stroke": [
        "Blood pressure measurement",
        "Blood glucose assessment",
        "Height and weight/BMI",
        "Cardiovascular risk assessment",
        "Smoking history",
        "Neurological evaluation when clinically indicated",
    ],
}

# ============================================================
# USER-FRIENDLY COLUMN INFORMATION
# ============================================================
COLUMN_INFO = {
    # Diabetes
    "Pregnancies": ("Number of Pregnancies", "Patient history"),
    "Glucose": ("Blood Glucose", "Blood glucose test"),
    "BloodPressure": ("Blood Pressure", "Blood pressure measurement"),
    "SkinThickness": ("Skin Thickness", "Clinical/anthropometric measurement"),
    "Insulin": ("Insulin Level", "Blood laboratory report"),
    "BMI": ("Body Mass Index (BMI)", "Height + weight"),
    "DiabetesPedigreeFunction": ("Diabetes Family-History Index", "Dataset/model feature"),
    "Age": ("Age", "Patient information"),

    # CKD
    "Bp": ("Blood Pressure", "Blood pressure measurement"),
    "Sg": ("Urine Specific Gravity", "Urine examination"),
    "Al": ("Urine Albumin", "Urine examination"),
    "Su": ("Urine Sugar", "Urine examination"),
    "Rbc": ("Urine Red Blood Cells", "Urine examination"),
    "Bu": ("Blood Urea", "Kidney function test"),
    "Sc": ("Serum Creatinine", "Kidney function test"),
    "Sod": ("Serum Sodium", "Electrolyte test"),
    "Pot": ("Serum Potassium", "Electrolyte test"),
    "Hemo": ("Hemoglobin", "CBC"),
    "Wbcc": ("White Blood Cell Count", "CBC"),
    "Rbcc": ("Red Blood Cell Count", "CBC"),
    "Htn": ("Hypertension", "Blood pressure/medical history"),

    # Heart
    "age": ("Age", "Patient information"),
    "sex": ("Sex", "Patient information"),
    "cp": ("Chest Pain Type", "Clinical cardiac assessment"),
    "trestbps": ("Resting Blood Pressure", "Blood pressure measurement"),
    "chol": ("Serum Cholesterol", "Lipid profile"),
    "fbs": ("Fasting Blood Sugar > 120", "Blood glucose test"),
    "restecg": ("Resting ECG Result", "ECG"),
    "thalach": ("Maximum Heart Rate", "Exercise/stress assessment"),
    "exang": ("Exercise-Induced Angina", "Clinical cardiac assessment"),
    "oldpeak": ("ST Depression", "ECG/stress test"),
    "slope": ("Peak Exercise ST Slope", "ECG/stress test"),
    "ca": ("Number of Major Vessels", "Cardiac imaging/test"),
    "thal": ("Thalium Stress-Test Result", "Cardiac stress test"),

    # Liver
    "Gender": ("Gender", "Patient information"),
    "Total_Bilirubin": ("Total Bilirubin", "Liver Function Test"),
    "Direct_Bilirubin": ("Direct Bilirubin", "Liver Function Test"),
    "Alkaline_Phosphotase": ("Alkaline Phosphatase", "Liver Function Test"),
    "Alamine_Aminotransferase": ("ALT", "Liver Function Test"),
    "Aspartate_Aminotransferase": ("AST", "Liver Function Test"),
    "Total_Protiens": ("Total Protein", "Liver Function Test"),
    "Albumin": ("Albumin", "Liver Function Test"),
    "Albumin_and_Globulin_Ratio": ("Albumin/Globulin Ratio", "Liver Function Test"),

    # Lung
    "GENDER": ("Gender", "Patient information"),
    "AGE": ("Age", "Patient information"),
    "SMOKING": ("Smoking", "Smoking history"),
    "YELLOW_FINGERS": ("Yellow Fingers", "Clinical history"),
    "ANXIETY": ("Anxiety", "Clinical history"),
    "PEER_PRESSURE": ("Peer Pressure", "History"),
    "CHRONIC_DISEASE": ("Chronic Disease", "Medical history"),
    "FATIGUE": ("Fatigue", "Clinical symptom"),
    "ALLERGY": ("Allergy", "Medical history"),
    "WHEEZING": ("Wheezing", "Clinical symptom"),
    "ALCOHOL_CONSUMING": ("Alcohol Use", "Medical history"),
    "COUGHING": ("Coughing", "Clinical symptom"),
    "SHORTNESS_OF_BREATH": ("Shortness of Breath", "Clinical symptom"),
    "SWALLOWING_DIFFICULTY": ("Swallowing Difficulty", "Clinical symptom"),
    "CHEST_PAIN": ("Chest Pain", "Clinical symptom"),

    # Stroke
    "gender": ("Gender", "Patient information"),
    "hypertension": ("Hypertension", "Medical history"),
    "heart_disease": ("Heart Disease History", "Medical history"),
    "avg_glucose_level": ("Average Glucose Level", "Blood glucose test"),
    "bmi": ("Body Mass Index (BMI)", "Height + weight"),
    "ever_married_Yes": ("Ever Married", "Patient history"),
    "work_type_Never_worked": ("Never Worked", "Patient history"),
    "work_type_Private": ("Private Work", "Patient history"),
    "work_type_Self-employed": ("Self-Employed", "Patient history"),
    "work_type_children": ("Children", "Patient history"),
    "Residence_type_Urban": ("Urban Residence", "Patient history"),
    "smoking_status_formerly smoked": ("Former Smoker", "Smoking history"),
    "smoking_status_never smoked": ("Never Smoked", "Smoking history"),
    "smoking_status_smokes": ("Currently Smokes", "Smoking history"),
}

# ============================================================
# POSITIVE RESULT SUGGESTIONS
# ============================================================
SUGGESTIONS = {
    "heart_disease": [
        "Consult a qualified doctor/cardiologist for clinical evaluation.",
        "Review blood pressure and lipid-profile results.",
        "Have the ECG findings reviewed by a healthcare professional.",
        "Further cardiac evaluation may be recommended based on symptoms and clinical findings.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "diabetes": [
        "Consult a qualified healthcare professional.",
        "Review fasting/random glucose and HbA1c results.",
        "Discuss diet, activity and other risk factors with a healthcare professional.",
        "Further testing may be recommended depending on the clinical situation.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "chronic_kidney": [
        "Consult a qualified healthcare professional.",
        "Review serum creatinine, urea and urine-test results.",
        "Review blood pressure and relevant electrolyte results.",
        "Further kidney evaluation may be recommended.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "liver_disease": [
        "Consult a qualified healthcare professional.",
        "Review the Liver Function Test results.",
        "Review bilirubin, ALT, AST, albumin and related results.",
        "Further liver evaluation may be recommended.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "lung_cancer": [
        "Consult a qualified healthcare professional.",
        "Review relevant chest-imaging and clinical findings.",
        "Discuss smoking history and other risk factors with a clinician.",
        "Further evaluation may be recommended based on clinical findings.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "stroke": [
        "Seek prompt medical evaluation, especially if concerning neurological symptoms are present.",
        "Review blood pressure and glucose status.",
        "Discuss cardiovascular and other stroke risk factors with a healthcare professional.",
        "Further neurological/cardiovascular evaluation may be recommended.",
        "Do not start or stop medication based only on this model prediction.",
    ],
    "breast_cancer": [
        "Consult a qualified healthcare professional.",
        "Review the clinical breast examination and relevant imaging.",
        "Mammography/ultrasound or pathology evaluation may be recommended when clinically appropriate.",
        "Further evaluation should be based on professional clinical assessment.",
        "Do not start or stop medication based only on this model prediction.",
    ],
}

# ============================================================
# SAFE MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_files(disease):
    cfg = DISEASES[disease]
    folder = os.path.join(MODEL_DIR, disease)

    paths = {
        "model": os.path.join(folder, cfg["model"]),
        "scaler": os.path.join(folder, cfg["scaler"]),
        "columns": os.path.join(folder, cfg["columns"]),
    }

    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) for {cfg['name']}: {', '.join(missing)}"
        )

    return (
        joblib.load(paths["model"]),
        joblib.load(paths["scaler"]),
        joblib.load(paths["columns"]),
    )

# ============================================================
# OPTIONAL SYMPTOM ROUTER
# ============================================================
@st.cache_resource
def load_router():
    path = os.path.join(MODEL_DIR, "symptom_router.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

try:
    router = load_router()
except Exception:
    router = None
    st.warning(
        "⚠️ symptom_router.pkl load করা যায়নি। "
        "Manual disease selection ব্যবহার করা যাবে।"
    )

DISPLAY_NAMES = {
    "heart": "heart_disease",
    "heart_disease": "heart_disease",
    "kidney": "chronic_kidney",
    "chronic_kidney": "chronic_kidney",
    "chronic_kidney_disease": "chronic_kidney",
    "diabetes": "diabetes",
    "breast": "breast_cancer",
    "breast_cancer": "breast_cancer",
    "liver": "liver_disease",
    "liver_disease": "liver_disease",
    "lung": "lung_cancer",
    "lung_cancer": "lung_cancer",
    "stroke": "stroke",
}

def normalize_disease(name):
    key = str(name).strip().lower().replace(" ", "_")
    return DISPLAY_NAMES.get(key, key)

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def extract_router_items(obj):
    if not isinstance(obj, dict):
        return []

    items = []
    for disease, value in obj.items():
        symptoms = []
        reports = []

        if isinstance(value, (list, tuple, set)):
            symptoms = list(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                key = str(k).lower()
                if "symptom" in key:
                    if isinstance(v, (list, tuple, set)):
                        symptoms.extend(v)
                    elif isinstance(v, str):
                        symptoms.append(v)
                if "report" in key or "test" in key:
                    if isinstance(v, (list, tuple, set)):
                        reports.extend(v)
                    elif isinstance(v, str):
                        reports.append(v)

        items.append({
            "disease": normalize_disease(disease),
            "symptoms": [clean_text(x) for x in symptoms],
            "reports": reports,
        })
    return items

def route_symptoms(text):
    if router is None:
        return []

    text = clean_text(text)
    results = []

    for item in extract_router_items(router):
        matched = [
            symptom for symptom in item["symptoms"]
            if symptom and symptom in text
        ]
        if matched:
            results.append({
                "disease": item["disease"],
                "score": len(set(matched)),
                "matched": list(dict.fromkeys(matched)),
                "reports": item["reports"],
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        top = results[0]["score"]
        results = [r for r in results if r["score"] == top]
    return results

# ============================================================
# INPUT HELPERS
# ============================================================
NUMERIC_DEFAULTS = {
    "Age": (1.0, 120.0, 30.0, 1.0),
    "age": (1.0, 120.0, 30.0, 1.0),
    "AGE": (1.0, 120.0, 30.0, 1.0),
    "Pregnancies": (0.0, 20.0, 1.0, 1.0),
    "Glucose": (40.0, 500.0, 120.0, 1.0),
    "BloodPressure": (40.0, 250.0, 80.0, 1.0),
    "BMI": (10.0, 80.0, 25.0, 0.1),
    "bmi": (10.0, 80.0, 25.0, 0.1),
    "Sc": (0.0, 20.0, 1.0, 0.1),
    "Sg": (1.0, 1.04, 1.02, 0.001),
    "Total_Bilirubin": (0.0, 100.0, 1.0, 0.1),
    "Direct_Bilirubin": (0.0, 100.0, 0.3, 0.1),
    "trestbps": (50.0, 250.0, 120.0, 1.0),
    "chol": (50.0, 700.0, 200.0, 1.0),
    "thalach": (50.0, 250.0, 150.0, 1.0),
    "oldpeak": (0.0, 10.0, 1.0, 0.1),
}

def numeric_input(column):
    pretty, source = COLUMN_INFO.get(
        column,
        (column.replace("_", " ").title(), "Relevant clinical/laboratory report")
    )
    st.caption(f"📄 Source: {source}")

    mn, mx, default, step = NUMERIC_DEFAULTS.get(
        column, (0.0, 100000.0, 0.0, 0.1)
    )

    return st.number_input(
        pretty,
        min_value=mn,
        max_value=mx,
        value=default,
        step=step,
        key=f"input_{column}",
    )

def categorical_input(column):
    pretty, source = COLUMN_INFO.get(
        column,
        (column.replace("_", " ").title(), "Relevant clinical history/report")
    )
    st.caption(f"📄 Source: {source}")

    if column in ["Gender", "gender", "GENDER", "sex"]:
        value = st.selectbox(pretty, ["Female", "Male"], key=f"cat_{column}")
        return 1 if value == "Male" else 0

    if column == "Rbc":
        value = st.selectbox(pretty, ["Normal", "Abnormal"], key=f"cat_{column}")
        return 0 if value == "Normal" else 1

    if column in ["Htn", "hypertension", "heart_disease", "fbs", "exang"]:
        value = st.selectbox(pretty, ["No", "Yes"], key=f"cat_{column}")
        return 1 if value == "Yes" else 0

    if column == "cp":
        return st.selectbox(
            pretty,
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Typical Angina",
                2: "Atypical Angina",
                3: "Non-anginal Pain",
                4: "Asymptomatic",
            }[x],
            key=f"cat_{column}",
        )

    if column == "thal":
        return st.selectbox(
            pretty,
            [3, 6, 7],
            format_func=lambda x: {
                3: "Normal",
                6: "Fixed Defect",
                7: "Reversible Defect",
            }[x],
            key=f"cat_{column}",
        )

    if column == "restecg":
        return st.selectbox(
            pretty, [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy",
            }[x],
            key=f"cat_{column}",
        )

    if column == "slope":
        return st.selectbox(
            pretty, [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping",
            }[x],
            key=f"cat_{column}",
        )

    if column == "ca":
        return st.selectbox(
            pretty,
            [0, 1, 2, 3, 4],
            format_func=lambda x: f"{x} major vessel(s)",
            key=f"cat_{column}",
        )

    value = st.selectbox(pretty, ["No", "Yes"], key=f"cat_{column}")
    return 1 if value == "Yes" else 0

# ============================================================
# BUILD MODEL INPUT
# ============================================================
def build_input(disease, columns):
    data = {}

    st.subheader("🧪 Patient Information")
    st.info(
        "Actual medical/laboratory report values দিন। "
        "The displayed labels are user-friendly; the original "
        "training feature names are preserved internally."
    )

    if disease == "stroke":
        # Base features
        for c in columns:
            if c == "gender":
                data[c] = categorical_input(c)
            elif c in ["age", "avg_glucose_level", "bmi"]:
                data[c] = numeric_input(c)
            elif c in ["hypertension", "heart_disease"]:
                data[c] = categorical_input(c)
            else:
                data[c] = 0

        if "ever_married_Yes" in columns:
            married = st.selectbox("Ever Married", ["No", "Yes"])
            data["ever_married_Yes"] = int(married == "Yes")

        work_cols = [c for c in columns if c.startswith("work_type_")]
        if work_cols:
            work = st.selectbox(
                "Work Type",
                ["Never_worked", "Private", "Self-employed", "children"]
            )
            for c in work_cols:
                data[c] = int(c == "work_type_" + work)

        if "Residence_type_Urban" in columns:
            residence = st.selectbox("Residence Type", ["Rural", "Urban"])
            data["Residence_type_Urban"] = int(residence == "Urban")

        smoking_cols = [c for c in columns if c.startswith("smoking_status_")]
        if smoking_cols:
            smoking = st.selectbox(
                "Smoking Status",
                ["formerly smoked", "never smoked", "smokes", "Unknown"]
            )
            for c in smoking_cols:
                data[c] = int(c == "smoking_status_" + smoking)

        return pd.DataFrame([data]).reindex(columns=columns)

    if disease == "lung_cancer":
        for c in columns:
            if c == "GENDER":
                data[c] = categorical_input(c)
            elif c == "AGE":
                data[c] = numeric_input(c)
            else:
                pretty, source = COLUMN_INFO.get(
                    c, (c.replace("_", " ").title(), "Clinical history")
                )
                st.caption(f"📄 Source: {source}")
                value = st.selectbox(pretty, ["No", "Yes"], key=f"lung_{c}")
                data[c] = int(value == "Yes")
        return pd.DataFrame([data]).reindex(columns=columns)

    for c in columns:
        if disease == "heart_disease" and c in [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ]:
            data[c] = categorical_input(c)
        elif disease == "chronic_kidney" and c in ["Rbc", "Htn"]:
            data[c] = categorical_input(c)
        elif disease == "liver_disease" and c == "Gender":
            data[c] = categorical_input(c)
        else:
            data[c] = numeric_input(c)

    return pd.DataFrame([data]).reindex(columns=columns)

# ============================================================
# PREPARE + PREDICT
# ============================================================
def prepare_input(input_df, columns):
    df = input_df.copy().reindex(columns=columns)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df.isnull().any().any():
        bad = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Invalid or missing value in: {bad}")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Input contains NaN or infinite values.")

    return df

def predict_disease(disease, input_df):
    model, scaler, columns = load_model_files(disease)
    df = prepare_input(input_df, columns)

    # Most of your models use scaler + model.
    # Lung model is kept unscaled to match the original app logic.
    if disease == "lung_cancer":
        X = df
    else:
        X = scaler.transform(df)

    prediction = model.predict(X)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0]

    return model, prediction, probability, df

# ============================================================
# RESULT DISPLAY
# ============================================================
def show_result(disease, model, prediction, probability):
    name = DISEASES[disease]["name"]

    st.divider()
    st.header("📊 Prediction Result")

    is_positive = str(prediction) == "1"

    if is_positive:
        st.error(f"⚠️ Possible Positive Result: {name}")

        st.warning(
            "The model detected a pattern associated with this condition. "
            "This is NOT a confirmed medical diagnosis."
        )

        st.subheader("🩺 Recommended Next Steps")
        for suggestion in SUGGESTIONS.get(disease, []):
            st.write(f"• {suggestion}")

    else:
        st.success(f"✅ Model Negative Result for {name}")

        st.info(
            "The model did not detect the target pattern for this input. "
            "This does not guarantee that the person is disease-free. "
            "If symptoms continue or worsen, seek professional medical evaluation."
        )

    if probability is not None:
        st.subheader("📈 Model Probability")

        classes = list(getattr(model, "classes_", range(len(probability))))
        rows = []

        for cls, prob in zip(classes, probability):
            label = "Positive" if str(cls) == "1" else "Negative"
            rows.append({
                "Class": label,
                "Model Probability": f"{prob * 100:.2f}%"
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "⚠️ Educational/research use only. Model output should be interpreted "
        "by a qualified healthcare professional."
    )

# ============================================================
# MODEL FEATURE GUIDE
# ============================================================
def show_feature_guide(columns):
    st.header("4️⃣ Model Features & Report Source")

    guide = []
    for c in columns:
        meaning, source = COLUMN_INFO.get(
            c,
            (c.replace("_", " ").title(), "Relevant clinical/laboratory report")
        )
        guide.append({
            "Feature": meaning,
            "Technical Name": c,
            "Value Source": source,
        })

    st.dataframe(
        pd.DataFrame(guide),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# HEART ENCODING GUIDE
# ============================================================
def show_heart_guide():
    st.subheader("❤️ Heart Disease Feature Guide")

    heart_guide = pd.DataFrame({
        "Feature": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ],
        "Meaning": [
            "Patient age",
            "Sex",
            "Chest pain type",
            "Resting blood pressure",
            "Serum cholesterol",
            "Fasting blood sugar above 120",
            "Resting ECG result",
            "Maximum heart rate",
            "Exercise-induced angina",
            "ST depression",
            "Peak exercise ST slope",
            "Number of major vessels",
            "Thalium stress-test result",
        ]
    })

    st.dataframe(
        heart_guide,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The encoded values are used internally by the trained model. "
        "During data entry, the app shows meaningful labels instead of "
        "asking the user to remember raw codes."
    )

# ============================================================
# MAIN SCREEN
# ============================================================
st.header("1️⃣ Patient Symptoms")

symptom_text = st.text_area(
    "Patient symptoms লিখুন",
    placeholder=(
        "Example:\n"
        "chest pain, shortness of breath, fatigue\n\n"
        "or\n"
        "frequent urination, excessive thirst, fatigue"
    ),
    height=130
)

c1, c2 = st.columns(2)

with c1:
    analyze = st.button(
        "🔎 Analyze Symptoms",
        type="primary",
        use_container_width=True
    )

with c2:
    manual = st.button(
        "🩺 Use Manual Disease Selection",
        use_container_width=True
    )

# ============================================================
# ROUTER
# ============================================================
if analyze:
    if not symptom_text.strip():
        st.warning("কমপক্ষে একটি symptom লিখুন.")
    elif router is None:
        st.warning(
            "Symptom router unavailable. Manual disease selection ব্যবহার করুন."
        )
    else:
        results = route_symptoms(symptom_text)
        if not results:
            st.warning(
                "Router কোনো matching disease খুঁজে পায়নি. "
                "Symptoms আরও specific করে লিখুন অথবা manual selection ব্যবহার করুন."
            )
        else:
            st.session_state["route_results"] = results
            st.session_state.pop("manual_mode", None)

if manual:
    st.session_state["manual_mode"] = True
    st.session_state.pop("route_results", None)

# ============================================================
# SELECT DISEASE
# ============================================================
selected = None
selected_result = {}

if "route_results" in st.session_state:
    results = st.session_state["route_results"]

    st.header("2️⃣ Possible Diseases")

    max_score = max(r["score"] for r in results)
    top_results = [r for r in results if r["score"] == max_score]

    if len(top_results) == 1:
        disease_name = DISEASES.get(
            top_results[0]["disease"], {}
        ).get("name", top_results[0]["disease"])

        st.success(
            f"🎯 Most Possible Disease: **{disease_name}** "
            f"(matched symptoms: {max_score})"
        )
    else:
        st.warning(
            f"🎯 {len(top_results)}টি disease-এর equal highest symptom match পাওয়া গেছে."
        )

    table = []
    for r in top_results:
        if r["disease"] in DISEASES:
            table.append({
                "Disease": DISEASES[r["disease"]]["name"],
                "Matched Symptoms": ", ".join(r["matched"]),
                "Match Score": r["score"],
            })

    if table:
        st.dataframe(
            pd.DataFrame(table),
            use_container_width=True,
            hide_index=True
        )

    available = [r["disease"] for r in top_results if r["disease"] in DISEASES]

    if available:
        selected = st.selectbox(
            "Select disease model",
            available,
            format_func=lambda x: DISEASES[x]["name"]
        )
        selected_result = next(
            r for r in top_results if r["disease"] == selected
        )

elif st.session_state.get("manual_mode"):
    st.header("2️⃣ Select Disease Manually")

    selected = st.selectbox(
        "Choose disease model",
        list(DISEASES.keys()),
        format_func=lambda x: DISEASES[x]["name"]
    )

# ============================================================
# SELECTED MODEL
# ============================================================
if selected:

    st.header("3️⃣ Required Reports")

    st.write(f"### {DISEASES[selected]['name']}")

    for i, report in enumerate(REPORTS.get(selected, []), 1):
        st.write(f"**{i}.** {report}")

    if selected_result.get("reports"):
        st.subheader("🔎 Router Report Suggestions")
        for report in selected_result["reports"]:
            st.write(f"• {report}")

    # Load model
    try:
        model, scaler, columns = load_model_files(selected)
    except Exception as e:
        st.error("Selected disease model load করা যায়নি.")
        st.exception(e)
        st.stop()

    # Protect against malformed columns.pkl
    if not isinstance(columns, (list, tuple, np.ndarray)):
        st.error(
            "❌ columns.pkl একটি valid list/array নয়. "
            "Training-এর সময় columns.pkl ঠিকভাবে save করুন."
        )
        st.stop()

    columns = list(columns)

    if not columns:
        st.error("❌ columns.pkl empty.")
        st.stop()

    # Heart verification
    if selected == "heart_disease":
        expected = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]

        if columns == expected:
            st.success("✅ Heart model: 13 training features loaded correctly.")
        else:
            st.warning(
                "⚠️ Heart model-এর columns expected 13 features-এর সাথে match করছে না."
            )
            st.code(str(columns))

    show_feature_guide(columns)

    if selected == "heart_disease":
        show_heart_guide()

    st.header("5️⃣ Enter Patient Data")

    with st.form(f"form_{selected}"):
        input_df = build_input(selected, columns)

        submitted = st.form_submit_button(
            "🩺 Predict",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            model, prediction, probability, final_input = predict_disease(
                selected, input_df
            )

            show_result(
                selected,
                model,
                prediction,
                probability
            )

            with st.expander("🔍 View Technical Model Input"):
                st.caption(
                    "These are the exact numeric values sent to the trained model. "
                    "They may contain encoded values such as 0/1 or dataset codes; "
                    "this is normal and is hidden from the main feature guide."
                )

                st.dataframe(
                    final_input,
                    use_container_width=True,
                    hide_index=True
                )

                st.write("Input shape:", final_input.shape)
                st.write("Training columns:", columns)

        except Exception as e:
            st.error("❌ Prediction করার সময় error হয়েছে.")
            st.error(f"Error Type: {type(e).__name__}")
            st.exception(e)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Healthcare Agent | Multi-Disease ML Screening System")
st.caption("For educational and research purposes only.")