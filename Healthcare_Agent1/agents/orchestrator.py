import os
import joblib


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


# ============================================================
# LOAD SYMPTOM ROUTER
# ============================================================

ROUTER_PATH = os.path.join(
    MODEL_DIR,
    "symptom_router.pkl"
)

symptom_router = joblib.load(
    ROUTER_PATH
)


# ============================================================
# ROUTER DISEASE → MODEL FOLDER MAPPING
# ============================================================

DISEASE_FOLDER_MAP = {

    "breast_cancer":
        "breast_cancer",

    "chronic_kidney_disease":
        "chronic_kidney",

    "diabetes":
        "diabetes",

    "heart_disease":
        "heart_disease",

    "liver_disease":
        "liver_disease",

    "lung_cancer":
        "lung_cancer",

    "stroke":
        "stroke"
}


# ============================================================
# DISPLAY NAMES
# ============================================================

DISEASE_DISPLAY_NAMES = {

    "breast_cancer":
        "Breast Cancer",

    "chronic_kidney_disease":
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
# REQUIRED REPORTS
# ============================================================

REQUIRED_REPORTS = {

    "breast_cancer": [
        "Breast examination",
        "Mammography",
        "Breast ultrasound",
        "Clinical evaluation"
    ],

    "chronic_kidney_disease": [
        "Blood Pressure measurement",
        "Kidney Function Test (KFT/RFT)",
        "Serum Creatinine",
        "Blood Urea",
        "Serum Electrolytes",
        "Urine Routine Examination",
        "Complete Blood Count (CBC)"
    ],

    "diabetes": [
        "Blood Glucose Test",
        "Fasting Blood Glucose",
        "HbA1c",
        "Blood Pressure measurement"
    ],

    "heart_disease": [
        "Blood Pressure measurement",
        "Lipid Profile",
        "ECG",
        "Heart rate measurement",
        "Clinical cardiac evaluation"
    ],

    "liver_disease": [
        "Liver Function Test (LFT)",
        "Bilirubin",
        "ALT",
        "AST",
        "Albumin",
        "Total Protein"
    ],

    "lung_cancer": [
        "Chest examination",
        "Chest X-ray",
        "CT scan if clinically indicated",
        "Smoking history",
        "Clinical evaluation"
    ],

    "stroke": [
        "Blood Pressure measurement",
        "Blood Glucose",
        "Lipid Profile",
        "Neurological evaluation",
        "Clinical assessment"
    ]
}


# ============================================================
# FIND DISEASES FROM SYMPTOMS
# ============================================================

def find_diseases_from_symptoms(
    symptoms
):

    """
    symptoms:
        list of patient symptoms

    Example:
        [
            "cough",
            "chest pain",
            "smoking"
        ]
    """

    if isinstance(
        symptoms,
        str
    ):

        symptoms = [
            symptoms
        ]


    # --------------------------------------------------------
    # Normalize symptoms
    # --------------------------------------------------------

    user_symptoms = {

        str(symptom).strip().lower()

        for symptom in symptoms
    }


    disease_scores = {}


    # ========================================================
    # CHECK EACH DISEASE
    # ========================================================

    for disease, symptom_list in symptom_router.items():

        # Make sure symptom_list is iterable
        if not isinstance(
            symptom_list,
            (list, tuple, set)
        ):
            continue


        router_symptoms = {

            str(symptom).strip().lower()

            for symptom in symptom_list
        }


        # ----------------------------------------------------
        # Matching symptoms
        # ----------------------------------------------------

        matched = (
            user_symptoms
            &
            router_symptoms
        )


        disease_scores[disease] = {

            "score": len(matched),

            "matched_symptoms": sorted(
                matched
            ),

            "total_symptoms": len(
                router_symptoms
            )
        }


    # ========================================================
    # SORT BY MATCHING SCORE
    # ========================================================

    ranked_diseases = sorted(

        disease_scores.items(),

        key=lambda x: x[1]["score"],

        reverse=True
    )


    return ranked_diseases


# ============================================================
# GET TOP DISEASES
# ============================================================

def get_possible_diseases(
    symptoms,
    top_n=3
):

    ranked = find_diseases_from_symptoms(
        symptoms
    )


    # Only diseases with at least
    # one matching symptom

    matched = [

        item

        for item in ranked

        if item[1]["score"] > 0
    ]


    return matched[:top_n]


# ============================================================
# GET REQUIRED REPORTS
# ============================================================

def get_required_reports(
    disease
):

    return REQUIRED_REPORTS.get(
        disease,
        []
    )


# ============================================================
# GET MODEL FOLDER
# ============================================================

def get_model_folder(
    disease
):

    return DISEASE_FOLDER_MAP.get(
        disease
    )


# ============================================================
# GET DISEASE DISPLAY NAME
# ============================================================

def get_display_name(
    disease
):

    return DISEASE_DISPLAY_NAMES.get(
        disease,
        disease
    )


# ============================================================
# COMPLETE ROUTING FUNCTION
# ============================================================

def route_patient(
    symptoms,
    top_n=3
):

    possible_diseases = (
        get_possible_diseases(
            symptoms,
            top_n
        )
    )


    results = []


    for disease, information in possible_diseases:

        results.append({

            "disease": disease,

            "display_name":
                get_display_name(
                    disease
                ),

            "match_score":
                information["score"],

            "matched_symptoms":
                information[
                    "matched_symptoms"
                ],

            "required_reports":
                get_required_reports(
                    disease
                ),

            "model_folder":
                get_model_folder(
                    disease
                )
        })


    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_symptoms = [

        "cough",

        "smoking",

        "chest pain",

        "shortness of breath"
    ]


    print("\n")
    print("=" * 60)
    print("SYMPTOM ROUTER TEST")
    print("=" * 60)


    results = route_patient(
        test_symptoms
    )


    for result in results:

        print(
            "\nDisease:",
            result["display_name"]
        )

        print(
            "Match Score:",
            result["match_score"]
        )

        print(
            "Matched Symptoms:",
            result["matched_symptoms"]
        )

        print(
            "Required Reports:"
        )

        for report in result[
            "required_reports"
        ]:

            print(
                "  -",
                report
            )

        print(
            "Model Folder:",
            result["model_folder"]
        )


    print("\n")
    print("=" * 60)