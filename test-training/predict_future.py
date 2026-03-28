import sys
import json
import joblib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    # =====================================================
    # 🔥 FLEXIBLE INPUT HANDLING (JSON OR CLI ARGS)
    # =====================================================
    
    if len(sys.argv) == 2:
        # JSON input mode
        input_json = sys.argv[1]
        live_data = json.loads(input_json)
    
    elif len(sys.argv) >= 3:
        # CLI args mode (your Node.js case)
        hour = int(sys.argv[1])
        is_weekend = int(sys.argv[2])

        live_data = {
            "Hour": hour,
            "is_weekend": is_weekend
        }
    
    else:
        raise Exception("Invalid input format")

    # =====================================================
    # 🔥 LOAD MODEL + COLUMNS
    # =====================================================
    
    model = joblib.load("xgboost_simulated_model.joblib")
    
    with open("simulated_model_columns.json", "r") as f:
        model_columns = json.load(f)

    # =====================================================
    # 🔥 PREPARE INPUT
    # =====================================================
    
    df = pd.DataFrame([live_data])

    # Align with training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # =====================================================
    # 🔥 PREDICT
    # =====================================================
    
    prediction = model.predict(df)[0]

    output = {
        "status": "success",
        "predicted_crowd": max(0, int(round(prediction)))
    }

    print(json.dumps(output))


except Exception as e:
    print(json.dumps({
        "status": "error",
        "message": str(e)
    }))