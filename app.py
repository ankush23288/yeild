from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.express as px
import plotly.io as pio # Import plotly.io
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score # Keep r2_score for training log

app = Flask(__name__)

# Define features and target
features = ["soil_moisture", "soil_pH", "N", "P", "K", "rainfall", "temperature"]
target = "yield"

# Define slider_cfg globally
slider_cfg = {
    "soil_moisture": (10, 45, 0.5, 28),
    "soil_pH": (5.4, 7.8, 0.1, 6.5),
    "N": (100, 300, 5, 180),
    "P": (20, 120, 5, 60),
    "K": (50, 250, 5, 120),
    "rainfall": (0, 15, 0.2, 4),
    "temperature": (5, 38, 0.5, 22),
}


# --- Model Loading and Initialization ---
models = {}
scalers = {}
selected_feats = {}
explainers = {}

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

# Load data and train models
try:
    # Assuming train_farms.csv is in the same directory as app.py
    df = pd.read_csv("train_farms.csv")

    for crop in df["crop"].unique():
        print(f"\nTraining model for {crop}...")
        df_crop = df[df["crop"] == crop].copy()
        X, y = df_crop[features], df_crop[target]
        # Use a consistent random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

        model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)
        print(f"{crop} → R²: {r2_score(y_test, pred):.3f} | RMSE: {rmse(y_test, pred):.1f}") # Keep R2 and RMSE for training log

        # Cost-based selection (copied from notebook)
        costs = {"soil_moisture": 12, "soil_pH": 18, "N": 25, "P": 25, "K": 25, "rainfall": 0, "temperature": 0}
        importances = model.feature_importances_
        adj_imp = [i / costs.get(f, 1) for f, i in zip(features, importances)]
        threshold = 0.08
        sel_feat = [f for f, a in zip(features, adj_imp) if a > threshold]
        print(f"Selected: {sel_feat}")

        # Retrain with selected features
        X_train_sel = X_train_scaled[sel_feat]
        sel_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
        sel_model.fit(X_train_sel, y_train)

        # SHAP Explainer
        if not X_train_sel.empty:
            # Use up to 10 samples for background
            background = shap.kmeans(X_train_sel, min(10, X_train_sel.shape[0]))
            explainer = shap.KernelExplainer(lambda x: sel_model.predict(x), background)
        else:
             explainer = None


        models[crop] = sel_model
        scalers[crop] = scaler
        selected_feats[crop] = sel_feat
        explainers[crop] = explainer

    print("\nAll models trained and loaded successfully!")
except FileNotFoundError:
    print("train_farms.csv not found. Please ensure the file is in the same directory as app.py.")
except Exception as e:
    print(f"Error loading data or training models: {e}")


# Helper function for recommendations (copied from notebook)
def get_recommendations(shap_df):
    recommendations = []
    for _, row in shap_df.iterrows():
        feature = row["Feature"]
        shap_value = row["SHAP"]

        if shap_value < -1:
            strength = "significantly low"
        elif shap_value < -0.5:
            strength = "moderately low"
        elif shap_value < -0.15:
            strength = "slightly low"
        else:
            strength = "optimal"

        if strength != "optimal":
            if "moisture" in feature:
                recommendations.append(f"Increase irrigation as soil moisture is {strength}.")
            elif "pH" in feature:
                recommendations.append(f"Adjust pH with lime/sulfur as soil pH is {strength}.")
            elif feature in ("N", "P", "K"):
                recommendations.append(f"Apply {feature} fertilizer as {feature} levels are {strength}.")
            elif "rain" in feature:
                recommendations.append(f"Consider supplemental irrigation as rainfall is {strength}.")
            elif "temperature" in feature:
                 recommendations.append(f"Monitor temperature and adjust practices accordingly as temperature is {strength} for yield.")
        else:
             if "moisture" in feature:
                recommendations.append("Maintain optimal irrigation.")
             elif "pH" in feature:
                recommendations.append("Maintain optimal pH levels.")
             elif feature in ("N", "P", "K"):
                recommendations.append(f"Maintain optimal {feature} levels.")
             elif "rain" in feature:
                recommendations.append("Ensure adequate drainage if rainfall is high.")
             elif "temperature" in feature:
                recommendations.append("Monitor temperature and adjust practices accordingly.")


    if not recommendations:
        recommendations.append("Maintain current practices.")
    return recommendations


@app.route('/')
def index():
    available_crops = list(models.keys())
    return render_template('index.html', features=features, slider_cfg=slider_cfg, crops=available_crops)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    crop = data['crop']
    inp_vals = data['features']

    if crop not in models or explainers.get(crop) is None:
        return jsonify({'error': f'Model not trained or explainer not available for {crop}. Please ensure train_farms.csv was loaded correctly.'}), 400

    inp = pd.DataFrame([inp_vals])

    scaler = scalers[crop]
    sel_feat = selected_feats[crop]
    model = models[crop]
    explainer = explainers[crop]

    try:
        inp_scaled = pd.DataFrame(scaler.transform(inp[features]), columns=features)
    except KeyError as e:
         return jsonify({'error': f'Missing input feature: {e}. Please provide all required features.'}), 400

    try:
        inp_sel = inp_scaled[sel_feat]
    except KeyError as e:
        return jsonify({'error': f'Selected feature missing after scaling: {e}. This might indicate an issue with feature selection or input features.'}), 500

    pred = model.predict(inp_sel)[0]

    # SHAP values
    shap_val = explainer.shap_values(inp_sel)[0]

    # Create SHAP DataFrame for recommendations and JSON output
    shap_df = pd.DataFrame({"Feature": sel_feat, "SHAP": shap_val}).sort_values("SHAP", ascending=False)
    shap_data = shap_df.to_dict(orient='records')

    # Recommendations
    recommendations = get_recommendations(shap_df)

    # Create Plotly figure
    fig = px.bar(shap_df, x="SHAP", y="Feature", color="SHAP", color_continuous_scale="RdYlGn",
                 title="Why This Yield?")

    # Convert Plotly figure to JSON
    plotly_json = pio.to_json(fig)

    return jsonify({
        'prediction': float(pred),
        'shap_data': shap_data, # Keep shap_data in case it's needed for something else
        'recommendations': recommendations,
        'plotly_json': plotly_json # Include Plotly JSON
    })

# Serve static files from the static directory
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Define slider_cfg here as well for access in index route - already defined globally
    # app.run(debug=True) # Removed debug=True for potential production use, though it's fine for development
    app.run(debug=True, port=5000, host='0.0.0.0') # Allow access from outside the container