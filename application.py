import os
import joblib
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import explainability as expl

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models using absolute paths
ridge_model = joblib.load(os.path.join(script_dir, 'Models/ridge_model.pkl'))
standard_scaler = joblib.load(os.path.join(script_dir, 'Models/scaler.pkl'))


application = Flask(__name__, 
                   template_folder=os.path.join(script_dir, 'Templates'),
                   static_folder=os.path.join(script_dir, 'static'))
app = application

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        data = request.form.to_dict()
        
        # Extract features in the correct order (matching training data)
        features = [
            float(data['Temperature']),
            float(data['RH']),
            float(data['Ws']),
            float(data['Rain']),
            float(data['FFMC']),
            float(data['DMC']),
            float(data['DC']),
            float(data['ISI']),
            float(data['BUI'])
        ]
        
        # Reshape for prediction
        features_array = np.array([features])
        scaled_features = standard_scaler.transform(features_array)
        
        # Make prediction
        prediction = ridge_model.predict(scaled_features)[0]
        
        return render_template('home.html', prediction_result=prediction)
    else:
        return render_template('home.html')

@app.route('/explain', methods=['GET','POST'])
def explain():
    """Generate SHAP explanation image for an input."""
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI']
            row = {c: float(data.get(c, 0)) for c in cols}
            X_row = pd.DataFrame([row])

            # Load background sample for SHAP
            try:
                X_bg = expl.load_sample_rows(50)
                print(f"Loaded background sample with shape: {X_bg.shape}")
            except Exception as e:
                print(f"Failed to load background sample: {e}")
                X_bg = X_row

            # Compute SHAP
            print("Computing SHAP values...")
            explainer, shap_values = expl.compute_shap(ridge_model, X_bg, nsamples=50)
            
            # Save plot
            static_dir = os.path.join(script_dir, 'static')
            os.makedirs(static_dir, exist_ok=True)
            out_path = os.path.join(static_dir, 'shap_summary.png')
            print(f"Saving SHAP plot to {out_path}")
            expl.save_shap_summary_plot(shap_values, X_bg, out_path)
            
            return render_template('home.html', shap_image='shap_summary.png', shap_success=True)
        except Exception as e:
            import traceback
            error_msg = f"Error computing SHAP: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return render_template('home.html', shap_error=error_msg)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host = "0.0.0.0")