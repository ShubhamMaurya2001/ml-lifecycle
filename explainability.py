import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# Utility to load a model by path
def load_model(model_path):
    if not os.path.isabs(model_path):
        base = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base, model_path)
    return joblib.load(model_path)

# Compute SHAP values for a given scikit-learn model and dataset (X)
# Returns (explainer, shap_values)
def compute_shap(model, X, nsamples=100):
    """
    Compute SHAP values for a model using background data X.
    Uses TreeExplainer for tree-based models, KernelExplainer for others.
    """
    # For ridge regression and linear models, use KernelExplainer
    try:
        explainer = shap.KernelExplainer(model.predict, X)
        print(f"Using KernelExplainer for model type: {type(model).__name__}")
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"KernelExplainer failed: {e}, trying TreeExplainer...")
        try:
            # Tree-based models
            explainer = shap.TreeExplainer(model)
            print(f"Using TreeExplainer for model type: {type(model).__name__}")
            shap_values = explainer.shap_values(X)
        except Exception as e2:
            print(f"TreeExplainer also failed: {e2}")
            raise

    return explainer, shap_values

# Save a SHAP summary plot to PNG
def save_shap_summary_plot(shap_values, X, out_path):
    """Save SHAP summary plot to PNG file."""
    plt.figure(figsize=(10, 6))
    try:
        # Handle different shap_values formats
        if isinstance(shap_values, list):
            # For binary classification, use the positive class
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        
        # Ensure X and sv are properly aligned
        if hasattr(sv, 'values'):
            sv = sv.values
        
        shap.summary_plot(sv, X, plot_type="bar", show=False)
    except Exception as e:
        print(f"Bar plot failed, trying dot plot: {e}")
        try:
            shap.summary_plot(sv, X, show=False)
        except Exception as e2:
            print(f"Dot plot failed: {e2}")
            raise
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {out_path}")

# Helper: load sample rows from the cleaned csv
def load_sample_rows(n=5):
    """Load sample rows from the forest fires dataset."""
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'Algerian_forest_fires_cleaned.csv')
    df = pd.read_csv(csv_path)
    # Select relevant features if present
    features = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI']
    present = [c for c in features if c in df.columns]
    return df[present].head(n)

if __name__ == '__main__':
    # quick local test when run as a script
    try:
        model = load_model('Models/ridge_model.pkl')
        X = load_sample_rows(50)
        print(f"Loaded {len(X)} sample rows with features: {list(X.columns)}")
        explainer, shap_values = compute_shap(model, X, nsamples=50)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shap_summary.png')
        save_shap_summary_plot(shap_values, X, out)
        print(f'✓ Successfully saved SHAP summary to {out}')
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
