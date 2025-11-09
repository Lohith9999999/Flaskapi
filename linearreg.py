from flask import Flask, request, jsonify, abort
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

# Load model, scaler and feature names if available
model = None
scaler = None
feature_names = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)

if model is None:
    raise RuntimeError("Model not found. Create and save model.pkl in the project folder before starting the API.")

def _prepare_array_from_payload(payload):
    """
    Accepts payload in multiple forms:
      - {"features": [v1, v2, ...]}           -> single instance list
      - {"features": {"f1": v1, ...}}         -> single instance dict keyed by feature names
      - {"instances": [[...], [...]]}          -> batch list of lists
      - {"instances": [{"f1":v1,...}, ...]}   -> batch list of dicts
    Returns numpy array of shape (n_samples, n_features)
    """
    if "features" in payload:
        feats = payload["features"]
        if isinstance(feats, dict):
            if feature_names is None:
                abort(400, "Feature names not available on server; send features as positional list.")
            row = [feats.get(fn) for fn in feature_names]
            return np.array([row], dtype=float)
        elif isinstance(feats, (list, tuple)):
            return np.array([feats], dtype=float)
        else:
            abort(400, "Invalid 'features' format. Must be list or dict.")
    elif "instances" in payload:
        inst = payload["instances"]
        if len(inst) == 0:
            abort(400, "Empty 'instances' list.")
        if all(isinstance(x, dict) for x in inst):
            if feature_names is None:
                abort(400, "Feature names not available; send instances as list-of-lists.")
            arr = [[x.get(fn) for fn in feature_names] for x in inst]
            return np.array(arr, dtype=float)
        elif all(isinstance(x, (list, tuple)) for x in inst):
            return np.array(inst, dtype=float)
        else:
            abort(400, "Invalid 'instances' contents. Use list-of-lists or list-of-dicts.")
    else:
        abort(400, "JSON must include 'features' or 'instances'.")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "features": feature_names}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        abort(400, "JSON body required")
    X = _prepare_array_from_payload(data)
    if scaler is not None:
        X = scaler.transform(X)
    preds = model.predict(X)
    # convert numpy floats to Python floats
    preds_list = [float(p) for p in np.ravel(preds)]
    # If single instance, return scalar prediction
    if X.shape[0] == 1:
        return jsonify({"prediction": preds_list[0]})
    return jsonify({"predictions": preds_list})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)