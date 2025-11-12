import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, abort, send_from_directory

app = Flask(__name__)

# Paths to artifacts
BASEDIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASEDIR, "model.pkl")
SCALER_PATH = os.path.join(BASEDIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASEDIR, "feature_names.pkl")
BUILD_DIR = os.path.join(BASEDIR, "frontend", "build")

# Load model, scaler and feature names if available
model = None
scaler = None
feature_names = None

def _load_artifact(path, name):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            app.logger.exception("Failed loading %s", name)
    return None

model = _load_artifact(MODEL_PATH, "model")
scaler = _load_artifact(SCALER_PATH, "scaler")
feature_names = _load_artifact(FEATURES_PATH, "feature_names")

if model is None:
    raise RuntimeError(
        "Model not found. Create and save model.pkl in the project folder before starting the API."
    )

def _prepare_array_from_payload(payload):
    """
    Accept payload formats:
      - {"features": [v1, v2, ...]}           -> single instance list
      - {"features": {"f1": v1, ...}}         -> single instance dict keyed by feature names
      - {"instances": [[...], [...]]}         -> batch list of lists
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
        if not isinstance(inst, (list, tuple)):
            abort(400, "'instances' must be a list.")
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

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "features": feature_names}), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data: 
        abort(400, "JSON body required")
    X = _prepare_array_from_payload(data)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            app.logger.exception("Scaler.transform failed")
            abort(400, "Failed to scale input features (check types/shape).")
    try:
        preds = model.predict(X)
    except Exception:
        app.logger.exception("Model prediction failed")
        abort(500, "Model prediction failed")
    preds_list = [float(p) for p in np.ravel(preds)]
    if X.shape[0] == 1:
        return jsonify({"prediction": preds_list[0]})
    return jsonify({"predictions": preds_list})

# Serve React build (if present) or a helpful message
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    # If build exists, serve static files and index.html for SPA routes
    if os.path.isdir(BUILD_DIR):
        if path and os.path.exists(os.path.join(BUILD_DIR, path)):
            return send_from_directory(BUILD_DIR, path)
        index_path = os.path.join(BUILD_DIR, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(BUILD_DIR, "index.html")
    # Build not present: provide minimal JSON info
    return jsonify({
        "message": "Frontend not built. During development run the React dev server. To serve frontend from Flask, run 'npm run build' in frontend/ and restart this app.",
        "api_endpoints": {
            "health": "/api/health",
            "predict": "/predict"
        }
    }), 200

# JSON error handlers
@app.errorhandler(400)
def handle_400(err):
    desc = getattr(err, "description", str(err))
    return jsonify({"error": str(desc)}), 400

@app.errorhandler(404)
def handle_404(err):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def handle_500(err):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)