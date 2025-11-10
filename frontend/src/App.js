import React, { useEffect, useState } from "react";

export default function App() {
  const [featureNames, setFeatureNames] = useState(null);
  const [values, setValues] = useState([]);
  const [nFeatures, setNFeatures] = useState(4);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // fetch feature names (optional)
    fetch("/api/health")
      .then((r) => r.json())
      .then((j) => {
        if (j.features) {
          setFeatureNames(j.features);
          setNFeatures(j.features.length);
          setValues(Array(j.features.length).fill(""));
        } else {
          setValues(Array(nFeatures).fill(""));
        }
      })
      .catch(() => {
        setValues(Array(nFeatures).fill(""));
      });
    // eslint-disable-next-line
  }, []);

  useEffect(() => {
    setValues((v) => {
      const copy = Array.from(v || []);
      while (copy.length < nFeatures) copy.push("");
      while (copy.length > nFeatures) copy.pop();
      return copy;
    });
  }, [nFeatures]);

  function handleChange(i, val) {
    const copy = [...values];
    copy[i] = val;
    setValues(copy);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const numeric = values.map((x) => Number(x));
      const payload = featureNames ? { features: Object.fromEntries(featureNames.map((fn, idx) => [fn, numeric[idx]])) } : { features: numeric };
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Status ${res.status}`);
      }
      const data = await res.json();
      setResult(data.prediction ?? data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Model Predictor</h1>
      <p>Enter feature values and press Predict.</p>

      <label>
        Number of features:
        <input
          type="number"
          min="1"
          value={nFeatures}
          onChange={(e) => setNFeatures(Number(e.target.value))}
          disabled={!!featureNames}
        />
        {featureNames ? " (using server feature names)" : ""}
      </label>

      <form onSubmit={handleSubmit}>
        <div className="grid">
          {Array.from({ length: nFeatures }).map((_, i) => (
            <div key={i} className="field">
              <label>{featureNames ? featureNames[i] : `f${i + 1}`}</label>
              <input
                type="number"
                step="any"
                value={values[i] ?? ""}
                onChange={(e) => handleChange(i, e.target.value)}
                required
              />
            </div>
          ))}
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <div className="error">Error: {String(error)}</div>}
      {result !== null && (
        <div className="result">
          <strong>Prediction:</strong> {Array.isArray(result) ? JSON.stringify(result) : String(result)}
        </div>
      )}
    </div>
  );
}