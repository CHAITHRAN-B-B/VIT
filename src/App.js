import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return alert("Please upload an image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData
      );

      setResult(res.data);
    } catch (err) {
      alert("Prediction failed");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getResultColor = () =>
    result?.prediction === "real" ? "#2ecc71" : "#e74c3c";

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>AI vs Real Image Detector</h1>
        

        <input type="file" accept="image/*" onChange={handleFileChange} />

        {preview && (
          <img src={preview} alt="preview" style={styles.image} />
        )}

        <button onClick={handlePredict} style={styles.button}>
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {result && (
          <div
            style={{
              ...styles.resultBox,
              borderColor: getResultColor(),
              color: getResultColor(),
            }}
          >
            <strong>{result.prediction.toUpperCase()}</strong>
            <span>Confidence: {result.confidence}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #667eea, #764ba2)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  card: {
    background: "#fff",
    padding: "30px",
    borderRadius: "12px",
    width: "360px",
    textAlign: "center",
    boxShadow: "0 10px 30px rgba(0,0,0,0.15)",
  },
  title: {
    marginBottom: "5px",
    color: "#2c3e50",
  },
  subtitle: {
    fontSize: "14px",
    color: "#7f8c8d",
    marginBottom: "20px",
  },
  image: {
    marginTop: "15px",
    maxWidth: "100%",
    borderRadius: "8px",
  },
  button: {
    marginTop: "20px",
    padding: "10px",
    width: "100%",
    border: "none",
    borderRadius: "8px",
    background: "#667eea",
    color: "#fff",
    fontSize: "16px",
    cursor: "pointer",
  },
  resultBox: {
    marginTop: "20px",
    padding: "12px",
    border: "2px solid",
    borderRadius: "8px",
    fontSize: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "5px",
  },
};

export default App;
