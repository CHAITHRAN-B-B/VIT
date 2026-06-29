import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (selectedFile) => {
    if (selectedFile.type && selectedFile.type.startsWith("image/")) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    } else {
      alert("Please upload an image file");
    }
  };

  const onBtnClick = () => {
    inputRef.current.click();
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

      const data = res.data;
      if (data.error) {
        alert(`Backend error: ${data.error}`);
      } else {
        setResult(data);
      }
    } catch (err) {
      alert("Prediction failed");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <header className="header">
          <h1>AI vs Real Image Detection</h1>
        </header>

        {!preview ? (
          <div 
            className={`dropzone ${dragActive ? "drag-active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={onBtnClick}
          >
            <input 
              ref={inputRef}
              type="file" 
              accept="image/*" 
              onChange={(e) => {
                if(e.target.files && e.target.files[0]) {
                  handleFileChange(e.target.files[0]);
                }
              }} 
              style={{ display: "none" }} 
            />
            <div className="dropzone-text">Drag & Drop your image here</div>
            <div className="dropzone-subtext">or click to browse from your device</div>
          </div>
        ) : (
          <div className="preview-container">
            <img src={preview} alt="preview" className="preview-image" />
            {!result ? (
              <button onClick={handlePredict} className="submit-btn" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner"></span> Analyzing...
                  </>
                ) : (
                  "Analyze"
                )}
              </button>
            ) : null}
          </div>
        )}

        {result && (
          <div className="results-section">
            <div className={`verdict-box ${result.prediction}`}>
              <div className="verdict-title">{result.prediction}</div>
              <div className="verdict-confidence">Confidence Level: {result.confidence}%</div>
            </div>
            
            <div style={{ textAlign: "center" }}>
              <button className="reset-btn" onClick={reset}>Analyze Another Image</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
