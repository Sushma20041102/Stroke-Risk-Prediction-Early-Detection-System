import { useState } from "react";
import "./App.css";

function App() {
  const [form, setForm] = useState({
    age: "",
    gender: "Male",
    hypertension: 0,
    heart_disease: 0,
    ever_married: "Yes",
    work_type: "Private",
    Residence_type: "Urban",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "never smoked"
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const submit = async () => {
  const res = await fetch(
  "https://stroke-ai-predictor-9ep6.onrender.com/predict",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(form)
  }
);


    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="app-container">
      <h1 className="app-title">🧠 Stroke Risk Predictor</h1>
      <p className="app-subtitle">Enter your health information below</p>

      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Age</label>
          <input 
            name="age" 
            type="number"
            className="form-input"
            placeholder="Enter age" 
            onChange={handleChange}
            value={form.age}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Gender</label>
          <select name="gender" className="form-select" onChange={handleChange} value={form.gender}>
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Hypertension</label>
          <select name="hypertension" className="form-select" onChange={handleChange} value={form.hypertension}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Heart Disease</label>
          <select name="heart_disease" className="form-select" onChange={handleChange} value={form.heart_disease}>
            <option value="0">No</option>
            <option value="1">Yes</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Ever Married</label>
          <select name="ever_married" className="form-select" onChange={handleChange} value={form.ever_married}>
            <option>Yes</option>
            <option>No</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Work Type</label>
          <select name="work_type" className="form-select" onChange={handleChange} value={form.work_type}>
            <option>Private</option>
            <option>Self-employed</option>
            <option>Govt_job</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Residence Type</label>
          <select name="Residence_type" className="form-select" onChange={handleChange} value={form.Residence_type}>
            <option>Urban</option>
            <option>Rural</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Average Glucose Level</label>
          <input 
            name="avg_glucose_level" 
            type="number"
            step="0.1"
            className="form-input"
            placeholder="e.g., 95.5" 
            onChange={handleChange}
            value={form.avg_glucose_level}
          />
        </div>

        <div className="form-group">
          <label className="form-label">BMI (Body Mass Index)</label>
          <input 
            name="bmi" 
            type="number"
            step="0.1"
            className="form-input"
            placeholder="e.g., 28.5" 
            onChange={handleChange}
            value={form.bmi}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Smoking Status</label>
          <select name="smoking_status" className="form-select" onChange={handleChange} value={form.smoking_status}>
            <option>never smoked</option>
            <option>formerly smoked</option>
            <option>smokes</option>
            <option>Unknown</option>
          </select>
        </div>
      </div>

      <button className="predict-button" onClick={submit}>
        🔍 Predict Stroke Risk
      </button>

      {result && (
        <div className={`result-container ${result.stroke_risk === 1 ? "result-high" : "result-low"}`}>
          <h2 className="result-text">
            Risk Level: {result.risk_level}
          </h2>
        </div>
      )}
    </div>
  );
}

export default App;
