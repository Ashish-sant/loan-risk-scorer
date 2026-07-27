import { useState } from "react";
import axios from "axios";
import "./App.css";

const fields = [
  { key: "RevolvingUtilizationOfUnsecuredLines", label: "Credit Utilization (0-1)", def: 0.4 },
  { key: "age", label: "Age", def: 35 },
  { key: "MonthlyIncome", label: "Monthly Income", def: 5000 },
  { key: "DebtRatio", label: "Debt Ratio", def: 0.3 },
  { key: "NumberOfOpenCreditLinesAndLoans", label: "Open Credit Lines", def: 6 },
  { key: "NumberOfTimes90DaysLate", label: "Times 90+ Days Late", def: 0 },
  { key: "NumberOfTime30to59DaysPastDueNotWorse", label: "Times 30-59 Days Late", def: 0 },
  { key: "NumberOfTime60to89DaysPastDueNotWorse", label: "Times 60-89 Days Late", def: 0 },
  { key: "NumberRealEstateLoansOrLines", label: "Real Estate Loans", def: 1 },
  { key: "NumberOfDependents", label: "Dependents", def: 0 },
];

const prettyNames = {
  RevolvingUtilizationOfUnsecuredLines: "Credit Utilization",
  TotalPastDue: "Total Past-Due History",
  "NumberOfTimes90DaysLate": "90+ Days Late",
  "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days Late",
  "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days Late",
  DebtRatio: "Debt Ratio",
  age: "Age",
  MonthlyIncome: "Monthly Income",
  IncomePerDependent: "Income per Dependent",
  NumberOfOpenCreditLinesAndLoans: "Open Credit Lines",
  NumberRealEstateLoansOrLines: "Real Estate Loans",
  NumberOfDependents: "Dependents",
};

function App() {
  const initial = {};
  fields.forEach((f) => (initial[f.key] = f.def));

  const [form, setForm] = useState(initial);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState("summary");

  const handleChange = (key, value) => {
    setForm({ ...form, [key]: value });
  };

  const assess = async () => {
    try {
      setLoading(true);
      const payload = {};
      fields.forEach((f) => (payload[f.key] = Number(form[f.key])));
      const res = await axios.post("http://127.0.0.1:8000/predict", payload);
      setResult(res.data);
      setTab("summary");
    } catch (err) {
      console.error(err);
      alert("Could not reach the API. Is the server running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const maxImpact = result?.top_factors?.[0]?.impact || 1;

  return (
    <div className="app">
      <div className="header">
        <h1>Loan Default Risk Assessment</h1>
        <p>Enter applicant details to assess credit default risk and understand why.</p>
      </div>

      <div className="layout">
        {/* LEFT: form */}
        <div className="card">
          <h2>Applicant Details</h2>
          {fields.map((f) => (
            <div className="field" key={f.key}>
              <label>{f.label}</label>
              <input
                type="number"
                value={form[f.key]}
                onChange={(e) => handleChange(f.key, e.target.value)}
              />
            </div>
          ))}
          <button className="btn" onClick={assess} disabled={loading}>
            {loading ? "Assessing..." : "Assess Risk"}
          </button>
        </div>

        {/* RIGHT: results */}
        <div className="card">
          <h2>Risk Assessment</h2>

          {!result ? (
            <div className="result-empty">
              Fill in the applicant details and click "Assess Risk" to see the result.
            </div>
          ) : (
            <>
              <div className="tabs">
                <button className={tab === "summary" ? "tab active" : "tab"} onClick={() => setTab("summary")}>Summary</button>
                <button className={tab === "explanation" ? "tab active" : "tab"} onClick={() => setTab("explanation")}>Explanation</button>
                <button className={tab === "model" ? "tab active" : "tab"} onClick={() => setTab("model")}>Model</button>
              </div>

              {tab === "summary" && (
                <div className={`band ${result.risk_band}`}>
                  <div className="label">{result.risk_band} Risk</div>
                  <div className="value">{Math.round(result.risk_probability * 100)}%</div>
                  <div className="prob">probability of default</div>
                </div>
              )}

              {tab === "explanation" && (
                <div className="factors">
                  <h3>Top factors driving this risk score</h3>
                  {result.top_factors.length === 0 ? (
                    <p className="result-empty">No strong risk factors — this profile looks low-risk.</p>
                  ) : (
                    result.top_factors.map((f) => (
                      <div className="factor" key={f.feature}>
                        <span className="name">{prettyNames[f.feature] || f.feature}</span>
                        <span className="bar" style={{ width: `${(f.impact / maxImpact) * 120 + 20}px` }}></span>
                      </div>
                    ))
                  )}
                  <p className="note">
                    Each bar shows how much that factor pushed this applicant's risk upward,
                    computed with SHAP values.
                  </p>
                </div>
              )}

              {tab === "model" && (
                <div className="model-info">
                  <p><strong>Model:</strong> Random Forest (100 trees)</p>
                  <p><strong>ROC-AUC:</strong> 0.845</p>
                  <p><strong>Training data:</strong> 150,000 applicants</p>
                  <p><strong>Handles:</strong> class imbalance (6.7% default rate), missing data, outliers</p>
                  <h3 style={{ marginTop: "16px" }}>Top overall risk factors</h3>
                  <div className="factor"><span className="name">Credit Utilization</span></div>
                  <div className="factor"><span className="name">Total Past-Due History</span></div>
                  <div className="factor"><span className="name">Debt Ratio</span></div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;