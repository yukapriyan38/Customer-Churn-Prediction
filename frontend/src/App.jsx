
import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

const initialCustomer = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.7,
  TotalCharges: 848.4,
};

const options = {
  gender: ["Female", "Male"],
  SeniorCitizen: [0, 1],
  Partner: ["Yes", "No"],
  Dependents: ["Yes", "No"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["No phone service", "No", "Yes"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["No internet service", "No", "Yes"],
  OnlineBackup: ["No internet service", "No", "Yes"],
  DeviceProtection: ["No internet service", "No", "Yes"],
  TechSupport: ["No internet service", "No", "Yes"],
  StreamingTV: ["No internet service", "No", "Yes"],
  StreamingMovies: ["No internet service", "No", "Yes"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ],
};

function App() {
  const [customer, setCustomer] = useState(initialCustomer);
  const [overview, setOverview] = useState(null);
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function loadDashboard() {
      try {
        const overviewResponse = await fetch(
          `${API_URL}/api/overview`
        );

        if (!overviewResponse.ok) {
          throw new Error("Could not load dashboard data.");
        }

        const overviewData = await overviewResponse.json();
        setOverview(overviewData);

        const modelResponse = await fetch(
          `${API_URL}/api/models`
        );

        if (modelResponse.ok) {
          const modelData = await modelResponse.json();
          setModels(modelData.models || []);
        }
      } catch (err) {
        setError(
          "Cannot connect to the backend. Make sure FastAPI is running."
        );
      }
    }

    loadDashboard();
  }, []);

  function handleChange(event) {
    const { name, value, type } = event.target;

    setCustomer({
      ...customer,
      [name]: type === "number" ? Number(value) : value,
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(customer),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function renderField(name) {
    const label = name.replace(/([A-Z])/g, " $1");

    return (
      <div className="form-group" key={name}>
        <label htmlFor={name}>{label}</label>

        {options[name] ? (
          <select
            id={name}
            name={name}
            value={customer[name]}
            onChange={handleChange}
          >
            {options[name].map((option) => (
              <option key={option} value={option}>
                {name === "SeniorCitizen"
                  ? option === 1
                    ? "Yes"
                    : "No"
                  : option}
              </option>
            ))}
          </select>
        ) : (
          <input
            id={name}
            name={name}
            type="number"
            min="0"
            max={name === "tenure" ? "100" : undefined}
            step={name === "tenure" ? "1" : "0.01"}
            value={customer[name]}
            onChange={handleChange}
            required
          />
        )}
      </div>
    );
  }

  return (
    <div className="app">
      <nav className="navbar">
        <h2>Churn<span>Predict</span></h2>

        <div className="nav-links">
          <a href="#dashboard">Dashboard</a>
          <a href="#prediction">Prediction</a>
          <a href="#models">Models</a>
        </div>
      </nav>

      <main className="main-content">
        <section className="welcome" id="dashboard">
          <p className="subtitle">CUSTOMER ANALYTICS</p>
          <h1>Customer Churn Dashboard</h1>
          <p>
            Understand your customers and predict who may leave.
          </p>
        </section>

        {error && <div className="error-message">{error}</div>}

        <section className="stats-grid">
          <div className="stat-card">
            <p>Total Customers</p>
            <h2>
              {overview
                ? overview.total_customers.toLocaleString()
                : "..."}
            </h2>
          </div>

          <div className="stat-card">
            <p>Churned Customers</p>
            <h2>
              {overview
                ? overview.churned_customers.toLocaleString()
                : "..."}
            </h2>
          </div>

          <div className="stat-card">
            <p>Churn Rate</p>
            <h2>
              {overview ? `${overview.churn_rate}%` : "..."}
            </h2>
          </div>
        </section>

        <section className="panel" id="prediction">
          <div className="section-heading">
            <h2>Predict Customer Churn</h2>
            <p>Enter the customer's information below.</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {Object.keys(initialCustomer).map(renderField)}
            </div>

            <button type="submit" disabled={loading}>
              {loading ? "Predicting..." : "Predict Churn"}
            </button>
          </form>

          {result && (
            <div
              className={`result-card ${
                result.class === 1 ? "churn" : "stay"
              }`}
            >
              <p className="subtitle">PREDICTION RESULT</p>
              <h2>{result.prediction}</h2>
              <p>
                Estimated churn probability:{" "}
                <strong>{result.churn_probability}%</strong>
              </p>

              <div className="progress-bar">
                <div
                  style={{
                    width: `${Math.min(
                      100,
                      Math.max(0, result.churn_probability)
                    )}%`,
                  }}
                />
              </div>
            </div>
          )}
        </section>

        <section className="panel" id="models">
          <div className="section-heading">
            <h2>Model Comparison</h2>
            <p>Results from your existing model comparison file.</p>
          </div>

          {models.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    {Object.keys(models[0]).map((key) => (
                      <th key={key}>{key.replace(/_/g, " ")}</th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {models.map((model, index) => (
                    <tr key={index}>
                      {Object.values(model).map((value, i) => (
                        <td key={i}>
                          {value === null ? "-" : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">
              {error
                ? "Model comparison is not available."
                : "Loading model comparison..."}
            </p>
          )}
        </section>

        <footer>
          Customer Churn Prediction Project · React + FastAPI
        </footer>
      </main>
    </div>
  );
}

export default App;
