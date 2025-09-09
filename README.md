# 🎓 Student Career Willingness Predictor

A Streamlit web application that predicts a student's **willingness to pursue a career based on their degree** using advanced machine learning techniques. 

Built with ❤️ using **Streamlit, Scikit-learn, and Plotly**.

---

## ℹ️ About

This application leverages **machine learning models** (Random Forest, Gradient Boosting, SVM, Logistic Regression) to help educators and career counselors:

- Identify students who may need additional support.
- Analyze key factors influencing student career decisions.
- Provide data-driven career guidance.

---

## 🎯 Purpose

To support **career counseling and education planning** by predicting career willingness from student data such as:

- Academic performance (10th, 12th, and college marks)
- Engagement (certification courses, degree satisfaction)
- Study habits (daily study time)
- Life balance (travel time, stress levels, work commitments)

---

## 📊 Features

- **Prediction**: Estimate career willingness percentages.
- **About Page**: Learn more about the methodology.

**Target Variable:**
- `willingness to pursue a career based on their degree`: percentage likelihood (with `%` sign).

---

## 🧠 Model Information

- **Algorithms**: Random Forest, Gradient Boosting, SVM, Logistic Regression  
- **Model Selection**: Best-performing algorithm chosen automatically  
- **Hyperparameter Optimization**: Grid Search  
- **Cross-validation**: 5-fold for robust results  
- **Features Used**: 9 key student characteristics  
- **Output**: Likelihood of career pursuit (%)  

---

## 🖥️ Installation & Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

1. **Clone the repository**
   ```bash
   git clone https://github.com/rupinmunjal/student-career-willingness-predictor.git
   cd student-career-willingness-predictor
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```
   This will create a virtual environment (`.venv/`) and install all dependencies from `pyproject.toml`.

---

## 🚀 Running the Application

Start the Streamlit app with:
```bash
uv run streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
student-career-willingness/
├── app.py                          # Main Streamlit app
├── pyproject.toml                  # Dependencies & project metadata
├── uv.lock                         # Dependency lock file
├── data/                           # Dataset(s)
│   └── Student Attitude and Behavior.csv
├── models/                         # Trained ML models
│   └── student_career_willingness_model.pkl
├── src/                            # Source code
│   ├── __init__.py                     # Marks src/ as a Python package
│   └── student_career_willingness_predictor.py
├── README.md                       # Project documentation
```

---

## 📈 Interpretation Guide

- **75–100%**: High willingness → Motivated & engaged
- **50–74%**: Moderate willingness → May benefit from career guidance
- **25–49%**: Low willingness → Likely needs significant support

---

## 🤝 Contributing

Pull requests are welcome! If you’d like to add new features (e.g., additional ML models, dashboards, or data sources), feel free to fork and submit a PR.