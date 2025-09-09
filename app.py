import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Import the separate logic modules
from src import StudentCareerWillingnessPredictor


# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize predictor with default dataset"""
    predictor = StudentCareerWillingnessPredictor()

    model_path = 'models/student_career_willingness_model.pkl'
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load pre-trained model: {e}")
            predictor.train("data/Student Attitude and Behavior.csv")
            predictor.save_model(model_path)
    else:
        predictor.train("data/Student Attitude and Behavior.csv")
        predictor.save_model(model_path)

    return predictor


def main():
    st.set_page_config(
        page_title="Student Career Willingness Predictor",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎓 Student Career Willingness Predictor")
    st.markdown("---")

    # Initialize predictor
    predictor = initialize_components()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Prediction", "About"])

    # Route pages
    if page == "Prediction":
        prediction_page(predictor)
    else:
        about_page()


def prediction_page(predictor):
    """Main prediction interface"""
    st.header("🔮 Career Willingness Prediction")

    # Check if model is available
    if predictor.model is None:
        st.error("⚠️ No trained model available. Please check default dataset.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Student Information")

        # Academic Information
        with st.expander("📚 Academic Performance", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                tenth_mark = st.slider("10th Grade Marks (%)", 0.0, 100.0, 75.0)
            with col_b:
                twelfth_mark = st.slider("12th Grade Marks (%)", 0.0, 100.0, 70.0)
            with col_c:
                college_mark = st.slider("College Marks (%)", 0.0, 100.0, 80.0)

        # Engagement Factors
        with st.expander("💡 Engagement & Interest", expanded=True):
            col_d, col_e = st.columns(2)
            with col_d:
                cert_course = st.selectbox("Certification Course", ["No", "Yes"])
            with col_e:
                like_degree = st.selectbox("Do you like your degree?", ["No", "Yes"])

        # Study Habits
        with st.expander("📖 Study Habits", expanded=True):
            study_time = st.selectbox(
                "Daily Study Time",
                ["0-30 minutes", "30-60 minutes", "1-2 hours", "2-3 hours", "3-4 hours", "More than 4 hours"]
            )

        # Life Balance
        with st.expander("⚖️ Life Balance", expanded=True):
            col_f, col_g, col_h = st.columns(3)
            with col_f:
                travel_time = st.selectbox(
                    "Daily Travel Time",
                    ["0-30 mins", "30-60 mins", "1-1.5 hrs", "1.5-2 hrs", "2-2.5 hrs", "2.5-3 hrs", "3+ hrs"]
                )
            with col_g:
                stress_level = st.selectbox("Stress Level", ["Bad", "Awful", "Good", "Fabulous"])
            with col_h:
                part_time = st.selectbox("Part-time Job", ["No", "Yes"])

    with col2:
        st.subheader("📊 Prediction Results")

        # Transform inputs
        study_time_mapping = {
            "0-30 minutes": 0.0, "30-60 minutes": 1.0, "1-2 hours": 2.0,
            "2-3 hours": 3.0, "3-4 hours": 4.0, "More than 4 hours": 5.0
        }

        travel_mapping = {
            "0-30 mins": 0.0, "30-60 mins": 1.0, "1-1.5 hrs": 2.0,
            "1.5-2 hrs": 3.0, "2-2.5 hrs": 4.0, "2.5-3 hrs": 5.0, "3+ hrs": 6.0
        }

        stress_mapping = {"Bad": 0.0, "Awful": 1.0, "Good": 2.0, "Fabulous": 3.0}

        input_data = {
            'Certification Course': 1.0 if cert_course == "Yes" else 0.0,
            '10th Mark': float(tenth_mark),
            '12th Mark': float(twelfth_mark),
            'college mark': float(college_mark),
            'daily studing time': study_time_mapping[study_time],
            'Do you like your degree?': 1.0 if like_degree == "Yes" else 0.0,
            'Travelling Time ': travel_mapping[travel_time],
            'Stress Level ': stress_mapping[stress_level],
            'part-time job': 1.0 if part_time == "Yes" else 0.0
        }

        # Make prediction
        if st.button("🎯 Predict Career Willingness", type="primary"):
            with st.spinner("Analyzing student profile..."):
                try:
                    prediction = predictor.predict(input_data)

                    # Display prediction with gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Career Willingness (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width='stretch')

                    # Interpretation
                    if prediction >= 75:
                        st.success(f"🌟 High Career Willingness ({prediction:.1f}%)")
                    elif prediction >= 50:
                        st.warning(f"⚖️ Moderate Career Willingness ({prediction:.1f}%)")
                    else:
                        st.error(f"⚠️ Low Career Willingness ({prediction:.1f}%)")

                except Exception as e:
                    st.error(f"Error making prediction: {e}")

        # Show input summary
        st.subheader("📋 Input Summary")
        summary_df = pd.DataFrame([input_data]).T
        summary_df.columns = ['Value']
        st.dataframe(summary_df, width='stretch')

        # Show feature importance if available
        # if predictor.model is not None:
        #     importance_df = predictor.get_feature_importance()
        #     if importance_df is not None:
        #         st.subheader("🔍 Feature Importance")
        #         fig = px.bar(importance_df.head(5),
        #                      x='importance', y='feature',
        #                      orientation='h',
        #                      title="Top 5 Most Important Features")
        #         fig.update_layout(height=300)
        #         st.plotly_chart(fig, width='stretch')

def about_page():
    """About page with application information"""
    st.header("ℹ️ About This Application")

    st.markdown("""
    This application predicts a student's willingness to pursue a career based on their degree using advanced machine learning techniques.

    ### 🎯 Purpose
    The goal is to help educators and career counselors identify students who may need additional support or guidance
    in their career development journey.

    ### 📊 Features Used
    The model considers multiple factors:

    - **Academic Performance**: 10th, 12th grade, and college marks
    - **Engagement**: Certification courses and degree satisfaction
    - **Study Habits**: Daily study time allocation
    - **Life Balance**: Travel time, stress levels, and work commitments

    ### 🧠 Model Information
    - **Algorithms**: Ensemble of Random Forest, Gradient Boosting, SVM, and Logistic Regression
    - **Model Selection**: Automatically selects the best performing algorithm
    - **Hyperparameter Optimization**: Grid search for optimal parameters
    - **Cross-validation**: 5-fold cross-validation for robust evaluation
    - **Features**: 9 key student characteristics
    - **Output**: Percentage likelihood of career pursuit

    ### 📈 Interpretation Guide
    - **75-100%**: High willingness - student is motivated and engaged
    - **50-74%**: Moderate willingness - may benefit from career guidance
    - **25-49%**: Low willingness - likely needs significant support

    ### 🚀 How to Use
    1. **Prediction**: Go to the 'Prediction' page to input student details and get career willingness predictions.
    2. **Analysis**: Explore the 'Data Analysis' page to view insights, feature impacts, and model performance.
    3. **About**: Learn more about the application and its purpose on the 'About' page.

    **Target Variable:**
    - `willingness to pursue a career based on their degree  `: Target percentage (with % sign)
    """)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Scikit-learn, and Plotly")


if __name__ == "__main__":
    main()
