import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings

warnings.filterwarnings('ignore')


class StudentCareerWillingnessPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_name = None

    def load_and_preprocess_data(self, file_path='data/Student Attitude and Behavior.csv'):
        """Load and preprocess the student data"""
        df = pd.read_csv(file_path)

        # Drop unnecessary columns
        drop_columns = ['Gender', 'Department', 'Height(CM)', 'Weight(KG)',
                        'hobbies', 'salary expectation', 'social medai & video',
                        'Financial Status', 'prefer to study in']
        df = df.drop(drop_columns, axis=1)
        df = df.dropna(axis=0)

        # Encode categorical variables
        # Yes/No columns
        binary_cols = ['Certification Course', 'Do you like your degree?', 'part-time job']
        for col in binary_cols:
            df[col] = df[col].map({'No': 0, 'Yes': 1})

        # Daily studying time
        study_time_mapping = {
            '0 - 30 minute': 0, '30 - 60 minute': 1, '1 - 2 Hour': 2,
            '2 - 3 hour': 3, '3 - 4 hour': 4, 'More Than 4 hour': 5
        }
        df['daily studing time'] = df['daily studing time'].map(study_time_mapping)

        # Travelling time
        travel_mapping = {
            '0 - 30 minutes': 0, '30 - 60 minutes': 1, '1 - 1.30 hour': 2,
            '1.30 - 2 hour': 3, '2 - 2.30 hour': 4, '2.30 - 3 hour': 5,
            'more than 3 hour': 6
        }
        df['Travelling Time '] = df['Travelling Time '].map(travel_mapping)

        # Stress level
        stress_mapping = {'Bad': 0, 'Awful': 1, 'Good': 2, 'fabulous': 3}
        df['Stress Level '] = df['Stress Level '].map(stress_mapping)

        # Clean target variable
        df['willingness to pursue a career based on their degree  '] = (
            df['willingness to pursue a career based on their degree  ']
            .str.replace('%', '').astype(float)
        )

        return df

    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        best_score = 0
        best_model = None
        best_name = None
        results = {}

        for name, model in models.items():
            # Use scaled data for SVM and Logistic Regression
            if name in ['SVM', 'Logistic Regression']:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test

            # Train model
            model.fit(X_tr, y_train)

            # Cross-validation score
            cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

            # Test score
            test_score = model.score(X_te, y_test)

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': test_score,
                'model': model
            }

            # Update best model
            if test_score > best_score:
                best_score = test_score
                best_model = model
                best_name = name

        return best_model, best_name, results

    def optimize_best_model(self, X_train, y_train, model_name):
        """Optimize hyperparameters for the best model"""

        if model_name == 'Random Forest':
            X_tr = X_train
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42)

        elif model_name == 'Gradient Boosting':
            X_tr = X_train
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            }
            base_model = GradientBoostingClassifier(random_state=42)

        elif model_name == 'SVM':
            X_tr = self.scaler.transform(X_train)
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
            base_model = SVC(kernel='rbf', random_state=42)

        elif model_name == 'Logistic Regression':
            X_tr = self.scaler.transform(X_train)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')

        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_tr, y_train)

        return grid_search.best_estimator_

    def train(self, file_path='data/Student Attitude and Behavior.csv'):
        """Complete training pipeline"""
        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)

        # Prepare features and target
        X = df.drop('willingness to pursue a career based on their degree  ', axis=1)
        y = df['willingness to pursue a career based on their degree  '].values

        self.feature_columns = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train multiple models
        best_model, best_name, results = self.train_multiple_models(X_train, y_train, X_test, y_test)

        # Optimize the best model
        optimized_model = self.optimize_best_model(X_train, y_train, best_name)

        self.model = optimized_model

        return self.model

    def predict(self, input_data):
        """Make prediction for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Convert to DataFrame if needed
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = pd.DataFrame(input_data)

        # Ensure correct column order
        df = df[self.feature_columns]

        # Scale if needed
        if self.model_name in ['SVM', 'Logistic Regression']:
            df_scaled = self.scaler.transform(df)
            return self.model.predict(df_scaled)[0]
        else:
            return self.model.predict(df)[0]

    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None

    def save_model(self, filepath='student_career_willingness_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_name': self.model_name
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath='student_career_willingness_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_name = model_data['model_name']
