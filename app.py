import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Student ki details daalo aur predicted exam marks dekho.")

@st.cache_data
def load_data():
    df = pd.read_csv("StudentPerformanceFactors.csv")
    df = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Exam_Score"]]
    return df.dropna().drop_duplicates()

@st.cache_resource
def train_model(df):
    X = df[["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]]
    y = df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, r2_score(y_test, pred), mean_absolute_error(y_test, pred)

df = load_data()
model, r2, mae = train_model(df)

st.subheader("Input")
hours = st.slider("Hours Studied (per week)", 1, 44, 20)
attendance = st.slider("Attendance (%)", 60, 100, 85)
sleep = st.slider("Sleep Hours (per day)", 4, 10, 7)
previous = st.slider("Previous Scores", 50, 100, 75)

if st.button("Predict Marks"):
    new_student = pd.DataFrame([[hours, attendance, sleep, previous]],
                               columns=["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"])
    result = model.predict(new_student)[0]
    st.success(f"Predicted Exam Score: {result:.1f}")

st.subheader("Model Performance")
st.write(f"R2 Score: {r2:.3f}")
st.write(f"Mean Absolute Error: {mae:.2f}")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))
st.subheader("Study Hours vs Exam Score")
st.scatter_chart(df, x="Hours_Studied", y="Exam_Score")
