import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Financial Forecast Tool", layout="wide")

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Navigation")
st.sidebar.info("""
👨‍💻 Developed by: Krishang Kapoor  
📊 Project: Financial Forecast Tool  
🚀 Tech: Python, Streamlit, ML  
""")

page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Forecast", "Visualization"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("💰 Financial Forecast Tool")

    st.markdown("""
    ### 📌 About Project
    This web application predicts future revenue using Machine Learning models.

    ### ⚙️ Technologies:
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas & Matplotlib

    ### 🎯 Use Case:
    Helps businesses analyze trends and plan future strategies.
    """)

    if "data" in st.session_state:
        data = st.session_state["data"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Revenue", int(data["Revenue"].sum()))
        col2.metric("Average Revenue", int(data["Revenue"].mean()))
        col3.metric("Max Revenue", int(data["Revenue"].max()))

# ---------------- UPLOAD ----------------
elif page == "Upload Data":
    st.title("📂 Upload CSV File")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file:
        data = pd.read_csv(file)

        if "Month" not in data.columns or "Revenue" not in data.columns:
            st.error("❌ CSV must contain 'Month' and 'Revenue'")
        else:
            st.success("✅ Dataset uploaded successfully!")
            st.dataframe(data)
            st.session_state["data"] = data

# ---------------- FORECAST ----------------
elif page == "Forecast":
    st.title("📈 Revenue Forecast")

    if "data" not in st.session_state:
        st.warning("⚠️ Please upload data first")
    else:
        data = st.session_state["data"]

        X = data[['Month']]
        y = data['Revenue']

        model_type = st.selectbox("Select Model", ["Linear Regression", "Polynomial Regression"])

        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
        else:
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)

        future_input = st.text_input("Enter future months (comma separated)", "6,7,8")

        try:
            future_months = np.array([int(i) for i in future_input.split(",")]).reshape(-1, 1)

            if model_type == "Polynomial Regression":
                future_months = poly.transform(future_months)

            predictions = model.predict(future_months)

            result_df = pd.DataFrame({
                "Month": future_months.flatten(),
                "Predicted Revenue": predictions
            })

            st.subheader("📊 Predictions")
            st.dataframe(result_df)

            # Download CSV
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Predictions", csv, "predictions.csv", "text/csv")

            st.session_state["predictions"] = predictions

            # AI Explanation
            st.subheader("🤖 AI Explanation")

            trend = "increasing 📈" if predictions[-1] > predictions[0] else "decreasing 📉"

            st.info(f"""
            The model predicts a **{trend} trend** in revenue.
            This is based on historical data patterns.
            """)

        except:
            st.error("❌ Invalid input. Use format like: 6,7,8")

# ---------------- VISUALIZATION ----------------
elif page == "Visualization":
    st.title("📊 Visualization")

    if "data" not in st.session_state:
        st.warning("Upload data first!")
    else:
        data = st.session_state["data"]

        fig, ax = plt.subplots()

        ax.plot(data['Month'], data['Revenue'], marker='o', label='Actual')

        if "predictions" in st.session_state:
            preds = st.session_state["predictions"]
            future_x = range(len(data) + 1, len(data) + 1 + len(preds))
            ax.plot(future_x, preds, marker='x', linestyle='dashed', label='Forecast')

        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.set_title("Financial Forecast")
        ax.legend()

        st.pyplot(fig)

# ---------------- CHATBOT ----------------
st.markdown("## 🤖 AI Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask about your data...")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    response = "Please generate forecast first."

    if "predictions" in st.session_state:
        preds = st.session_state["predictions"]

        if "increase" in user_input.lower():
            response = "Revenue is increasing 📈 based on forecast."
        elif "decrease" in user_input.lower():
            response = "Revenue is decreasing 📉 based on forecast."
        elif "future" in user_input.lower():
            response = f"Future predictions are: {list(preds.round(2))}"
        elif "why" in user_input.lower():
            response = "Predictions are generated using machine learning models based on past data."
        else:
            response = "Ask about increase, decrease, future or explanation 😊"

    st.session_state.chat.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Krishang Kapoor | 🚀 BTech CSE Project")