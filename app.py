import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")


try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "❌ house_data.csv not found. "
        "Please keep house_data.csv in the same folder as app.py."
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🏠 House Price Prediction System")

st.markdown(
    "### Machine Learning Powered House Price Analysis & Prediction"
)

st.write(
    "This dashboard analyzes housing data and predicts house prices "
    "using Linear Regression."
)


# ============================================================
# MODEL TRAINING
# ============================================================

data = data.dropna()

X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📊 Dataset Analysis",
        "📈 Visualizations",
        "🤖 Model Performance",
        "🔮 Price Prediction"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.header("🏠 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🏘️ Total Houses",
            f"{len(data):,}"
        )

    with col2:
        st.metric(
            "📊 Features",
            X.shape[1]
        )

    with col3:
        st.metric(
            "💰 Average Price",
            f"{y.mean():,.2f}"
        )

    with col4:
        st.metric(
            "🎯 R² Score",
            f"{r2:.2f}"
        )

    st.divider()

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        data.head(10),
        width="stretch"
    )

    st.divider()

    st.subheader("📌 Project Information")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "🤖 Machine Learning Model\n\n"
            "Linear Regression"
        )

    with col2:

        st.success(
            "📈 Prediction Target\n\n"
            "House Price"
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":

    st.header("📊 Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            f"{data.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            data.shape[1]
        )

    st.divider()

    st.subheader("🔍 Dataset Information")

    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str),
        "Missing Values": data.isnull().sum().values
    })

    st.dataframe(
        info_df,
        width="stretch"
    )

    st.divider()

    st.subheader("📈 Statistical Summary")

    st.dataframe(
        data.describe(),
        width="stretch"
    )

    st.divider()

    st.subheader("🏠 Complete Dataset")

    st.dataframe(
        data,
        width="stretch"
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":

    st.header("📈 House Price Visualizations")

    # --------------------------------------------------------
    # Area vs Price
    # --------------------------------------------------------

    if "Area" in data.columns:

        st.subheader("📐 Area vs House Price")

        fig1, ax1 = plt.subplots(
            figsize=(10, 5)
        )

        ax1.scatter(
            data["Area"],
            data["Price"]
        )

        ax1.set_xlabel(
            "Area"
        )

        ax1.set_ylabel(
            "House Price"
        )

        ax1.set_title(
            "Area vs House Price"
        )

        st.pyplot(fig1)

        plt.close(fig1)

    # --------------------------------------------------------
    # Actual vs Predicted
    # --------------------------------------------------------

    st.subheader("🎯 Actual vs Predicted Price")

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.scatter(
        y_test,
        y_pred
    )

    ax2.set_xlabel(
        "Actual House Price"
    )

    ax2.set_ylabel(
        "Predicted House Price"
    )

    ax2.set_title(
        "Actual vs Predicted House Price"
    )

    st.pyplot(fig2)

    plt.close(fig2)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Machine Learning Model Performance")

    st.write(
        "Model Used: **Linear Regression**"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"{mae:,.2f}"
        )

    with col2:

        st.metric(
            "R² Score",
            f"{r2:.2f}"
        )

    with col3:

        st.metric(
            "Training Samples",
            f"{len(X_train):,}"
        )

    st.divider()

    if r2 >= 0.80:

        st.success(
            "🟢 Excellent Model Performance"
        )

    elif r2 >= 0.60:

        st.info(
            "🟡 Good Model Performance"
        )

    else:

        st.warning(
            "🔴 Model Needs Improvement"
        )

    st.divider()

    st.subheader(
        "📋 Actual vs Predicted Values"
    )

    comparison = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": y_pred
    })

    st.dataframe(
        comparison.head(20),
        width="stretch"
    )


# ============================================================
# PRICE PREDICTION
# ============================================================

elif page == "🔮 Price Prediction":

    st.header("🔮 House Price Prediction")

    st.write(
        "Enter the property features below to predict the house price."
    )

    input_values = {}

    for column in X.columns:

        default_value = float(
            data[column].median()
        )

        input_values[column] = st.number_input(
            f"Enter {column}",
            value=default_value
        )

    st.divider()

    if st.button(
        "🏠 Predict House Price",
        width="stretch"
    ):

        input_df = pd.DataFrame(
            [input_values]
        )

        prediction = model.predict(
            input_df
        )[0]

        st.success(
            "✅ House Price Prediction Completed!"
        )

        st.metric(
            "🏠 Predicted House Price",
            f"{prediction:,.2f}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 House Price Prediction System | "
    "Python • Pandas • Scikit-learn • Streamlit"
)