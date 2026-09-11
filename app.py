import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")


try:
    data = load_data()

except FileNotFoundError:
    st.error("❌ house_data.csv not found!")
    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

data = data.dropna()


# =========================================================
# FEATURE AND TARGET
# =========================================================

if "Price" not in data.columns:
    st.error("❌ 'Price' column not found in dataset!")
    st.stop()

X = data.drop("Price", axis=1)
y = data["Price"]


# =========================================================
# CHECK NUMERIC FEATURES
# =========================================================

if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):

    st.error(
        "❌ All prediction features must be numeric. "
        "Please check your dataset."
    )

    st.stop()


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# LINEAR REGRESSION MODEL
# =========================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)


# =========================================================
# LINEAR REGRESSION EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


# =========================================================
# MODEL COMPARISON
# =========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


results = []

trained_models = {}


for name, ml_model in models.items():

    ml_model.fit(
        X_train,
        y_train
    )

    predictions = ml_model.predict(
        X_test
    )

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    model_r2 = r2_score(
        y_test,
        predictions
    )

    results.append({

        "Model": name,

        "MAE": model_mae,

        "R² Score": model_r2

    })

    trained_models[name] = ml_model


results_df = pd.DataFrame(
    results
)


# =========================================================
# BEST MODEL
# =========================================================

best_index = results_df[
    "R² Score"
].idxmax()


best_model_name = results_df.loc[
    best_index,
    "Model"
]


best_r2 = results_df.loc[
    best_index,
    "R² Score"
]


best_mae = results_df.loc[
    best_index,
    "MAE"
]


best_model = trained_models[
    best_model_name
]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🏠 House Price Prediction"
)

st.sidebar.markdown(
    "### Navigation"
)


page = st.sidebar.radio(

    "Select Page",

    [

        "Dashboard",

        "Dataset Analysis",

        "Visualizations",

        "Model Comparison",

        "Model Performance",

        "Price Prediction"

    ]
)


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🏠 House Price Prediction System"
)

st.markdown(
    "### Machine Learning based House Price Prediction Dashboard"
)

st.markdown("---")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.header(
        "📊 Dashboard"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🏠 Total Houses",
            len(data)
        )


    with col2:

        st.metric(
            "📌 Features",
            X.shape[1]
        )


    with col3:

        st.metric(
            "💰 Average Price",
            f"{y.mean():,.2f}"
        )


    with col4:

        st.metric(
            "📈 Best R² Score",
            f"{best_r2:.3f}"
        )


    st.markdown("---")


    st.subheader(
        "📋 Dataset Preview"
    )


    st.dataframe(
        data.head(10),
        width="stretch"
    )


    st.markdown("---")


    st.subheader(
        "🏆 Best Performing Model"
    )


    st.success(
        f"Best Model: {best_model_name}"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"📈 R² Score: {best_r2:.3f}"
        )


    with col2:

        st.info(
            f"📉 MAE: {best_mae:,.2f}"
        )


    st.markdown("---")


    st.subheader(
        "📌 Project Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            """
            **Machine Learning**

            Linear Regression

            Decision Tree

            Random Forest
            """
        )


    with col2:

        st.success(
            """
            **Prediction Target**

            House Price
            """
        )


# =========================================================
# DATASET ANALYSIS
# =========================================================

elif page == "Dataset Analysis":

    st.header(
        "📋 Dataset Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Rows",
            data.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            data.shape[1]
        )


    st.markdown("---")


    st.subheader(
        "🔍 Dataset"
    )


    st.dataframe(
        data,
        width="stretch"
    )


    st.markdown("---")


    st.subheader(
        "📊 Statistical Summary"
    )


    st.dataframe(
        data.describe(),
        width="stretch"
    )


    st.markdown("---")


    st.subheader(
        "❓ Missing Values"
    )


    missing_values = data.isnull().sum()


    st.dataframe(
        missing_values.rename(
            "Missing Values"
        )
    )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "Visualizations":

    st.header(
        "📈 Data Visualizations"
    )


    # ---------------------------------------------
    # AREA VS PRICE
    # ---------------------------------------------

    if "Area" in data.columns:

        st.subheader(
            "🏠 Area vs House Price"
        )


        fig1, ax1 = plt.subplots(
            figsize=(8, 5)
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


        st.pyplot(
            fig1
        )


    st.markdown("---")


    # ---------------------------------------------
    # ACTUAL VS PREDICTED
    # ---------------------------------------------

    st.subheader(
        "🎯 Actual vs Predicted Prices"
    )


    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
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


    st.pyplot(
        fig2
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header(
        "🏆 Model Comparison"
    )


    st.write(
        "Compare different Machine Learning "
        "models using MAE and R² Score."
    )


    st.markdown("---")


    st.subheader(
        "📊 Performance Comparison"
    )


    display_results = results_df.copy()


    display_results["MAE"] = display_results[
        "MAE"
    ].round(2)


    display_results["R² Score"] = display_results[
        "R² Score"
    ].round(3)


    st.dataframe(
        display_results,
        width="stretch"
    )


    st.markdown("---")


    st.subheader(
        "🏆 Best Performing Model"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🏆 Best Model",
            best_model_name
        )


    with col2:

        st.metric(
            "📈 Best R²",
            f"{best_r2:.3f}"
        )


    with col3:

        st.metric(
            "📉 Best MAE",
            f"{best_mae:,.2f}"
        )


    st.success(
        f"🏆 {best_model_name} performed best "
        f"based on R² Score."
    )


    st.markdown("---")


    # ---------------------------------------------
    # R2 SCORE CHART
    # ---------------------------------------------

    st.subheader(
        "📈 R² Score Comparison"
    )


    chart_data = results_df.set_index(
        "Model"
    )["R² Score"]


    st.bar_chart(
        chart_data
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        "🤖 Model Performance"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Mean Absolute Error",
            f"{mae:,.2f}"
        )


    with col2:

        st.metric(
            "R² Score",
            f"{r2:.3f}"
        )


    st.markdown("---")


    st.subheader(
        "📌 Linear Regression Performance"
    )


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


    st.markdown("---")


    st.subheader(
        "📊 Actual vs Predicted"
    )


    comparison = pd.DataFrame({

        "Actual Price":
            y_test.values,

        "Predicted Price":
            y_pred

    })


    st.dataframe(
        comparison.head(20),
        width="stretch"
    )


# =========================================================
# PRICE PREDICTION
# =========================================================

elif page == "Price Prediction":

    st.header(
        "🏠 Predict House Price"
    )


    st.write(
        f"Prediction is performed using the "
        f"best model: **{best_model_name}**"
    )


    st.markdown("---")


    input_data = {}


    col1, col2 = st.columns(2)


    for index, feature in enumerate(
        X.columns
    ):


        default_value = float(
            data[feature].mean()
        )


        if index % 2 == 0:

            with col1:

                input_data[feature] = st.number_input(

                    f"{feature}",

                    value=default_value

                )

        else:

            with col2:

                input_data[feature] = st.number_input(

                    f"{feature}",

                    value=default_value

                )


    st.markdown("---")


    if st.button(
        "🔮 Predict House Price",
        width="stretch"
    ):


        input_df = pd.DataFrame(
            [input_data]
        )


        prediction = best_model.predict(
            input_df
        )[0]


        st.success(
            f"🏠 Predicted House Price: "
            f"{prediction:,.2f}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.caption(
    "🏠 House Price Prediction System | "
    "Python • Pandas • Scikit-learn • Streamlit"
)