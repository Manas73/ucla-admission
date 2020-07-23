import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier


def user_input_features():

    gre = st.sidebar.slider("GRE Score", min_value=260, max_value=340, step=1)
    toefl = st.sidebar.slider(
        "TOEFL Score", min_value=0, max_value=120, step=1
    )
    univ = st.sidebar.slider(
        "University Rating", min_value=1, max_value=5, step=1
    )
    sop = st.sidebar.slider(
        "SOP Rating", min_value=1.0, max_value=5.0, step=0.5
    )
    lor = st.sidebar.slider(
        "LOR Rating", min_value=1.0, max_value=5.0, step=0.5
    )
    cgpa = st.sidebar.slider("CGPA", min_value=1.0, max_value=4.0, step=0.01)
    research_button = st.sidebar.radio(
        "Contributed to Research?", ("Yes", "No")
    )
    if research_button == "Yes":
        research = 1
    else:
        research = 0

    data = {
        "GRE Score": gre,
        "TOEFL Score": toefl,
        "University Rating": univ,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research,
    }
    features = pd.DataFrame(data, index=[0])
    return features


@st.cache()
def get_feature_importance(model):
    features = [
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
    ]
    df_features = pd.DataFrame()
    df_features["Features"] = features
    df_features["Importance"] = model.feature_importances_
    fig_features = go.Figure(
        [
            go.Pie(
                labels=df_features["Features"],
                values=df_features["Importance"],
            )
        ]
    )
    fig_features.update_layout(
        title={
            "text": "Feature Importance",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )
    return fig_features


def main():
    """Graduate Admission into UCLA Analysis"""

    st.title("Chances of getting into UCLA :mortar_board:")
    st.sidebar.markdown("**Developed by -** Manas Sambare")
    st.sidebar.markdown(
        "**[GitHub Repo](https://github.com/Manas73/ucla-admission) - "
        "[Kaggle Project](https://www.kaggle.com/msambare/analysis-of-graduate-admissions) - "
        "[LinkedIn](https://www.linkedin.com/in/manas-sambare)**"
    )
    st.sidebar.header("Provide your Academic Details")

    st.markdown(
        """
    ## **Context** \n
    Used the parameters: GRE Scores, TOEFL Scores, University Rating, \
    Statement of Purpose and Letter of Recommendation Strength, Undergraduate GPA, \
    and Research Experience to predict the Chance of Admit into UCLA.
    """
    )

    input_df = user_input_features()
    with st.spinner("Wait for it..."):
        st.subheader("User Input features")
        st.write(input_df)
        # Reads in saved classification model
        model = pickle.load(open("model.pkl", "rb"))

        # Apply model to make predictions
        prediction = model.predict(input_df)

        labels = ["Yes", "No"]
        values = [prediction[0], 1 - prediction[0]]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig.update_layout(
            title={
                "text": "Probability of getting into UCLA",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        if prediction > 0.9:
            st.balloons()

    st.markdown(
        """
    ## **Inspiration** \n
    The dataset was built with the purpose of helping students in shortlisting universities with their profiles. \
    The predicted output gives them a fair idea about their chances for a particular university.

    ## **Methodology** \n
    Evaluated and compared: Linear Regression, K-nearest Neighbors, Support-Vector Machine, and Random Forest models.

    ## **Results** \n
    The Random Forest model could best predict the Chance of Admit by reducing the error below 0.05. \
    It was found that the Undergraduate GPA is the most important feature (80%). \
    However, it cannot be relied upon due to the existence of multicollinearity.
    """
    )

    fig_features = get_feature_importance(model)
    st.plotly_chart(fig_features)

    st.markdown(
        """
    ## **Acknowledgement** \n
    The dataset used is inspired by the UCLA Graduate Dataset and is created by Mohan S Acharya to estimate chances of graduate admission from an Indian perspective.
    """
    )


if __name__ == "__main__":
    main()
