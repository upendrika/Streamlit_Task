import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model() #execute

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

#create stremlit app

def show_predict_page():
    st.markdown(
            """
            <style>
            .video-container {
                position: relative;
                padding-bottom: 56.25%;
                height: 0;
                overflow: hidden;
                max-width: 100%;
                background: #000;
            }
            .video-container iframe {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            </style>
            <div class="video-container">
                <iframe src="https://www.youtube.com/embed/VIDEO_ID" frameborder="0" allowfullscreen></iframe>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    #st.video("Software Developers Video.mp4")
    st.title("Software Developer Salary Prediction")
    
    st.sidebar.header(" We need some information to predict the salary")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.sidebar.selectbox("Country", countries)
    education = st.sidebar.selectbox("Education Level", education)

    expericence = st.sidebar.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
