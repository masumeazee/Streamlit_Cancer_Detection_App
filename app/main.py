import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_data():
    data = pd.read_csv('data/cancer_data.csv')
    print(data.head())
    return data


def add_sidebar():
    st.sidebar.header("Cell nuclei Measurements")

    data = get_data()

    slider_labels = [
        ("Radius (mean)", "mean radius"),
        ("Texture (mean)", "mean texture"),
        ("Perimeter (mean)", "mean perimeter"),
        ("Area (mean)", "mean area"),
        ("Smoothness (mean)", "mean smoothness"),
        ("Compactness (mean)", "mean compactness"),
        ("Concavity (mean)", "mean concavity"),
        ("Concave points (mean)", "mean concave points"),
        ("Symmetry (mean)", "mean symmetry"),
        ("Fractal dimension (mean)", "mean fractal dimension"),
        ("Radius (se)", "radius error"),
        ("Texture (se)", "texture error"),
        ("Perimeter (se)", "perimeter error"),
        ("Area (se)", "area error"),
        ("Smoothness (se)", "smoothness error"),
        ("Compactness (se)", "compactness error"),
        ("Concavity (se)", "concavity error"),
        ("Concave points (se)", "concave points error"),
        ("Symmetry (se)", "symmetry error"),
        ("Fractal dimension (se)", "fractal dimension error"),
        ("Radius (worst)", "worst radius"),
        ("Texture (worst)", "worst texture"),
        ("Perimeter (worst)", "worst perimeter"),
        ("Area (worst)", "worst area"),
        ("Smoothness (worst)", "worst smoothness"),
        ("Compactness (worst)", "worst compactness"),
        ("Concavity (worst)", "worst concavity"),
        ("Concave points (worst)", "worst concave points"),
        ("Symmetry (worst)", "worst symmetry"),
        ("Fractal dimension (worst)", "worst fractal dimension"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return(input_dict)


def get_scaled_values(input_dict):
    data = get_data()

    X = data.drop(['target'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Primeter', 'Area',
                  'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['mean radius'], input_data['mean texture'], input_data['mean perimeter'],
           input_data['mean area'], input_data['mean smoothness'], input_data['mean compactness'],
           input_data['mean concavity'], input_data['mean concave points'], input_data['mean symmetry'],
           input_data['mean fractal dimension'],
           ],
        theta=categories,
        fill='toself',
        name='mean value'

    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius error'], input_data['texture error'], input_data['perimeter error'],
           input_data['area error'], input_data['smoothness error'], input_data['compactness error'],
           input_data['concavity error'], input_data['concave points error'], input_data['symmetry error'],
           input_data['fractal dimension error'],
           ],
        theta=categories,
        fill='toself',
        name='standard error'

    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['worst radius'], input_data['worst texture'], input_data['worst perimeter'],
           input_data['worst area'], input_data['worst smoothness'], input_data['worst compactness'],
           input_data['worst concavity'], input_data['worst concave points'], input_data['worst symmetry'],
           input_data['worst fractal dimension'],
           ],
        theta=categories,
        fill='toself',
        name='worst value'

    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )

    return fig


def add_predictions(input_data):
    # import the model and scaler as pickle file
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write('-----------------------------')
    st.write('Has been detected as : ')
    if prediction[0] == 0:
        st.write("<span calss ='diagnosis benign'>Benign</span>",
                 unsafe_allow_html=True)
        # st.write("Benign")
    else:
        # st.write("Malicious")
        st.write("<span calss ='diagnosis malicious'>malicious</span>",
                 unsafe_allow_html=True)
    # st.write(prediction)

    # Make it little more user friendly
    st.write("The probability of being benign:",
             model.predict_proba(input_array_scaled)[0][0])
    st.write("The probability of being malicious:",
             model.predict_proba(input_array_scaled)[0][1])


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor ",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()),
                    unsafe_allow_html=True)

    input_data = add_sidebar()
    # st.write(input_data)

    with st.container():
        st.title("Breast Cancer Predector")  # same like h1 or h2 in html code
        # same like p or paragraph in Html code
        st.write("please connect this app to your cytology app")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
