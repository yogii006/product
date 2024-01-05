import streamlit as st  # to host system on web.
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests
import streamlit as st

import streamlit as st

# Custom CSS to set the background color
custom_css = """
    <style>
        body {
            background-color: #7a26a5; /* Replace with your preferred color */
            color: #ffffff
        }
        .stApp {
            background-color: transparent; /* Set the background of the app to transparent */
        }
        .css-10trblm {
            color: #ffffff
        }
        p {
             color: #ffffff;
        }
    </style>
"""

# # Display the custom CSS using st.markdown
st.markdown(custom_css, unsafe_allow_html=True)
# 
# # Now you can add other Streamlit elements
# st.title("My Streamlit App")
# st.write("This is some content.")






r = requests.get(url="http://127.0.0.1:8000/getData")
data = r.json()

print(data['history'])
inputImage = data['history'][0]
print(inputImage)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')


# to upload the file - this code is required
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


# calling the feature extraction function for the uploaded image.
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# steps
# file upload -> save
uploaded_file = True

# if st.button('Recommend product'):
#     print(uploaded_file)
st.header("Customer Information")
if uploaded_file is not None:

    # if save_uploaded_file(uploaded_file):
    # display the file
    b1, b2, g1, g2 = st.columns(4)
    with b1:
        st.write("Name")
    with b2:
        st.write(data["name"])

    b3, b4, g3, g4 = st.columns(4)
    with b3:
        st.write("Contact")
    with b4:
        st.write(data["contact"])

    b5, b6, g5, g6 = st.columns(4)
    with b5:
        st.write("Email")
    with b6:
        st.write(data["email"])

    # display_image = Image.open(os.path.join("uploads", inputImage))
    st.header("Input items")
    col6, col7, col8 = st.columns(3)
    with col6:
        st.image(os.path.join("uploads", data["history"][0]))
    with col7:
        st.image(os.path.join("uploads", data["history"][1]))
    with col8:
        st.image(os.path.join("uploads", data["history"][2]))
    # feature extract

    features1 = feature_extraction(os.path.join("uploads", data["history"][0]), model)
    features2 = feature_extraction(os.path.join("uploads", data["history"][1]), model)
    features3 = feature_extraction(os.path.join("uploads", data["history"][2]), model)
    # st.text(features)
    # recommendention
    indices1 = recommend(features1, feature_list)
    indices2 = recommend(features2, feature_list)
    indices3 = recommend(features3, feature_list)
    # show
    st.header("Recommended Products")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(filenames[indices1[0][0]])
    with col2:
        st.image(filenames[indices1[0][1]])
    with col3:
        st.image(filenames[indices1[0][2]])
    with col4:
        st.image(filenames[indices1[0][3]])
    with col5:
        st.image(filenames[indices1[0][4]])

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.image(filenames[indices2[0][0]])
    with c7:
        st.image(filenames[indices2[0][1]])
    with c8:
        st.image(filenames[indices2[0][2]])
    with c9:
        st.image(filenames[indices2[0][3]])
    with c10:
        st.image(filenames[indices2[0][4]])
    c11, c12, c13, c14, c15 = st.columns(5)
    with c11:
        st.image(filenames[indices3[0][0]])
    with c12:
        st.image(filenames[indices3[0][1]])
    with c13:
        st.image(filenames[indices3[0][2]])
    with c14:
        st.image(filenames[indices3[0][3]])
    with c15:
        st.image(filenames[indices3[0][4]])
        # else:

        #     st.header("Some error occurred in file upload")
