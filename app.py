import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image
import tensorflow as tf 
keras = tf.keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import time

model = VGG16() 

# create web design
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/zvTspPPN/Untitled-design.png "); 
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Create tabs menu
tab_titles = ['Image Classifier', 'About']
tab1, tab2 = st.tabs(tab_titles)
# Add content to each tab
with tab1:
    # load an image 
    st.markdown('<h4 style="color:#1642e0; font-family: Cascadia Code;">Insert image <br>for classification:</h1>', unsafe_allow_html=True)
    upload= st.file_uploader(' ', type=['png','jpg'])

    c1, c2= st.columns(2)
    if upload is not None:
      im = Image.open(upload)
      # convert the image pixels to a numpy array
      image = np.asarray(im)
      image = cv.resize(image,(224, 224))
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = preprocess_input(image)

      c1.markdown('<h4 style="color:#1642e0; font-family: Cascadia Code;">Your Image >> </h1>', unsafe_allow_html=True) 
      c1.image(im)
      c1.write(image.shape)

      # predict the probability across all output classes
      # also added the progress bar while model thinking
      progress_bar = st.progress(0)

      with st.spinner('Predicting...'):
        prediction = model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(prediction)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]

        c2.markdown('<h4 style="color:#1642e0; font-family: Cascadia Code;">My Output >> </h1>', unsafe_allow_html=True)
        c2.markdown('<h4 style="color:#bd2709; font-family: Cascadia Code;">Predicted class:</h1>', unsafe_allow_html=True) 
        c2.markdown(f'<h4 style="color:#ffffff; font-family: Cascadia Code;">{label[1]} ({label[2]*100:.2f}%)</h1>', unsafe_allow_html=True) 
        vgg_pred_classes = np.argmax(prediction, axis=1)

        st.success('Prediction complete!')
 
with tab2:
    st.header('About')
    background_image = "https://i.postimg.cc/d1QbbHMF/bck.png"
    html_code = f"""
    <style>
    .background-image {{
        background-image: url("{background_image}");
        background-size: cover;
        background-position: center;
        color: black;
        font-size: 16px;
        font-family: "Roboto";
        text-align: left;
        padding: 20px;
        margin-top: 10px; /* Adjust margin to position the background image */
        margin-bottom: 10px; /* Adjust margin to position the background image */
    }}
    </style>

    <div class="background-image">
        {"VGG16 is a convolutional neural network trained on a subset of the ImageNet dataset, a collection of over 14 million images belonging to 22,000 categories. K. Simonyan and A. Zisserman proposed this model in the 2015 paper, Very Deep Convolutional Networks for Large-Scale Image Recognition."}
        {"You also can check out this model in GitHub:"}
    </div>
    """
    # Display the background image with text using markdown
    st.markdown(html_code, unsafe_allow_html=True)
    st.write("https://github.com/poojatambe/VGG-Image-classification")