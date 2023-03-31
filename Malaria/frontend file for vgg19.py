# Using streamlit framework for front end development
# Run this python file after giving the location of model in line 30
# Copy the link provided in the output paste it in command prompt 
# Add ' " ' in the starting and ending of location (e.g. -streamlit run "E:/malaria/frontend file for vgg19.py") and run the command
# It will automatically show a interface where one can test the model.
from PIL import Image,ImageOps
import keras_preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

def main():
#     We can add a picture to the interface by providing the location below
    image = Image.open('E:/abc.jpg')
    st.image(image, caption='malaria', use_column_width=True)
    st.title("Malaria Parasite detector application")
    file_uploaded=st.file_uploader("Upload a blood smear image",type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result=predict_class(image)
        st.write(result)
        st.pyplot(figure)
def predict_class(image):
    if (st.button('Predict')):
        classifier_model=tf.keras.models.load_model(r'E:\Malaria\malaria_prediction_model.h5')
        shape=((224,224,3))
        model=tf.keras.Sequential(hub.KerasLayer(classifier_model,input_shape=shape))
        test_image=image.resize((224,224))
        test_image=keras_preprocessing.image.img_to_array(test_image)
        test_image=test_image/255.0
        test_image=np.expand_dims(test_image,axis=0)
        class_names=['infected','uninfected']
        predictions=model.predict(test_image)
        scores=tf.nn.softmax(predictions[0])
        scores=scores.numpy()
        image_class=class_names[np.argmax(scores)]
        a=''
        if(image_class=='infected'):
            a='You are infected with malaria.Please consult your doctor as soon as possible.'
        else:
            a='Congrats!! You are not infected with malaria.'
        result=a
        return result

if __name__=='__main__':
    main()
