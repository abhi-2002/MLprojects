
from PIL import Image,ImageOps
import keras_preprocessing.image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

def main():
    image = Image.open('E:/abc.jpg')
    """ Path of 'Pythonfileforcnnalgo.h5'"""
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
        classifier_model=tf.keras.models.load_model(r'E:\malaria predictor\malaria_omdena10epoch.h5')
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
