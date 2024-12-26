import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import platform
import pathlib
import math

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Convolution Neural Network(CNN) yordamida rasmlarni klaslarga(classification) ajratish")

# rasmni joylash (input)
file = st.file_uploader("Rasmni Yuklang", type=['jpeg', 'png', 'svg', 'gif'])
if file:
  st.image(file)
  img = PILImage.create(file)
  # modelni yuklash
  model = load_learner("cnn-classification-model.pkl")
  # prediction (bashorat)
  pred, pred_id, probs = model.predict(img)
  if int(math.sqrt(probs[pred_id])) > 90:
    st.success(f"Topdim! Bu:  {pred}")
    st.info(f"<{pred}> bo'lish Ehtimolligi: {probs[pred_id]*100}%")
    
    # plotting (grafik)
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
  else:
    st.info(f"Siz boshqa rasm kiritdingiz!\n Iltimos, bulardan birini kiritganingizga ishonch hosil qiling: Worm(Chuvalchang), Dragonfly(Ninachi), Caterpillar(Qurt), Butterfly(Kapalak)")
  
