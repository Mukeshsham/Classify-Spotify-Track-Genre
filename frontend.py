import streamlit as st
import pickle
from PIL import Image
import requests

url_link="http://127.0.0.1:8000/predict"

with open("xgboost_genre_classifier.pkl","rb") as f:
    model=pickle.load(f)
    
with open("scalar.pkl","rb") as f:
    scalar=pickle.load(f) 



page=st.sidebar.radio("Go to",["Home","Audio Genre Prediction"])
if page=="Home":
    st.header("🎧Classify Audio Genre-Spotify")
    image=Image.open("C:\\Users\\Dell\\Downloads\\image.png")
    st.image(image,use_container_width=True)
elif page=="Audio Genre Prediction":
    st.header("🎧Real Time Audio Genre Prediction")
    st.subheader("Tune the values below to predict audio genres")
    
    #input features
    popularity=st.number_input("Popularity",min_value=0.0,max_value=99.0,step=5.0)
    duration_ms=st.number_input("duration_ms",min_value=0.425,max_value=2.0,step=0.1)
    danceability=st.number_input("danceability",min_value=0.0,max_value=1.0,step=0.02)
    energy=st.number_input("energy",min_value=0.0,max_value=1.0,step=0.10)
    loudness=st.number_input("loudness",min_value=-20,max_value=20,step=5)
    mode=st.number_input("mode",min_value=0,max_value=1)
    speechiness=st.number_input("speechiness",min_value=0.00,max_value=0.15,step=0.03)
    acousticness=st.number_input("acousticness",min_value=0.00,max_value=1.00)
    instrumentalness=st.number_input("instrumentalness",min_value=0.00,max_value=0.11)
    liveness=st.number_input("liveness",min_value=0.00,max_value=0.53)
    valence=st.number_input("valence",min_value=0.00,max_value=1.0)
    tempo=st.number_input("tempo",min_value=0,max_value=250,step=50)
    
    
    if st.button("Predict"):
        with st.spinner("Thinking...."):
            
            input_data={"popularity":popularity,
                        "duration_ms":duration_ms,
                        "danceability":danceability,
                        "energy":energy,
                        "loudness":loudness,
                        "mode":mode,
                        "speechiness":speechiness,
                        "acousticness":acousticness,
                        "instrumentalness":acousticness,
                        "liveness":liveness,
                        "valence":valence,
                        "tempo":tempo}
            
            try:
                response=requests.post(url_link,json=input_data)
                if response.status_code in [200,201,202]:
                    output=response.json()
                    st.write("Response:",output['result'])
                else:
                    st.warning(f"{response.status_code}->{response.text}")
            except Exception as e:
                st.write(e)
    
    
    
 
        
    
        

    
    
