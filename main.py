from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Annotated
import pickle
from fastapi.responses import JSONResponse
import pandas as pd

app=FastAPI()

with open("xgboost_genre_classifier.pkl","rb") as f:
    model=pickle.load(f)
    
with open("scalar.pkl","rb") as f:
    scalar=pickle.load(f)
    

class user_input(BaseModel):
    popularity:Annotated[float,Field(...,description="Enter the Popularity of the song",gt=0.0,lt=100.0)]
    duration_ms:Annotated[float,Field(...,description="Enter the duration of the song",gt=0.425,lt=2.0)]
    danceability:Annotated[float,Field(...,description="Enter the danceability of the song",gt=0,lt=1.0)]
    energy:Annotated[float,Field(...,description="Enter the energy scale of the song",gt=0.0,lt=1.0)]
    loudness:Annotated[float,Field(...,description='Enter the loudness of the song',gt=-20.0,lt=20.0)]
    mode:Annotated[int,Field(...,description="Enter the mode of the song either 0 or 1")]
    speechiness:Annotated[float,Field(...,description="Enter the spechiness of the song",gt=0.00,lt=0.15)]
    acousticness:Annotated[float,Field(...,description="Enter the accousticness of the song",gt=0.00,lt=1.00)]
    instrumentalness:Annotated[float,Field(...,description="Enter the instrumentalness of the song",gt=0.00,lt=0.12)]
    liveness:Annotated[float,Field(...,description="Enter the liveness of the song",gt=0.00,lt=0.53)]
    valence:Annotated[float,Field(...,description="Enter the valence",gt=0.0,lt=1.0)]
    tempo:Annotated[float,Field(...,description="Enter the tempo of the song",gt=0,lt=250)]



@app.get("/about")
def view():
    return {"message":"Classify the audio genre"}


@app.post("/predict")
def predict(value:user_input):
    
    input_data=pd.DataFrame([{"popularity":value.popularity,
                              "duration_ms":value.duration_ms,
                              "danceability":value.danceability,
                              "energy":value.energy,
                              "loudness":value.loudness,
                              "mode":value.mode,
                              "speechiness":value.speechiness,
                              "acousticness":value.acousticness,
                              "instrumentalness":value.instrumentalness,
                              "liveness":value.liveness,
                              "valence":value.valence,
                              "tempo":value.tempo}])
    
    scaler_input=scalar.transform(input_data)
    
    prediction=model.predict(scaler_input)
    
    cluster_map = {
    0: "Accosutic",
    1: "Chill",
    2: "Country",
    3: "Feeling",
    4: "Hip-hop"}
    
    predicted_cluster=prediction[0]
    final_result=cluster_map.get(predicted_cluster,"Unknown genre")
    return JSONResponse(status_code=200,content={"result":final_result})
    