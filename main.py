from typing import Optional


from fastapi import FastAPI,HTTPException
import joblib
from pydantic import BaseModel


model = joblib.load("C:/Users/User/Desktop/Tuwaiq Assignmets/Usecase - Lab/Usecase-7/ML_Deploy/DBSCAN_model.joblib","r")
scaler = joblib.load("C:/Users/User/Desktop/Tuwaiq Assignmets/Usecase - Lab/Usecase-7/ML_Deploy/scaler.joblib","r")

app = FastAPI()

class InputFeatures(BaseModel):
    position_encoded: int
    winger: int
    appearance: int
    award: int
    current_value: int
    goals: float
    assists: float
    goals_conceded: float





def preprocessing(input_features: InputFeatures):
    # Create a dictionary without the 'award' field if it's not needed for scaling
    dict_f = {
        'position_encoded': input_features.position_encoded,
        #'winger': input_features.winger,
        'appearance': input_features.appearance,
        'current_value': input_features.current_value,
        'goals': input_features.goals,
        'assists': input_features.assists,
        'goals_conceded': input_features.goals_conceded
    }
    
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    return scaled_features




@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.fit_predict(data)
    return {"pred": y_pred.tolist()[0]}
 

