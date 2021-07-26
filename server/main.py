import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv
import pandas as pd
import os
from pycaret.classification import *

load_dotenv()
app = FastAPI()

class Model:
    def __init__(self, modelname, bucketname):
        self.model = load_model(modelname, platform = 'aws', authentication = { 'bucket' : bucketname })
    
    def predict(self, data):
        predictions = predict_model(self.model, data=data).Label.to_list()
        return predictions


model_et = Model("et_deployed", "lsmlops200020065")
model_lightgbm = Model("lightgbm_deployed", "lsmlops200020065")

@app.post("/et/predict")
async def create_upload_file(file: UploadFile = File(...)):
        if file.filename.endswith(".csv"):
        
            with open(file.filename, "wb")as f:
                f.write(file.file.read())
            data = pd.read_csv(file.filename)
            os.remove(file.filename)

            try:
                predictions = model_et.predict(data)
            except:
                raise HTTPException(status_code=406, detail="Invalid data format.")
            else:
                return {
                    "Labels": predictions
                }

        else:
            raise HTTPException(status_code=406, detail="Invalid file format. Only CSV Files accepted.")


@app.post(f"/lightgbm/predict")
async def create_upload_file(file: UploadFile = File(...)):
        if file.filename.endswith(".csv"):
        
            with open(file.filename, "wb")as f:
                f.write(file.file.read())
            data = pd.read_csv(file.filename)
            os.remove(file.filename)

            try:
                predictions = model_lightgbm.predict(data)
            except:
                raise HTTPException(status_code=406, detail="Invalid data format.")
            else:
                return {
                    "Labels": predictions
                }

        else:
            raise HTTPException(status_code=406, detail="Invalid file format. Only CSV Files accepted.")


if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)
