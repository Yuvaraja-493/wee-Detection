from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from predict import predict_image

app = FastAPI()

@app.post("/predict/")
async def detect_weed(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(file_path)
    return JSONResponse(content={"prediction": result})
