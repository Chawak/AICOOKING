import requests

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
# from model import model_repo

app = FastAPI()


class Body(BaseModel):
    prompt: str = "ข้อความในรูปเขียนว่าอะไร"
    url: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def chat(body: Body):
    prompt = body.prompt
    url = body.url
    try:
        image = Image.open(requests.get(url, stream=True).raw)

    except Exception:
        return JSONResponse(status_code=400, content={"message": "Bad url"})

    outputs = model_repo.predict(image, prompt)

    return {"success": True, "output": outputs}
