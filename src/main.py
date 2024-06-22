import requests

from fastapi import FastAPI, Security, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi.security import APIKeyHeader

from model import model_repo

api_key_header = APIKeyHeader(name="X-API-Key")
app = FastAPI()


def get_user(api_key_header: str = Security(api_key_header)):
    if api_key_header != "moota":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key ผิดนะ อิอิ",
        )
    return True


class Body(BaseModel):
    prompt: str = "ข้อความในรูปเขียนว่าอะไร"
    url: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def chat(body: Body, middleware=Depends(get_user)):
    prompt = body.prompt
    url = body.url
    try:
        image = Image.open(requests.get(url, stream=True).raw)

    except Exception:
        return JSONResponse(status_code=400, content={"message": "Bad url"})

    outputs = model_repo.predict(image, prompt)

    return {"success": True, "output": outputs}
