
# fast api code start
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

app.imageURL = ["0b7f4987-34e4-4c85-9f28-35e04ae78ece.jpg"]
app.name = "yogii"
class User(BaseModel):
    url: str


@app.get("/get_user_info/")
async def get_user_info(img: User):

    ImageURL = img
    return True

@app.get("/changeName/{name}")
async def test(name: str):
    app.name = name
    return {"name" : name}

@app.get("/imageurl")
async def test():
    return {"url" : app.imageURL}

@app.get("/getData")
async def test():
    r = requests.get(url=f"https://project-backend-wuav.onrender.com/userDetails?name={app.name.lower()}")
    data = r.json()
    print(data)
    inputImage = data['orders']
    print(inputImage)
    return {"name": data["name"],"history" : data["orders"],"email": data["email"],"contact": data["contact"]}

@app.get("/image/{image_name}")
async def test(image_name: str):
    print(image_name)
    print(app.imageURL)
    app.imageURL = image_name
    return {"Hello": app.imageURL}

