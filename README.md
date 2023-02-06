# House_Price_Prediction

Account Requirements:

1. [Heroku Account](https://id.heroku.com/login)
2. [GIT CLI](https://git-scm.com/downloads)

Create conda environment
```
conda create -p venv python==3.7 -y
```
```
conda activate venv or conda activate venv/
```

Installing all the packages:
```
pip install -r requirements.txt
```

Setup CI/CD Pipeline in Heruko

1. Heruko_email = "maheshmouli225@gmail.com"
2. Heruko_API_key = "69dfd3e7-d69d-4c46-96b0-b8b384d60d83"
3. Heruko_App_Name = "house-price-prediction"

Build Docker Image
```
docker build -t <image_name>:<tagname> .
```
> Note: Image name should be lowercase

To list docker images
```
docker images
```
Run Docker image
```
docker run -p 5000:5000 -e PORT=5000 imageID
```
To check running container in docker
```
docker ps
```
To stop docker container
```
docker stop <container_ID>
```
