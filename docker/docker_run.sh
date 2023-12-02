# Build the docker file 
docker build -t digits:v1 -f ./docker/Dockerfile .
# Create out volume
docker volume create mltrain
# Mount our volume to models directory (where train data is stored)
docker run -d  -p 80:5000 -v mltrain:/digits/models digits:v1