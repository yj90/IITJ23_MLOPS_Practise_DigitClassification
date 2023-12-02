# Build the docker file 
docker build -t base:latest -f ./docker/DependencyDockerfile .
docker build -t digits:latest -f ./docker/FinalDockerfile .
docker run digits:latest 