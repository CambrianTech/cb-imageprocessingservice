sudo sh stoplocal.sh

echo "Creating local..."
echo "Creating network"
sudo docker network create --driver bridge cb-backend

echo "Starting planes"
sudo docker run --name planes --gpus all -p 8081:8081 -dt planes

echo "Starting imageproc"
RUN_OPTIONS="--image-local-dir=results --results-local-dir=images"
sudo docker run --name imageproc --gpus all -e PLANES_ADDRESS="http://planes:8081/" -e USER_UPLOADS_BUCKET=cb-user-image-uploads -e RESULTS_BUCKET=cb-user-image-uploads -e FILES_BUCKET=cb-imageprocessingservice-models -e RUN_OPTIONS=${RUN_OPTIONS} -p 8080:8080 -dt imageproc

echo "Connecting networks"

sudo docker network connect cb-backend planes
sudo docker network connect cb-backend imageproc

echo "Done. Creating local"