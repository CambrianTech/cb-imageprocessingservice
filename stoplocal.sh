echo "Stopping existing containers"
sudo docker container kill planes
sudo docker container rm planes
sudo docker container kill imageproc 
sudo docker container rm imageproc