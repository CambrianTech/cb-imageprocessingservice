echo "\nStopping existing containers..."

echo "Stopping planes"
sudo docker container kill planes
sudo docker container rm planes

echo "Stopping imageproc"
sudo docker container kill imageproc 
sudo docker container rm imageproc

echo "Removing network cb-backend"
sudo docker network rm cb-backend

echo "Done. Stopping existing containers.\n"