if [ -z "$1" ]
  then
    echo "Supply a version number. E.g. bash createRelease.sh 0.0.1"
    exit 1
fi

sudo docker build -t gerovanmi/toxic-detection-server -t gerovanmi/toxic-detection-server:$1 .
# sudo docker push gerovanmi/toxic-detection-server:latest
# sudo docker push gerovanmi/toxic-detection-server:$1
