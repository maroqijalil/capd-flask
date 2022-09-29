#!/bin/bash

docker image rm -f capd-detection:latest

docker build -f docker/Dockerfile -t capd-detection:latest capd-detection/

if [ "$(docker images | grep capd-detection)" ]; then
  docker run -dp 5001:5001 --rm -it capd-detection
fi
