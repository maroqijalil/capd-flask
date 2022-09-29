# capd-flask
CAPD Detection using Flask

## Build
`docker build -f docker/Dockerfile -t capd-detection capd-detection/`

## Run
`docker run -dp 5001:5001 --rm -it capd-detection`
