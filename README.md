### Data
First create a new folder for CLIP data:
```
mkdir clip_data
```
and add the binary file into the newly created folder. Don't forget to update the filename in .env file.

### Run
```
docker build -t clip-server .
docker run -d --net host --name clip_server clip-server

# stop
docker stop -t 1 clip_server
docker rm clip_server
```

### Debug
```
docker build -t clip-server .
docker run -it --rm --net host clip-server

# stop
docker stop -t 1 clip_server
docker rm clip_server
```

### Query server
```
curl -v http://localhost:5354/clip/my%20text%20query
```
- returns float32 vector of size 640 representing the query in feature space

```
curl -v http://localhost:5354/clip-results/my%20text%20query
```
- returns int32 vector of size 10000 of sorted indexes of the most similar frames