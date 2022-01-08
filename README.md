# Ranking Server
Embeds the text queries into repsective spaces and computes scores for given queries.

This is part of the [SOMHunter](https://github.com/siret-junior/somhunter) project. But is not dependant on in in any way since it is used by other tools as well (e.g. CVHunter). Therefore you must provide data and confgure this server separately.


## Data & Configuration
First create a new folder called `clip_data` and add the binary file with frame features in it. Don't forget to update the filename in .env file as well.

## **Build & Run with Docker (recommended)**
```sh
docker build -t ranking-server .
docker run -p 8083:8083 ranking-server
```

## **Build & Run**
```sh
sh install.sh
sh run.sh
```

## How to Query the Server
```
curl -v http://localhost:5354/clip/my%20text%20query
```
- returns float32 vector of size 640 representing the query in feature space

```
curl -v http://localhost:5354/clip-results/my%20text%20query
```
- returns int32 vector of size 10000 of sorted indexes of the most similar frames
