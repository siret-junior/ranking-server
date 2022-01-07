import os
import torch
import numpy as np
from flask import Flask, make_response
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
clip_function = None
clip_features = None


def init_clip():
    import clip
    model, _ = clip.load("RN50x4", device="cpu")

    def fc(query):
        with torch.no_grad():
            tensor = clip.tokenize([query])
            tensor = model.encode_text(tensor)
            tensor = tensor.numpy()[0]
        return tensor / np.linalg.norm(tensor)
    return fc

def load_clip_features():
    features = np.fromfile(os.getenv('CLIP_FEATURES'), dtype=np.float32)
    dim = int(os.getenv('CLIP_DIMENSION'))
    features_amount = len(features) / dim
    features = features.reshape(int(features_amount), dim)
    return features

def get_clip(query):
    print(f"clip: '{query}'")
    numpy_response = clip_function(query)
    response = make_response(numpy_response.astype(np.float32).tobytes("C"))
    response.headers.set("Content-Type", "application/octet-stream")
    return response

def get_clip_results(query):
    print(f"clip-results: '{query}'")

    # compute clip representation of query and compute similarities with respect to the query
    numpy_response = clip_function(query)
    cosine_similarities = cosine_similarity(np.array([numpy_response]), clip_features)[0]

    # take only the N closest results
    how_many = int(os.getenv('HOW_MANY_RESULTS'))
    first_N_results = np.argsort(cosine_similarities)[::-1][:how_many]
    first_N_similarities = cosine_similarities[first_N_results]

    response = make_response(first_N_results.astype(np.int32).tobytes("C") +
                             first_N_similarities.astype(np.float32).tobytes("C"))
    response.headers.set("Content-Type", "application/octet-stream")
    return response

if __name__ == "__main__":
    load_dotenv()
    clip_features = load_clip_features()
    clip_function = init_clip()
    
    app.route("/clip/<query>", methods=["GET"])(get_clip)
    app.route("/clip-results/<query>", methods=["GET"])(get_clip_results)

    app.run(host='0.0.0.0', port=int(os.getenv('PORT')), debug=True)