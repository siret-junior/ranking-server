# This file is part of Ranking Server.
#
# Copyright (C) 2022    Vít Škrhák <vitek.skrhak@seznam.cz>
#                       Tomáš Souček <soucek.gns@gmail.com>
# 
#  Ranking Server is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 2 of the License, or (at your option)
#  any later version.
# 
#  Ranking Server is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
#  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
#  details.
# 
#  You should have received a copy of the GNU General Public License along with
#  Ranking Server. If not, see <https://www.gnu.org/licenses/>.

import os
import torch
import numpy as np
from flask import Flask, make_response
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
clip_function = None
# CLIP normalized features
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
    # Norm the features for faster cos sim
    features = features / np.linalg.norm(features, axis=-1)[:,np.newaxis]
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
    cosine_similarities = np.dot(clip_features, numpy_response)

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

    env_type = os.getenv('ENV')
    debug = True if env_type == "debug" else False
    app.debug = debug

    app.run(host='0.0.0.0', port=int(os.getenv('PORT')), debug=debug)
