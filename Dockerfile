FROM continuumio/miniconda3:4.9.2

RUN apt-get --allow-releaseinfo-change update && apt-get install -y g++

RUN pip install \
        flask \
        Cython \
        ftfy regex tqdm python-dotenv scikit-learn \
        torch==1.7.1+cpu \
        torchvision==0.8.2+cpu \
        git+https://github.com/openai/CLIP.git@cfcffb90e69f37bf2ff1e988237a0fbe41f33c04 \
        --find-links https://download.pytorch.org/whl/torch_stable.html

RUN pip install fairseq==0.9.0

# download clip model into docker image so it is downloaded only once
RUN mkdir -p /root/.cache/clip \
 && cd /root/.cache/clip \
 && wget -q https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt

RUN mkdir -p /root/.cache/torch/pytorch_fairseq \
 && cd /root/.cache/torch/pytorch_fairseq \
 && wget -q https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json \
         -O e2aab4d600e7568c2d88fc7732130ccc815ea84ec63906cb0913c7a3a4906a2e.0f323dfaed92d080380e63f0291d0f31adfa8c61a62cbcb3cb8114f061be27f7 \
 && wget -q https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe \
         -O b04a6d337c09f464fe8f0df1d3524db88a597007d63f05d97e437f65840cdba5.939bed25cbdab15712bac084ee713d6c78e221c5156c68cb0076b03f5170600f

COPY app.py /ranking-server/app.py
COPY clip_data /ranking-server/clip_data
COPY .env /ranking-server/.env

WORKDIR /ranking-server

EXPOSE 8083

#CMD ["python", "app.py"]
