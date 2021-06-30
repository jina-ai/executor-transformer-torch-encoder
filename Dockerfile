FROM jinaai/jina:2.0

COPY . ./transformer-text-encoder/
WORKDIR ./transformer-text-encoder

RUN pip install .