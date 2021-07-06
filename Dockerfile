FROM jinaai/jina:2.0.3

COPY . ./transformer-text-encoder/
WORKDIR ./transformer-text-encoder

RUN pip install .

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]

