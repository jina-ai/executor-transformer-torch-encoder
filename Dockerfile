FROM jinaai/jina:master as base

COPY . ./transformer-text-encoder/
WORKDIR ./transformer-text-encoder

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]