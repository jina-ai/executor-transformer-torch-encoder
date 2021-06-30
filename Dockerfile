FROM jinaai/jina:2.0

COPY . ./transformer-text-encoder/
WORKDIR ./transformer-text-encoder

RUN pip install .

FROM base
RUN pip install -r tests/requirements.txt
RUN pytest -s -v tests

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]