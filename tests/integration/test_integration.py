__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Callable
import pytest

from jina import Flow, DocumentArray
from jinahub.text.encoders.transform_encoder import TransformerTorchEncoder


@pytest.mark.parametrize(
    'request_size', [1, 10, 100]
)
def test_integration(
    data_generator: Callable,
    request_size: int
):
    with Flow(return_results=True).add(uses=TransformerTorchEncoder) as flow:
        data = flow.post(on='/index', inputs=data_generator(), request_size=request_size)
        docs = data[0].docs
        for doc in docs:
            assert doc.embedding is not None
