from typing import Callable

import pytest
import os
import torch

from jina import DocumentArray, Document
from jinahub.text.encoders.transform_encoder import TransformerTorchEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_initialization():
    enc = TransformerTorchEncoder()


def test_encoding():
    enc = TransformerTorchEncoder(device='cpu')

    input_data = DocumentArray([Document(text='hello world')])
    enc.encode(docs=input_data, parameters={})
    assert input_data[0].embedding.shape == (768,)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='GPU is needed for this test')
def test_gpu(data_generator: Callable):
    enc = TransformerTorchEncoder(device='cuda')

    input_data = DocumentArray([Document(text='hello world')])
    enc.encode(docs=input_data, parameters={})
    assert input_data[0].embedding.shape == (768, )
