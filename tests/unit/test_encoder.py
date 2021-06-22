from typing import Callable

import pytest
import os
import torch

from jina import DocumentArray, Document
from jinahub.text.encoders.transform_encoder import TransformerTorchEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_compute_tokens():
    enc = TransformerTorchEncoder(base_tokenizer_model='bert-base-cased')

    tokens = enc._generate_input_tokens(['hello this is a test', 'and another test'])

    assert tokens['input_ids'].shape == (2, 7)
    assert tokens['token_type_ids'].shape == (2, 7)
    assert tokens['attention_mask'].shape == (2, 7)


def test_compute_embeddings():
    embedding_size = 10
    enc = TransformerTorchEncoder()
    tokens = enc._generate_input_tokens(['hello world'])
    hidden_states = tuple(torch.zeros(1, 4, embedding_size) for _ in range(7))

    embeddings = enc._compute_embedding(
        hidden_states=hidden_states,
        input_tokens=tokens
    )

    assert embeddings.shape == (1, embeddings)


def test_encoding_cpu():
    enc = TransformerTorchEncoder(device='cpu')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (768,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU is needed for this test')
def test_encoding_gpu(data_generator: Callable):
    enc = TransformerTorchEncoder(device='cuda')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (768, )
