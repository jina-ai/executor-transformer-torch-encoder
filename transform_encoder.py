__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Dict, Generator, Optional, Tuple, List

import numpy as np
import torch
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from transformers import AutoModel, AutoTokenizer


class TransformerTorchEncoder(Executor):
    """
    The transformer torch encoder encodes sentences into embeddings.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
        base_tokenizer_model: Optional[str] = None,
        pooling_strategy: str = 'mean',
        layer_index: int = -1,
        max_length: Optional[int] = None,
        embedding_fn_name: str = '__call__',
        device: str = 'cpu',
        default_traversal_path: Optional[List[str]] = None,
        default_batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param pretrained_model_name_or_path:
        :param base_tokenizer_model:
        :param pooling_strategy:
        :param layer_index:
        :param max_length:
        :param embedding_fn_name:
        :param device:
        :param default_traversal_path: Used in the encode method an define traversal on the received `DocumentArray`
        :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if default_traversal_path is not None:
            self.default_traversal_path = default_traversal_path
        else:
            self.default_traversal_path = ['r']
        self.default_batch_size = default_batch_size
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.base_tokenizer_model = (
            base_tokenizer_model or pretrained_model_name_or_path
        )
        self.pooling_strategy = pooling_strategy
        self.layer_index = layer_index
        self.max_length = max_length
        self.logger = JinaLogger(self.__class__.__name__)
        if not device in ['cpu', 'cuda']:
            self.logger.error('Torch device not supported. Must be cpu or cuda!')
            raise RuntimeError('Torch device not supported. Must be cpu or cuda!')
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your'
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'
        self.device = device
        self.embedding_fn_name = embedding_fn_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_model)
        self.model = AutoModel.from_pretrained(
            self.pretrained_model_name_or_path, output_hidden_states=True
        )
        self.model.to(torch.device(device))

    def _compute_embedding(
        self, hidden_states: Tuple['torch.Tensor'], input_tokens: Dict
    ):
        fill_vals = {'cls': 0.0, 'mean': 0.0, 'max': -np.inf, 'min': np.inf}
        fill_val = torch.tensor(
            fill_vals[self.pooling_strategy], device=torch.device(self.device)
        )
        layer = hidden_states[self.layer_index]
        attn_mask = input_tokens['attention_mask'].unsqueeze(-1).expand_as(layer)
        layer = torch.where(attn_mask.bool(), layer, fill_val)
        embeddings = layer.sum(dim=1) / attn_mask.sum(dim=1)
        return embeddings.cpu().numpy()

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        for batch in self._get_docs_batch_generator(docs, parameters):
            texts = batch.get_attributes('text')

            with torch.no_grad():
                input_tokens = self._generate_input_tokens(texts)
                outputs = getattr(self.model, self.embedding_fn_name)(**input_tokens)
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()
                hidden_states = outputs.hidden_states
                embeds = self._compute_embedding(hidden_states, input_tokens)
                for doc, embed in zip(batch, embeds):
                    doc.embedding = embed

    def _generate_input_tokens(self, texts):
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer.vocab))

        input_tokens = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt',
        )
        input_tokens = {
            k: v.to(torch.device(self.device)) for k, v in input_tokens.items()
        }
        return input_tokens

    def _get_docs_batch_generator(self, docs: DocumentArray, parameters: Dict):
        traversal_path = parameters.get('traversal_path', self.default_traversal_path)
        batch_size = parameters.get('batch_size', self.default_batch_size)
        flat_docs = docs.traverse_flat(traversal_path)
        filtered_docs = DocumentArray(
            [doc for doc in flat_docs if doc is not None and doc.text is not None]
        )
        return _batch_generator(filtered_docs, batch_size)


def _batch_generator(
    data: DocumentArray, batch_size: int
) -> Generator[DocumentArray, None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]
