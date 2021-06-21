<p align="center">
<img src="https://github.com/jina-ai/jina/blob/master/.github/logo-only.gif?raw=true" alt="Jina banner" width="200px">
</p>

# Transformer Torch Encoder

### Description
The transformer torch encoder encodes sentences into embeddings.

### Parameters
The following parameters can be used:

- `pretrained_model_name_or_path` (str): Path to pretrained model or name of the model in transformers package
- `base_tokenizer_model` (str, default None): Base tokenizer model
- `pooling_strategy` (str, default 'mean'): Pooling Strategy
- `layer_index` (int, default -1): Index of the layer which contains the embeddings
- `max_length` (int, default None): Max length argument for the tokenizer
- `embedding_fn_name` (str, default __call__): Function to call on the model in order to get output
- `device` (str, default 'cpu'): Device to be used. Use 'cuda' for GPU

