import transformers


def load_model(model_type: str, model_name: str) -> transformers.PreTrainedModel:
    return getattr(transformers, model_type).from_pretrained(model_name)


def load_tokenizer(tokenizer_type: str, model_name: str) -> transformers.PreTrainedTokenizer:
    return getattr(transformers, tokenizer_type).from_pretrained(model_name)
