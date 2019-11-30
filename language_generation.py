import logging
import argparse
from typing import Callable, Tuple
from collections import Counter

import torch
import torch.nn.functional as F
import transformers
from helpers import loading, filtering


MODEL_TYPE = "GPT2LMHeadModel"
TOKENIZER_TYPE = "GPT2Tokenizer"
MODEL_NAME = "distilgpt2"

END_TEXT_TOKEN = "<|endoftext|>"


logging.getLogger("transformers").setLevel(logging.ERROR)


def exactly_n_chars(text: str, n: int, chars: Tuple[str] = ("?", ".", "!")) -> bool:
    char_counter = Counter(text)
    return sum(char_counter[char] for char in chars) >= n or text.endswith(END_TEXT_TOKEN)


def to_tensor(text: str, tokenizer: transformers.PreTrainedTokenizer) -> torch.tensor:
    text_encoded = tokenizer.encode(text)
    return torch.tensor([text_encoded], dtype=torch.long)


def to_text(tensor: torch.tensor, tokenizer: transformers.PreTrainedTokenizer) -> str:
    return tokenizer.decode(tensor.tolist())


def generate_sequence(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
                      text: str, stop_generation: Callable[[str], bool],
                      repetition_penalty: float, temperature: float, top_p: float, top_k: int) -> str:
    text_generated = to_tensor(text, tokenizer)
    result = text
    with torch.no_grad():
        while not stop_generation(result):
            predictions = model(text_generated)
            next_token_logits = predictions[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            for token_index in set(text_generated[0].tolist()):
                next_token_logits[0, token_index] /= repetition_penalty

            filtered_logits = filtering.top_k_top_p_filtering(next_token_logits, top_p=top_p, top_k=top_k)
            if temperature == 0:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            text_generated = torch.cat((text_generated, next_token), dim=1)
            result = to_text(text_generated.squeeze(), tokenizer)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", default=None, type=str, required=True, help="Input text for generation")
    parser.add_argument("--n_sentences", default=2, type=int, required=False, help="How many sentences to generate")

    parser.add_argument("--temperature", type=float, default=1.2, help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.7)
    args = parser.parse_args()

    model = loading.load_model(MODEL_TYPE, MODEL_NAME)
    model.eval()
    tokenizer = loading.load_tokenizer(TOKENIZER_TYPE, MODEL_NAME)

    generated_sequence = generate_sequence(model, tokenizer, args.input_text,
                                           stop_generation=lambda x: exactly_n_chars(x, args.n_sentences),
                                           temperature=args.temperature,
                                           repetition_penalty=args.repetition_penalty,
                                           top_k=args.top_k,
                                           top_p=args.top_p)
    print(generated_sequence)


if __name__ == '__main__':
    main()
