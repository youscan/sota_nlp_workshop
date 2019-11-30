import logging
import argparse
from typing import List

import torch
import transformers
from helpers import loading


MODEL_TYPE = "DistilBertForQuestionAnswering"
TOKENIZER_TYPE = "DistilBertTokenizer"
MODEL_NAME = "distilbert-base-uncased-distilled-squad"


logging.getLogger("transformers").setLevel(logging.ERROR)


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def get_answer(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
               question: str, paragraph: str, n_answers: int) -> List[str]:
    text_tokenized = tokenizer.encode(question, paragraph)
    text_vectorized = torch.tensor([text_tokenized], dtype=torch.long)
    with torch.no_grad():
        start_span_logits, end_span_logits = model(text_vectorized)
        start_span_top = torch.topk(start_span_logits, n_answers).indices.squeeze()
        end_span_top = torch.topk(end_span_logits, n_answers).indices.squeeze()
        if n_answers > 1:
            top_answers = [tokenizer.decode(text_tokenized[start_span:end_span+1])
                           for start_span, end_span in zip(start_span_top.tolist(), end_span_top.tolist())
                           if start_span <= end_span]
        else:
            top_answers = [tokenizer.decode(text_tokenized[start_span_top:end_span_top+1])]
    return top_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, type=str, required=True, help="Question to a paragraph")
    parser.add_argument("--file_path", default="data/paraghraph", type=str, required=False,
                        help="Path to file with a paragraph")
    parser.add_argument("--n_answers", default=1, type=int, required=False, help="How many answers return")
    args = parser.parse_args()

    model = loading.load_model(MODEL_TYPE, MODEL_NAME)
    model.eval()
    tokenizer = loading.load_tokenizer(TOKENIZER_TYPE, MODEL_NAME)

    paragraph = read_file(args.file_path)

    top_answers = get_answer(model, tokenizer, args.question, paragraph, args.n_answers)

    print(f"Question:  {args.question}")
    for answer_index, answer in enumerate(top_answers):
        if args.n_answers > 1:
            print(f"Answer {answer_index + 1}:    {answer}")
        else:
            print(f"Answer:    {answer}")


if __name__ == '__main__':
    main()
