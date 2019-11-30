import logging
import argparse

from typing import List

import torch
import transformers

from helpers import loading


MODEL_TYPE = "BertForNextSentencePrediction"
TOKENIZER_TYPE = "BertTokenizer"
MODEL_NAME = "bert-base-uncased"


logging.getLogger("transformers").setLevel(logging.ERROR)


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        texts = f.readlines()
    return texts


def choose_best_answer(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer,
                       question: str, contender_answers: List[str]) -> str:
    answer_score: torch.tensor = torch.zeros((len(contender_answers), 1))
    with torch.no_grad():
        for text_index, answer in enumerate(contender_answers):
            qa_encoded = tokenizer.encode_plus(question, answer)
            del qa_encoded["special_tokens_mask"]
            qa_vectorized = {arg_name: torch.tensor([arg_value], dtype=torch.long)
                             for arg_name, arg_value in qa_encoded.items()}
            outputs = model(**qa_vectorized)[0]

            answer_score[text_index] = outputs.squeeze()[0]

    best_answer_index = answer_score.argmax()
    return contender_answers[best_answer_index.item()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="How are you?", type=str, required=False, help="Question to answer")
    parser.add_argument("--file_path", default="data/answers", type=str, required=False,
                        help="Path to file with answers")
    args = parser.parse_args()

    model = loading.load_model(MODEL_TYPE, MODEL_NAME)
    model.eval()
    tokenizer = loading.load_tokenizer(TOKENIZER_TYPE, MODEL_NAME)

    answers = read_file(args.file_path)

    best_answer = choose_best_answer(model, tokenizer, args.question, answers)

    print(f"Question: {args.question}")
    print(f"Answer:   {best_answer}")


if __name__ == '__main__':
    main()
