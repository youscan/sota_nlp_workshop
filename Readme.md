# SOTA NLP Workshop @ YouScan

For the last year there was a big progress in NLP, and now it is easier to build a good prototype for your product.
We will show how to solve three tasks really quickly with a small effort.

We will use [transformers](https://github.com/huggingface/transformers) library under the hood.

## Set virtual environment
1. Create virtual env with `python3.6 -m venv venv`
2. Activate environment with `source venv/bin/activate`
3. Install all packages with `pip install -r requirements.txt`

## Language generation

Simple language generation with GPT-2 model.

We provide command line interface with the next params:

* main parameters:
    * input_text – starting point of our generation
    * n_sentences – number of sentences to generate
    
* technical:
    * temperature – technical parameter to add some randomness
    * repetition_penalty – penalize model for repetition
    * top_k – choose prediction from k most probable tokens
    * top_p – choose prediction from the top tokens with cumulative probability >= top_p

Run `python language_generation.py --input_text="Write some text"`

## Choose the best answer

The task is to find the best answer from the list to the asked question. 

We provide command line interface with the next params:

* question – the question, we must find the answer to
* file_path – file path to file with answers list

Questions:
- How are you?
- How old are you?
- What is your job?
- Did you enjoy last trip?
- Was it an improvisation?

Run `python choose_best_answer.py --question="How are you?"`


## Find the answer in the paragraph

The task is to find the answer to the question based on the provided document.

We provide command line interface with the next params:
* question – the question, we must find the answer to
* file_path – file path to a paragraph

Questions:
- When was he born?
- What is his profession?
- How many goals did he score in Champions League?
- Where does Ronaldo play now?
- What trophies did Ronaldo win?
- How many goals did he score?

Run `python find_answer_in_paragraph.py --question="What is his profession?"`
