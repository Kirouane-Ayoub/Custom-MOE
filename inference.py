import argparse

import torch

from dataset import new_tokenizer
from model import SMoELanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using SMoE Language Model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum length of generated sequences",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=10,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--input_text", type=str, default="راك ", help="Input text for generation"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    return parser.parse_args()


def generate_text(
    model: SMoELanguageModel,
    max_length: int,
    num_return_sequences: int,
    input_text: str,
):
    tokens = new_tokenizer.encode(input_text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    ids = model.generate(tokens, max_length=max_length)

    for i in range(num_return_sequences):
        decoded = new_tokenizer.decode(ids[i], skip_special_tokens=True)
        print(">", decoded)


if __name__ == "__main__":
    args = parse_args()

    # Load the model
    model = SMoELanguageModel()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    generate_text(
        model=model,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        input_text=args.input_text,
    )
