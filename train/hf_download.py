import argparse
import sys
import transformers

from datasets import load_dataset


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Download Hugging Face model and dataset artifacts into cache.

Why no `hf download`? It works for datasets but not not for model, as
it doesn't unpack model in the cache as needed for the offline usage.

        """
    )
    parser.add_argument(
        "--model",
        default = "gpt2",
        help="Model config name from src/recipe/config/model (for example: gpt2).",
    )
    parser.add_argument(
        "--dataset",
        default = "imdb",
        help="Dataset config name from src/recipe/config/dataset (for example: imdb).",
    )

    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.model:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    if args.dataset:
        dataset = load_dataset(args.dataset)


if __name__ == "__main__":
    main()
