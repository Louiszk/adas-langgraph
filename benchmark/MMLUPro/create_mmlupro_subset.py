import os
import json
import random
from datasets import load_dataset  # type: ignore
from adas_core.logging_config import setup_logging, get_logger

logger = get_logger("create_mmlupro_subset")


def generate_subset():
    logger.info("Loading MMLU-Pro dataset from HuggingFace...")
    # Load the test split
    try:
        hf_dataset = load_dataset("TIGER-Lab/MMLU-Pro")["test"]
    except Exception as e:
        logger.error(f"Error loading HF dataset: {e}")
        return

    # Filter for Computer Science first
    target_category = "computer science"
    category_data = [item for item in hf_dataset if item["category"].lower() == target_category.lower()]

    logger.info(f"Found {len(category_data)} items in category '{target_category}'")

    # Generate single batch (120 items, seed 42)
    random.seed(42)
    target_size = 120

    if len(category_data) < target_size:
        logger.warning(f"Category only has {len(category_data)} items. Using all.")
        indices_120 = category_data
    else:
        indices_120 = random.sample(category_data, target_size)

    logger.info(f"Selected {len(indices_120)} problems (seed 42).")

    # Define output path
    output_dir = "benchmark/MMLUPro"
    output_file = os.path.join(output_dir, "problem_subset.json")

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(indices_120, f, indent=2)

    logger.info(f"Successfully saved {len(indices_120)} problems to {output_file}")


if __name__ == "__main__":
    setup_logging()
    generate_subset()
