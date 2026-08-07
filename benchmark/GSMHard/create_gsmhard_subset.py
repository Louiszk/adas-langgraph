import os
import json
import random
from datasets import load_dataset  # type: ignore
from adas_core.logging_config import setup_logging, get_logger

logger = get_logger("create_gsmhard_subset")


def generate_subset():
    logger.info("Loading GSM-Hard dataset from HuggingFace...")
    # Load the full dataset
    try:
        hf_dataset = load_dataset("reasoning-machines/gsm-hard")["train"]
    except Exception as e:
        logger.error(f"Error loading HF dataset: {e}")
        return

    full_indices = range(len(hf_dataset))

    # Generate single batch (120 items, seed 42)
    random.seed(42)
    target_size = 120

    if len(full_indices) < target_size:
        logger.warning(f"Dataset only has {len(full_indices)} items. Using all.")
        indices_120 = list(full_indices)
    else:
        indices_120 = random.sample(full_indices, target_size)

    logger.info(f"Selected {len(indices_120)} problems (seed 42).")

    # Construct the list of problems
    output_data = []
    for idx in indices_120:
        item = hf_dataset[int(idx)]
        output_data.append(
            {
                "input": item["input"],
                "target": item["target"],
            }
        )

    # Define output path
    output_dir = "benchmark/GSMHard"
    output_file = os.path.join(output_dir, "problem_subset.json")

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Successfully saved {len(output_data)} problems to {output_file}")


if __name__ == "__main__":
    setup_logging()
    generate_subset()
