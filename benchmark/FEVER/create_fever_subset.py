# https://fever.ai/dataset/fever.html
import json
import os
import random
from collections import defaultdict

from adas_core.logging_config import get_logger, setup_logging

logger = get_logger("create_fever_subset")


def extract_fever_combined_subset(input_file, output_file):
    logger.info(f"Reading input file: {input_file}")
    samples_by_label = defaultdict(list)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                label = sample.get("label")
                if label:
                    # Remove evidence as per original script logic
                    if "evidence" in sample:
                        del sample["evidence"]
                    samples_by_label[label].append(sample)
    except FileNotFoundError:
        logger.error(f"Could not find {input_file}")
        return

    # Ensure we have the standard labels
    target_labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    # Configuration for single batch
    random.seed(42)
    final_subset = []
    samples_per_label = 40

    logger.info("Generating Subset (Seed 42, 40 per label)")

    for label in target_labels:
        available = samples_by_label[label]
        if len(available) < samples_per_label:
            logger.warning(f"Only found {len(available)} samples for {label}, taking all.")
            selected = available
        else:
            selected = random.sample(available, samples_per_label)

        final_subset.extend(selected)
        logger.info(f"  {label}: Selected {len(selected)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_subset, f, indent=2)

    logger.info(f"Total size: {len(final_subset)}")
    logger.info(f"Saved to: {output_file}")


if __name__ == "__main__":
    setup_logging()
    # Adjust paths as necessary for your folder structure
    input_path = "benchmark/FEVER/train.jsonl"
    output_path = "benchmark/FEVER/problem_subset.json"

    extract_fever_combined_subset(input_path, output_path)
