# Example: Running the full evaluation pipeline

from vlm_weather.evaluators.compute_score import compute_text_generation

if __name__ == "__main__":
    result = compute_text_generation("data/results/vlm_test_dataset.json")
    print(result)
