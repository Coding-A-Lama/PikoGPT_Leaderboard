from datasets import Dataset


class DatasetSubsetService:
    """Creates a random subset of a dataset based on a reduction factor."""

    def create_subset(self, dataset: Dataset, reduction_factor: float, seed: int = 42) -> Dataset:
        if reduction_factor <= 0:
            raise ValueError("reduction_factor must be greater than 0")

        target_size = int(dataset.num_rows / reduction_factor)
        target_size = max(1, min(dataset.num_rows, target_size))

        shuffled_dataset = dataset.shuffle(seed=seed)
        indices = list(range(target_size))
        return shuffled_dataset.select(indices)
