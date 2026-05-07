from dataclasses import dataclass

from datasets import Dataset


@dataclass(frozen=True)
class DatasetSplitResult:
    subset_a: Dataset
    subset_b: Dataset


class DatasetSplittingService:
    """Splits a dataset into two disjoint randomized subsets."""

    def split_dataset(
        self,
        dataset: Dataset,
        percentage_a: float,
        seed: int = 42,
    ) -> DatasetSplitResult:
        if not 0 <= percentage_a <= 100:
            raise ValueError("percentage_a must be between 0 and 100")

        split_index = int(dataset.num_rows * (percentage_a / 100))
        shuffled_dataset = dataset.shuffle(seed=seed)

        subset_a = shuffled_dataset.select(range(split_index))
        subset_b = shuffled_dataset.select(range(split_index, shuffled_dataset.num_rows))
        return DatasetSplitResult(subset_a=subset_a, subset_b=subset_b)
