from datasets import Dataset, load_dataset


class DatasetDownloadService:
    """Downloads a dataset split through Hugging Face datasets."""

    def download_train_split(self, dataset_name: str, cache_dir: str) -> Dataset:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir, split="train")
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Dataset {dataset} is not a Dataset")
        return dataset
