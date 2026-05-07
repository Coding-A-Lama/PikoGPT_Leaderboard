from enum import Enum


class TrainingDatasetImplementation(str, Enum):
    DOCUMENT = "document"
    CONTINUOUS = "continuous"
