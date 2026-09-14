import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


class TransformWrapper(Dataset):
    """Apply a transform to a subset sharing one base dataset."""

    def __init__(self, subset, transform=None):
        """Store the base subset and its split-specific transform."""
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        """Load and transform one indexed image-label pair."""
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        """Return the number of samples exposed by the subset."""
        return len(self.subset)


class DatasetSampler(torch.utils.data.sampler.Sampler):
    """Sample underrepresented classes more frequently."""

    def __init__(self, dataset, labels=None, indices=None, num_samples=None, seed=None):
        """Compute inverse-frequency sampling weights."""
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        if labels is None:
            if hasattr(dataset, "targets"):
                labels = dataset.targets
            elif hasattr(dataset, "labels"):
                labels = dataset.labels
            elif isinstance(dataset, Subset):
                base_dataset = dataset.dataset
                if hasattr(base_dataset, "targets"):
                    labels = [base_dataset.targets[i] for i in dataset.indices]
                elif hasattr(base_dataset, "labels"):
                    labels = [base_dataset.labels[i] for i in dataset.indices]
                else:
                    raise ValueError("Dataset does not expose class labels")
            else:
                raise ValueError("Dataset does not expose class labels")

        if len(labels) == len(dataset) and len(labels) != len(self.indices):
            labels = [labels[i] for i in self.indices]
        elif len(labels) != len(self.indices):
            raise ValueError("labels and sampler indices must have matching lengths")

        df = pd.DataFrame({"label": labels}, index=self.indices)
        label_counts = df["label"].value_counts()
        weights = 1.0 / label_counts[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        """Draw a balanced sequence of dataset indices with replacement."""
        return (
            self.indices[i]
            for i in torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
                generator=self.generator,
            )
        )

    def __len__(self):
        """Return the number of samples drawn per epoch."""
        return self.num_samples
