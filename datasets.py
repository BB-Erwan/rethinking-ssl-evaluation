from torch.utils.data import Dataset, Subset
import torch

class SemiSupervisedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def split_balanced(self, num_labeled_per_class, labeled_transform=None, unlabeled_transform=None):

        indices = self.get_balanced_indices(num_labeled_per_class)
        labeled_dataset  = Subset(self.dataset, indices)
        unlabeled_dataset = Subset(self.dataset, list(set(range(len(self.dataset))) - set(indices)))

        return SemiSupervisedDataset(labeled_dataset, labeled_transform), SemiSupervisedDataset(unlabeled_dataset, unlabeled_transform)
    
    def get_balanced_indices(self, num_samples_per_class):
        # Iterate through the dataset and collect indices for each class
        class_indices = {}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            label = label.item() if isinstance(label, torch.Tensor) else label
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)

        # Select the specified number of samples per class
        selected_indices = []
        for label, indices in class_indices.items():
            if len(indices) < num_samples_per_class:
                raise ValueError(f"Not enough samples for class {label}")
            selected_indices.extend(indices[:num_samples_per_class])

        return selected_indices

    
class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)
        return image, label
    
class WeakStrongDataset(Dataset):
    def __init__(self, dataset, weak_transform, strong_transform):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        weak_image = self.weak_transform(image)
        strong_image = self.strong_transform(image)
        return weak_image, strong_image, idx