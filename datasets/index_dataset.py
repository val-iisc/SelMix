from collections import Counter
from torch.utils.data import Dataset

class IndexDataset(Dataset):
    """
    IndexDataset returns a pair of index and target.

    If targets are not provided, IndexDataset returns None as the target.
    This class also calculates the prior probability distribution of the targets.
    """

    def __init__(self, targets=None):
        """
        Initialize the IndexDataset.

        Args:
            targets (list, optional): List of target labels. Defaults to None.
        """
        super().__init__()
        self.targets = targets
        self.calculate_prior()

    def __getitem__(self, index):
        """
        Get the index-target pair at the given index.

        Args:
            index (int): Index of the data point.

        Returns:
            tuple: Index-target pair.
        """
        if self.targets is None:
            return index, None
        return index, self.targets[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        if self.targets is None:
            return 0
        return len(self.targets)

    def calculate_prior(self):
        """
        Calculate the prior probability distribution of the targets.
        """
        if self.targets is None:
            self.prior = None
        else:
            target_counts = Counter(self.targets)
            self.prior = [count / len(self.targets) for count in target_counts.values()]