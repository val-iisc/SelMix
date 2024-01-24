from collections import Counter
from torch.utils.data import Dataset

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]



class IndexDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """
    def __init__(self,
                 targets=None):
        """
        Args
            targets: y_data (if not exist, None)
        """
        super(IndexDataset, self).__init__()
        self.targets = targets
        self.prior = [n/len(targets) for n in Counter(self.targets).values()] # type: ignore

        
    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        return idx, self.targets[idx] # type: ignore
    
    def __len__(self):
        return len(self.targets) # type: ignore

