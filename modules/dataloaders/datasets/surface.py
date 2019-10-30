from torch.utils.data import Dataset
import settings

class SurfaceSegmentation(Dataset):
    NUM_CLASSES = len(settings.class_names)

    def __init__(self, base_dir=settings.root_dir, split='train'):
        super().__init__()