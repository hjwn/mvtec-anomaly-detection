from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecAD(Dataset):
    """
    mode:
      - "train": train/good only
      - "test":  test/* (good + defects)
    returns:
      image: Tensor [3,H,W]
      label: 0 (good) or 1 (anomaly)
      mask : Tensor [1,H,W] or None (good has no mask)
      meta : dict (category, defect_type, filename)
    """
    def __init__(self, root: str, category: str, mode: str = "train", image_size: int = 224):
        self.root = Path(root)
        self.category = category
        self.mode = mode

        if mode not in ("train", "test"):
            raise ValueError("mode must be 'train' or 'test'")

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # keeps 0/1-ish
        ])

        self.samples = self._build_index()

    def _build_index(self):
        cat_dir = self.root / self.category
        if self.mode == "train":
            img_dir = cat_dir / "train" / "good"
            paths = sorted(img_dir.glob("*.*"))
            return [(p, 0, None, "good") for p in paths]

        # test
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"
        samples = []
        for defect_dir in sorted([p for p in test_dir.iterdir() if p.is_dir()]):
            defect_type = defect_dir.name
            for img_path in sorted(defect_dir.glob("*.*")):
                if defect_type == "good":
                    samples.append((img_path, 0, None, defect_type))
                else:
                    # ground truth mask path: ground_truth/<defect>/<name>_mask.png
                    mask_path = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                    samples.append((img_path, 1, mask_path, defect_type))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path, defect_type = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        if mask_path is not None and mask_path.exists():
            m = Image.open(mask_path).convert("L")
            mask = self.mask_transform(m)
            mask = (mask > 0.5).float()
        else:
            # dummy zero mask for good samples
            mask = torch.zeros((1, img.shape[1], img.shape[2]))

        meta = {
            "category": self.category,
            "defect_type": defect_type,
            "filename": img_path.name
        }
        return img, torch.tensor(label, dtype=torch.long), mask, meta
