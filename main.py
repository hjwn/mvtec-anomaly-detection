from torch.utils.data import DataLoader
from src.datasets.mvtec import MVTecAD

def main():
    root = "data/mvtec_ad"
    category = "bottle"

    train_ds = MVTecAD(root, category, mode="train", image_size=224)
    test_ds  = MVTecAD(root, category, mode="test",  image_size=224)

    print("train size:", len(train_ds))
    print("test size :", len(test_ds))

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    x, y, m, meta = next(iter(loader))
    print("batch x:", x.shape)
    print("batch y:", y.shape, y[:8])

if __name__ == "__main__":
    main()
