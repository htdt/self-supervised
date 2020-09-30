import torchvision.transforms as T


def aug_transform(crop, base_transform, cfg):
    """ augmentation transform generated from config """
    return T.Compose(
        [
            T.RandomApply(
                [T.ColorJitter(cfg.cj0, cfg.cj1, cfg.cj2, cfg.cj3)], p=cfg.cj_p
            ),
            T.RandomGrayscale(p=cfg.gs_p),
            T.RandomResizedCrop(
                crop,
                scale=(cfg.crop_s0, cfg.crop_s1),
                ratio=(cfg.crop_r0, cfg.crop_r1),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.hf_p),
            base_transform(),
        ]
    )


class MultiSample:
    """ generates n samples with augmentation """
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))
