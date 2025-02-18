from pathlib import Path

from contour_uncertainty.augmentations.affine import RandomRotation, RandomTranslation
from contour_uncertainty.augmentations.augmentation import Compose
from contour_uncertainty.augmentations.brightnesscontrast import RandomBrightnessContrast
from contour_uncertainty.augmentations.gamma import RandomGamma
import random
from argparse import ArgumentParser

from matplotlib import pyplot as plt

from contour_uncertainty.data.camus.dataset import CamusContour
from vital.data.config import Subset

if __name__ == "__main__":
    args = ArgumentParser(add_help=False)
    args.add_argument("path", type=str)
    params = args.parse_args()

    ds = CamusContour(
        Path(params.path),
        image_set=Subset.TEST,
        fold=5,
        predict=False,
        points_per_side=11,
    )

    print(len(ds))

    sample = ds[random.randint(0, len(ds) - 1)]

    img = sample['img']
    gt = sample['gt']
    contour = sample['contour']

    t = Compose([RandomRotation(25),
                 RandomGamma((.8, 1.2)),
                 RandomBrightnessContrast(.2, .2),
                 RandomTranslation(10, 10)])

    params = t.get_params()
    print(params)

    output = t.apply({'image': img, 'mask': gt, 'keypoints': contour}, params=params)
    output2 = t.un_apply(output, params=params)

    print(img[0, 150:160, 159])
    print(output['image'][0, 150:160, 159])

    print(contour[0])
    print(output['keypoints'][0])
    print(output2['keypoints'][0])

    print(output['mask'].shape)
    print(output2['mask'].shape)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(img.squeeze(), cmap="gray")
    ax[0, 0].scatter(contour[:, 0], contour[:, 1], s=10, c="r")
    ax[1, 0].imshow(gt.squeeze())

    ax[0, 1].imshow(output['image'].squeeze(), cmap="gray")
    ax[0, 1].scatter(output['keypoints'][:, 0], output['keypoints'][:, 1], s=10, c="r")
    ax[1, 1].scatter(output['keypoints'][:, 0], output['keypoints'][:, 1], s=10, c="r")
    ax[1, 1].imshow(output['mask'].squeeze())

    ax[0, 2].imshow(output2['image'].squeeze(), cmap="gray")
    ax[0, 2].scatter(output2['keypoints'][:, 0], output2['keypoints'][:, 1], s=10, c="r")
    ax[1, 2].scatter(output2['keypoints'][:, 0], output2['keypoints'][:, 1], s=10, c="r")
    ax[1, 2].imshow(output2['mask'].squeeze())

    plt.show()
