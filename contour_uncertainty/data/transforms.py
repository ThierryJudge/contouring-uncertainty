from albumentations import ImageOnlyTransform


class Normalize255Sample(ImageOnlyTransform):
    def apply(self, img, **params):
        return img / 255
