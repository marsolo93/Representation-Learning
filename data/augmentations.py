import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from dataclasses import dataclass
import PIL


@dataclass
class DINOImageData:
    globals: list[torch.Tensor]
    locals: list[torch.Tensor]


class DINODataAugmentator:

    def __init__(
        self,
        global_scale: tuple[float] = (0.6, 0.8),
        local_scale: tuple[float] = (0.2, 0.4),
        local_numbers: int = 8,
        global_image_size: int = 224,
        local_image_size: int = 96,
    ):
        self.local_numbers = local_numbers
        global_geometric_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_image_size,
                    scale=global_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        local_geometric_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_image_size,
                    scale=local_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transform_2 = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=0.1,
                ),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transform_extra = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
            p=0.1,
        )

        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.global_1_pipe = transforms.Compose(
            [global_geometric_transforms, color_jittering, normalize]
        )
        self.global_2_pipe = transforms.Compose(
            [
                global_geometric_transforms,
                color_jittering,
                global_transform_2,
                normalize,
            ]
        )
        self.local_pipe = transforms.Compose(
            [
                local_geometric_transforms,
                color_jittering,
                local_transform_extra,
                normalize,
            ]
        )

    def __call__(self, image: PIL.Image.Image) -> DINOImageData:
        global_1: torch.Tensor = self.global_1_pipe(img=image)
        global_2: torch.Tensor = self.global_2_pipe(img=image)
        local_image: list[torch.Tensor] = [
            self.local_pipe(img=image) for _ in range(self.local_numbers)
        ]
        output = DINOImageData(locals=local_image, globals=[global_1, global_2])
        return output


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("../test_image.png")
    image = image.convert("RGB")
    augmentator = DINODataAugmentator()
    data = augmentator(image)
    fig, axs = plt.subplots(nrows=5, ncols=2)
    for i in range(2):
        axs[0, i].imshow(data.globals[i].permute(1, 2, 0).cpu().numpy())
        axs[0, i].imshow(data.globals[i].permute(1, 2, 0).cpu().numpy())
    for i in range(8):
        row = int(i / 2) + 1
        col = int(i % 2)
        axs[row, col].imshow(data.locals[i].permute(1, 2, 0).cpu().numpy())

    plt.show()
