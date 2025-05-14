import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Define helper structures and classes
class TrainTricks:
    def __init__(self):
        # Placeholder for training tricks, not used in prediction
        pass


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, upsampling: float = 1.0):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        self.upsampling = nn.Upsample(scale_factor=upsampling) if upsampling != 1.0 else nn.Identity()
        self.register_module("conv2d", self.conv2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x


class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels: List[int],
            encoder_depth: int,
            pyramid_channels: int,
            segmentation_channels: int,
            dropout: float,
            merge_policy: str
    ):
        super().__init__()
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3")

        # Reverse encoder channels to match FPN processing order
        encoder_channels = encoder_channels[-encoder_depth - 1:][::-1]

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4

        # FPN blocks
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        # Segmentation blocks
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, i)
            for i in range(4)[::-1]
        ])

        # Merge block
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.register_module("p5", self.p5)
        self.register_module("p4", self.p4)
        self.register_module("p3", self.p3)
        self.register_module("p2", self.p2)
        self.register_module("seg_blocks", self.seg_blocks)
        self.register_module("merge", self.merge)
        self.register_module("dropout", self.dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        length_features = len(features)
        p5 = self.p5(features[length_features - 1])
        p4 = self.p4(p5, features[length_features - 2])
        p3 = self.p3(p4, features[length_features - 3])
        p2 = self.p2(p3, features[length_features - 4])

        p5 = self.seg_blocks[0](p5)
        p4 = self.seg_blocks[1](p4)
        p3 = self.seg_blocks[2](p3)
        p2 = self.seg_blocks[3](p2)

        x = self.merge([p5, p4, p3, p2])
        x = self.dropout(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels: int, skip_channels: int):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest')
        self.register_module("skip_conv", self.skip_conv)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        return x + skip


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int):
        super().__init__()
        blocks = []
        blocks.append(Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples)))
        for _ in range(1, n_upsamples):
            blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)
        self.register_module("block", self.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2.0, mode='nearest') if upsample else nn.Identity()
        self.register_module("block", self.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.upsample(x)
        return x


class MergeBlock(nn.Module):
    def __init__(self, policy: str):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("Policy must be 'add' or 'cat'")
        self.policy = policy

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.policy == "add":
            return sum(x)
        else:  # cat
            return torch.cat(x, dim=1)


def encoder_parameters():
    return {
        "resnet18": {
            "class_type": "resnet",
            "outChannels": [3, 64, 64, 128, 256, 512],
            "layers": [2, 2, 2, 2]
        },
        "resnet34": {
            "class_type": "resnet",
            "outChannels": [3, 64, 64, 128, 256, 512],
            "layers": [3, 4, 6, 3]
        },
        "resnet50": {
            "class_type": "resnet",
            "outChannels": [3, 64, 256, 512, 1024, 2048],
            "layers": [3, 4, 6, 3]
        }
    }


class Block(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, downsample: Optional[nn.Module], groups: int,
                 base_width: int, is_basic: bool):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        if not is_basic:
            self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
            self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.is_basic = is_basic

        self.register_module("conv1", self.conv1)
        self.register_module("bn1", self.bn1)
        self.register_module("conv2", self.conv2)
        self.register_module("bn2", self.bn2)
        if not is_basic:
            self.register_module("conv3", self.conv3)
            self.register_module("bn3", self.bn3)
        if downsample is not None:
            self.register_module("downsample", self.downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if not self.is_basic:
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers: List[int], num_classes: int, model_type: str, groups: int = 1,
                 width_per_group: int = 64):
        super().__init__()
        self.type_model = model_type
        self.expansion = 1 if model_type in ["resnet18", "resnet34"] else 4
        self.is_basic = model_type in ["resnet18", "resnet34"]

        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64  # Initialize inplanes to match conv1 output channels

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        self.register_module("conv1", self.conv1)
        self.register_module("bn1", self.bn1)
        self.register_module("layer1", self.layer1)
        self.register_module("layer2", self.layer2)
        self.register_module("layer3", self.layer3)
        self.register_module("layer4", self.layer4)
        self.register_module("fc", self.fc)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

        layers = []
        layers.append(Block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.is_basic))
        self.inplanes = planes * self.expansion  # Update inplanes for the next block
        for _ in range(1, blocks):
            layers.append(Block(self.inplanes, planes, 1, None, self.groups, self.base_width, self.is_basic))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

    def features(self, x: torch.Tensor, encoder_depth: int = 5) -> List[torch.Tensor]:
        features = [x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)

        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(encoder_depth - 1, len(stages))):
            x = stages[i](x)
            features.append(x)
        return features

    def load_pretrained(self, pretrained_path: str):
        try:
            loaded_data = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            if isinstance(loaded_data, dict):
                pretrained_dict = loaded_data
            elif isinstance(loaded_data, torch.jit._script.RecursiveScriptModule):
                pretrained_dict = loaded_data.state_dict()
            else:
                pretrained_dict = loaded_data.state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке весов из {pretrained_path}: {str(e)}")
    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

        layers = []
        layers.append(Block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, self.is_basic))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(Block(self.inplanes, planes, 1, None, self.groups, self.base_width, self.is_basic))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)

    def features(self, x: torch.Tensor, encoder_depth: int = 5) -> List[torch.Tensor]:
        features = [x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)

        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(min(encoder_depth - 1, len(stages))):
            x = stages[i](x)
            features.append(x)
        return features

    def load_pretrained(self, pretrained_path: str):
        try:
            # Загружаем TorchScript-модель
            loaded_model = torch.jit.load(pretrained_path, map_location='cpu')

            # Извлекаем state_dict
            pretrained_dict = loaded_model.state_dict()

            # Получаем state_dict текущей модели
            model_dict = self.state_dict()

            # Фильтруем fc слой
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc.')}

            # Обновляем только совпадающие ключи
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл весов {pretrained_path} не найден.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке весов из {pretrained_path}: {str(e)}")


class Block(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, downsample: Optional[nn.Module], groups: int,
                 base_width: int, is_basic: bool):
        super().__init__()
        self.inplanes = inplanes
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        if not is_basic:
            self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
            self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
            self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.is_basic = is_basic

        self.register_module("conv1", self.conv1)
        self.register_module("bn1", self.bn1)
        self.register_module("conv2", self.conv2)
        self.register_module("bn2", self.bn2)
        if not is_basic:
            self.register_module("conv3", self.conv3)
            self.register_module("bn3", self.bn3)
        if downsample is not None:
            self.register_module("downsample", self.downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if not self.is_basic:
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class FPN(nn.Module):
    def __init__(self, num_classes: int, encoder_name: str, pretrained_path: str, encoder_depth: int = 5,
                 pyramid_channels: int = 256, segmentation_channels: int = 128, merge_policy: str = "add",
                 dropout: float = 0.2, upsampling: float = 1.0):
        super().__init__()
        encoder_params = encoder_parameters()
        if encoder_name not in encoder_params:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        encoder_info = encoder_params[encoder_name]
        if encoder_info["class_type"] != "resnet":
            raise ValueError("Only resnet encoders are supported")

        self.encoder = ResNet(encoder_info["layers"], 1000, encoder_name)
        self.encoder.load_pretrained(pretrained_path)

        self.decoder = FPNDecoder(
            encoder_info["outChannels"],
            encoder_depth,
            pyramid_channels,
            segmentation_channels,
            dropout,
            merge_policy
        )

        self.segmentation_head = SegmentationHead(
            self.decoder.out_channels,
            num_classes,
            kernel_size=1,
            upsampling=upsampling
        )

        self.register_module("encoder", self.encoder)
        self.register_module("decoder", self.decoder)
        self.register_module("segmentation_head", self.segmentation_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.features(x)
        x = self.decoder(features)
        x = self.segmentation_head(x)
        return x


class Segmentor:
    def __init__(self):
        self.width: int = 512
        self.height: int = 512
        self.list_name: List[str] = []
        self.device: torch.device = torch.device("cpu")
        self.tricks: TrainTricks = TrainTricks()
        self.fpn: Optional[FPN] = None

    def initialize(self, width: int, height: int, list_name: List[str], encoder_name: str,
                   pretrained_path: str):
        self.width = width
        self.height = height
        self.list_name = list_name

        if len(list_name) < 2:
            raise ValueError("Class number is less than 1")

        self.device = torch.device("cpu")

        self.fpn = FPN(len(list_name), encoder_name, pretrained_path)
        self.fpn.to(self.device)
        self.fpn.eval()

    def load_weight(self, path_weight: str):
        if self.fpn is None:
            raise ValueError("Модель не инициализирована. Сначала вызовите initialize().")

        try:
            # Загружаем TorchScript-модель
            loaded_model = torch.jit.load(path_weight, map_location=self.device)

            # Извлекаем state_dict
            state_dict = loaded_model.state_dict()

            # Загружаем state_dict в модель
            self.fpn.load_state_dict(state_dict)
            self.fpn.to(self.device)
            self.fpn.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл весов {path_weight} не найден.")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке весов из {path_weight}: {str(e)}")

    def predict(self, image: np.ndarray, which_class: str):
        if self.fpn is None:
            raise ValueError("Модель не инициализирована. Сначала вызовите initialize().")

        # Проверка класса
        which_class_index = -1
        for i, name in enumerate(self.list_name):
            if name == which_class:
                which_class_index = i
                break
        if which_class_index == -1:
            print(f"{which_class} не в списке имен")
            return

        # Подготовка изображения
        src_img = image.copy()
        original_height, original_width = image.shape[:2]
        image = cv2.resize(image, (self.width, self.height))

        # Преобразование в тензор
        tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        try:
            with torch.no_grad():
                output = self.fpn(tensor_image)
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return

        # Обработка выхода
        output = torch.softmax(output, dim=1).mul(255.0).byte()
        mask = output[0, which_class_index].cpu().numpy()

        # Создание результирующего изображения
        result = np.ones((self.height, self.width), dtype=np.uint8)

        # Если размер маски не совпадает, масштабируем
        if mask.shape != (self.height, self.width):
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        result[:] = mask

        # Масштабирование обратно к исходному размеру
        result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Сохранение и отображение
        cv2.imwrite("prediction.jpg", result)
        cv2.imshow("prediction", result)
        cv2.imshow("srcImage", src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

