import torch
import pytest

from src.models.factory import build_model, SUPPORTED_MODELS
from src.models.resnet import ResNet, BasicBlock, Bottleneck, build_resnet
from src.models.simple_cnn import SimpleCNN


def _make_cfg(model_name="simple_cnn", num_classes=10):
    return {
        "model": {
            "name": model_name,
            "num_classes": num_classes,
            "pretrained": False,
            "dropout": 0.0,
        }
    }


class TestSimpleCNN:
    def test_output_shape(self):
        model = SimpleCNN(num_classes=10)
        out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)

    def test_factory_returns_correct_type(self):
        model = build_model(_make_cfg("simple_cnn", 10))
        assert isinstance(model, SimpleCNN)


class TestResNet:
    def test_resnet18_output_shape(self):
        model = build_resnet("resnet18", num_classes=10)
        assert isinstance(model, ResNet)
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 10)

    def test_resnet50_output_shape(self):
        model = build_resnet("resnet50", num_classes=100)
        out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 100)

    def test_bottleneck_expansion(self):
        model = build_resnet("resnet50", num_classes=10)
        assert model.layer1[-1].conv3.out_channels == 256
        assert model.layer2[-1].conv3.out_channels == 512
        assert model.layer3[-1].conv3.out_channels == 1024
        assert model.layer4[-1].conv3.out_channels == 2048

    def test_layer_counts(self):
        model = build_resnet("resnet50", num_classes=10)
        assert len(model.layer1) == 3
        assert len(model.layer2) == 4
        assert len(model.layer3) == 6
        assert len(model.layer4) == 3

    def test_head_dimensions(self):
        model = build_resnet("resnet50", num_classes=42)
        fc_layer = model.head[-1]
        assert fc_layer.in_features == 2048
        assert fc_layer.out_features == 42

    def test_resnet34_uses_basic_block(self):
        model = build_resnet("resnet34", num_classes=10)
        assert isinstance(model.layer1[0], BasicBlock)

    def test_resnet50_uses_bottleneck(self):
        model = build_resnet("resnet50", num_classes=10)
        assert isinstance(model.layer1[0], Bottleneck)


class TestFactory:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_model(_make_cfg("unknown_model", 10))

    def test_pretrained_raises(self):
        with pytest.raises(NotImplementedError):
            build_resnet("resnet50", pretrained=True)

    def test_supported_models_list(self):
        assert "resnet50" in SUPPORTED_MODELS
        assert "simple_cnn" in SUPPORTED_MODELS
