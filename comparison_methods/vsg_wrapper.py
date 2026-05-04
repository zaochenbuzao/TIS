import torch
import sys
from PIL import Image
import numpy as np
import tempfile
import os

sys.path.append("comparison_methods/vsg")

from .vsg.similarity_graph import SimilarityGraph


class VSGWrapper():
    """
    Wrapper for VSG method: Wrap the method to allow similar usage in scripts.
    Supports both external models (timm or HuggingFace) and internal pretrained models.
    """
    def __init__(self, model, model_type='vit', device='cuda', token_ratio=0.5, masks_layers=4, starting_layer=0, **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations (timm or HuggingFace model)
        :param model_type: type of internal model to use if no external model provided ('vit' or 'deit')
        :param device: device to run the model on
        :param token_ratio: percentage of top nodes to consider for binary masking
        :param masks_layers: number of masks
        :param starting_layer: starting layer for the method
        """
        self.device = device
        self.token_ratio = token_ratio
        self.masks_layers = masks_layers
        self.starting_layer = starting_layer

        self.vsg = SimilarityGraph(
            model=None,
            device=device,
            model_type=model_type,
            external_model=model
        )

        self.mean = model.default_cfg.get('mean', [0.485, 0.456, 0.406])
        self.std = model.default_cfg.get('std', [0.229, 0.224, 0.225])

    def _tensor_to_pil(self, x):
        """
        Convert a normalized tensor back to PIL Image for VSG processing.

        Args:
            x: normalized tensor of shape (3, H, W) or (B, 3, H, W)

        Returns:
            PIL Image
        """
        if x.dim() == 4:
            x = x[0]

        x = x.cpu().numpy()

        if x.shape[0] == 3:
            x = x.transpose(1, 2, 0)

        x = x * np.array(self.std) + np.array(self.mean)
        x = (np.clip(x, 0, 1) * 255).astype('uint8')
        return Image.fromarray(x)

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor (normalized)
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (14, 14)
        """
        with torch.no_grad():
            pil_image = self._tensor_to_pil(x)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name

            try:
                pil_image.save(temp_path)
                saliency_map, _ = self.vsg.get_saliency(temp_path, self.token_ratio, self.masks_layers, self.starting_layer, class_idx)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            return saliency_map