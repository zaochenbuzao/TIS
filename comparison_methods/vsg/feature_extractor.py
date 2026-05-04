from transformers import ViTImageProcessor


class Custom_feature_extractor:

    def __init__(self, device, model):
        if model == 'vit':
            self.preprocess = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        elif model == 'deit':
            self.preprocess = ViTImageProcessor(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

        self.device = device

    def __call__(self, images, return_tensors=True):
        return self.preprocess(images=images, return_tensors=return_tensors).to(self.device)


class Timm_feature_extractor:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.config = model.default_cfg
        self.mean = self.config['mean']
        self.std = self.config['std']

    def __call__(self, images, return_tensors=True):
        import torch
        from PIL import Image

        if isinstance(images, Image.Image):
            images = [images]

        processed = []
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                img = (img * 255).astype('uint8')
                img = Image.fromarray(img)

            img = img.resize((224, 224), Image.BILINEAR)
            img_array = np.array(img).astype(np.float32) / 255.0

            for i in range(3):
                img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]

            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            processed.append(img_tensor)

        if return_tensors:
            batch = torch.stack(processed, dim=0)
            return {'pixel_values': batch.to(self.device)}
        return processed

import numpy as np