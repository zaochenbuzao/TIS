import numpy as np
import math
from transformers import ViTForImageClassification, ViTImageProcessor, DeiTForImageClassificationWithTeacher
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision
from torch.nn.functional import interpolate
from hook import VIT_Hook, DEIT_Hook, TimmViT_Hook
from feature_extractor import Custom_feature_extractor, Timm_feature_extractor

import timm
from timm.models.vision_transformer import VisionTransformer as TimmViT


class SimilarityGraph:

    def __init__(self, model=None, device='cpu', model_type='vit', external_model=None):
        """
        Initialize SimilarityGraph.

        Args:
            model: Model type string ('vit' or 'deit') for creating internal model, or None if external_model is provided.
            device: Device to run the model on.
            model_type: Type of model to use if creating internal model.
            external_model: Optional external timm or HuggingFace model. If provided, this will be used instead of creating a new one.
        """

        self.device = torch.device(device)

        if external_model is not None:
            self.model = external_model
            self.model_type = 'external'

            if isinstance(external_model, TimmViT):
                self.hook = TimmViT_Hook(external_model)
                self.image_processor = Timm_feature_extractor(external_model, device)
            elif isinstance(external_model, ViTForImageClassification):
                self.hook = VIT_Hook(external_model)
                self.image_processor = Custom_feature_extractor(device, 'vit')
            elif isinstance(external_model, DeiTForImageClassificationWithTeacher):
                self.hook = DEIT_Hook(external_model)
                self.image_processor = Custom_feature_extractor(device, 'deit')
            else:
                raise ValueError(f"External model type {type(external_model)} not supported. Use timm.models.vision_transformer.VisionTransformer or transformers.ViTForImageClassification")
        else:
            assert model == 'deit' or model == 'vit', "Model must be 'deit' or 'vit'"

            self.model_type = model

            if model == 'vit':
                self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
                self.image_processor = Custom_feature_extractor(device, model)
                self.hook = VIT_Hook(self.model)
            else:
                self.model = DeiTForImageClassificationWithTeacher.from_pretrained(
                    'facebook/deit-base-distilled-patch16-224')
                self.image_processor = Custom_feature_extractor(device, model)
                self.hook = DEIT_Hook(self.model)

        self.model.to(self.device)


    def get_saliency(self, img_path, token_ratio, masks_layer, starting_layer = 0, label=False):
        """
        Generates a saliency heatmap for an input image based on embedding similarity.

        Args:
            img_path (str): Path to the input image file.
            token_ratio (float): The percentage of top nodes to consider for binary masking.
            label (bool or int, optional): If provided, the ground truth label for the image. If not provided,
                                          the predicted label will be used.

        Returns:
            tuple: Tuple containing the saliency heatmap (reshaped) and the image label.
        """

        torch.manual_seed(42)

        image = Image.open(img_path).convert('RGB')

        processed_image, attentions_scores, emb, predicted_label = self.classify(image, self.hook, self.image_processor)

        ground_truth_label = predicted_label if not label else torch.tensor(label)

        starting_nodes = self.get_best_cls(attentions_scores, masks_layer, starting_layer)

        multilayer = self.create_multilayer_emb(emb, starting_layer)

        num_layers = multilayer.shape[0]
        total_patches = multilayer.shape[2]

        masks_array = self.get_masks(multilayer=multilayer, token_ratio=token_ratio,  masks_layer = masks_layer, starting_nodes = starting_nodes)

        mask_tensor = torch.stack(masks_array).to(self.device)

        confidence_ground_truth_class = self.hook.classify_with_sampled_tokens(processed_image, mask_tensor,
                                                                                   ground_truth_label)

        confidence_ground_truth_class = torch.tensor(confidence_ground_truth_class).to(self.device)

        B, _ = mask_tensor.shape
        binary_mask_tensor = torch.zeros((B, total_patches), dtype=torch.int32).to(self.device)
        binary_mask_tensor.scatter_(1, mask_tensor, 1)

        heatmap_ground_truth_class = binary_mask_tensor * confidence_ground_truth_class.view(-1, 1)

        heatmap_ground_truth_class = torch.sum(heatmap_ground_truth_class, dim=0)
        coverage_bias = torch.sum(binary_mask_tensor, dim=0)
        coverage_bias = torch.where(coverage_bias > 0, coverage_bias, 1)

        heatmap_ground_truth_class = heatmap_ground_truth_class / coverage_bias
        heatmap_ground_truth_class_reshape = heatmap_ground_truth_class.reshape((14, 14))


        return heatmap_ground_truth_class_reshape.to('cpu'), ground_truth_label.item()



    def classify(self, image, model, image_processor):
        """
        Classifies an image using the specified model and image processor.

        Args:
            image: The input image (PIL Image or tensor).
            model: The hook object for the model.
            image_processor: The image processor.

        Returns:
            tuple: A tuple containing input features, embedding, and the predicted class index.
        """
        inputs = image_processor(images=image, return_tensors="pt")

        logits, attentions_scores, embeddings = model.classify_and_capture_outputs(inputs, output_attentions = True)

        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1)

        return inputs, attentions_scores, embeddings, predicted_class_idx

    def get_best_cls(self, attentions, masks_layer, starting_layer):
        """
        Select the best and worst CLS embeddings based on attention scores.

        Parameters:
        attentions (list of torch.Tensor): A list of attention score tensors.
        masks_layer (int): Number of masks.
        starting_layer (int): The layer from which to start considering attention scores.

        Returns:
        torch.Tensor: A tensor containing the indices of the selected top and worst attention scores.
        """

        att_list = []

        if self.model_type == 'external' and isinstance(self.model, TimmViT):
            for i in range(len(attentions)):
                att = attentions[i]
                att_reshaped = att.reshape(att.shape[0] * att.shape[1], att.shape[2])
                att_max = torch.max(att_reshaped, dim=1)[0]
                att_no_cls = att_max[1:]
                att_list.append(att_no_cls)
        else:
            for i in range(len(attentions)):
                att_no_head = torch.max(attentions[i][0], dim=0)[0]
                if isinstance(self.model, ViTForImageClassification):
                    att_no_head_cls = att_no_head[0, 1:]
                elif isinstance(self.model, DeiTForImageClassificationWithTeacher):
                    att_no_head_cls = att_no_head[0, 2:]
                else:
                    att_no_head_cls = att_no_head[0, 1:]
                att_list.append(att_no_head_cls)

        worst_number = int(masks_layer / 2)
        top_number = masks_layer - worst_number

        attns = torch.stack(att_list, dim=0)
        attns = attns[starting_layer:, :]

        topk_values, topk_indices = torch.topk(attns, k=top_number, dim=1, largest=True, sorted=True)
        worstk_values, worstk_indices = torch.topk(attns, k=worst_number, dim=1, largest=False, sorted=True)

        indices = torch.cat([topk_indices, worstk_indices], dim=1)

        return indices

    def get_similarity(self, embeddings):
        """
        Creates the similarity matrix of embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings weights of the nodes.
        Returns:
            torch.Tensor: adjacency matrices.
        """

        norm_embeddings = F.normalize(embeddings, p=2, dim=2)

        similarity_matrix = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2))

        return similarity_matrix


    def create_multilayer_emb(self, embeddings, starting_layer):
        """
        Creates adjacency matrices based on the similarity of the embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings weights across layers and nodes.
        Returns:
            torch.Tensor: Multilayer adjacency matrices.
        """

        embeddings_list = []

        if self.model_type == 'external' and isinstance(self.model, TimmViT):
            for i in range(len(embeddings)):
                embeddings_image = embeddings[i][0]
                embeddings_list.append(embeddings_image)
            embeddings_tensor = torch.stack(embeddings_list, dim=0)
            embeddings_tensor = embeddings_tensor[:, 1:, :].to(self.device)
        else:
            for i in range(1, len(embeddings)):
                embeddings_image = embeddings[i][0]
                embeddings_list.append(embeddings_image)

            embeddings_tensor = torch.stack(embeddings_list, dim=0)

            if isinstance(self.model, ViTForImageClassification):
                embeddings_tensor = embeddings_tensor[:, 1:, :].to(self.device)
            elif isinstance(self.model, DeiTForImageClassificationWithTeacher):
                embeddings_tensor = embeddings_tensor[:, 2:, :].to(self.device)
            else:
                embeddings_tensor = embeddings_tensor[:, 1:, :].to(self.device)

        embeddings_tensor = embeddings_tensor[starting_layer:, :, :]

        similarity_multilayer = self.get_similarity(embeddings_tensor)

        return similarity_multilayer


    def modify_image(self, operation, heatmap, image, percentage, baseline, device):
        """
        Modifies an image based on the given operation, heatmap, and baseline.

        Args:
            operation (str): The operation to perform ('deletion' or 'insertion').
            heatmap (torch.Tensor): The heatmap indicating pixel importance.
            image (dict): The image dictionary containing 'pixel_values'.
            percentage (float): The percentage of top pixels to consider for modification.
            baseline (str): The baseline image type ('black', 'blur', 'random', or 'mean').
            device: The device on which to perform the operation.

        Returns:
            torch.Tensor: The modified image tensor.
        """
        if operation not in ['deletion', 'insertion']:
            raise ValueError("Operation must be either 'deletion' or 'insertion'.")

        num_top_pixels = int(percentage * heatmap.shape[0] * heatmap.shape[1])
        top_pixels_indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_top_pixels:], heatmap.shape)

        img_tensor = image['pixel_values'].squeeze(0)
        img_tensor = img_tensor.permute(1, 2, 0)
        modified_image = np.copy(img_tensor.cpu().numpy())

        tensor_img_reshaped = img_tensor.permute(2, 0, 1)

        if baseline == "black":
            img_baseline = torch.zeros(tensor_img_reshaped.shape, dtype=bool).to(device)
        elif baseline == "blur":
            img_baseline = torchvision.transforms.functional.gaussian_blur(tensor_img_reshaped, kernel_size=[15, 15],
                                                                           sigma=[7, 7])
        elif baseline == "random":
            img_baseline = torch.randn_like(tensor_img_reshaped)
        elif baseline == "mean":
            img_baseline = torch.ones_like(tensor_img_reshaped) * tensor_img_reshaped.mean()

        if operation == 'deletion':
            darken_mask = torch.zeros(heatmap.shape, dtype=torch.bool).to(device)
            darken_mask[top_pixels_indices] = True
            modified_image = torch.where(darken_mask > 0, img_baseline, tensor_img_reshaped)

        elif operation == 'insertion':
            keep_mask = torch.zeros(heatmap.shape, dtype=torch.bool).to(device)
            keep_mask[top_pixels_indices] = True
            modified_image = torch.where(keep_mask > 0, tensor_img_reshaped, img_baseline)

        return modified_image


    def calculate_masks_layer(self, adj_matrix, masks_length, starting_node):
        """
        Create a masks on a given adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): The adjacency matrix of the graph.
        masks_length (int): The length of the masks.
        starting_node (int): The starting node for the masks.

        Returns:
        torch.Tensor: A tensor containing the sequence of nodes visited.
        """

        N = adj_matrix.size(0)

        masks = torch.full((masks_length + 1,), starting_node, dtype=torch.long)

        visited = torch.zeros(N, dtype=torch.bool)

        visited[starting_node] = True

        current_node = starting_node

        for i in range(1, masks_length + 1):
            probabilities = adj_matrix[current_node]

            probabilities[visited] = 0

            next_node = torch.max(probabilities, dim=0)[1]

            masks[i] = next_node

            visited[next_node] = True

            current_node = next_node

        return masks

    def get_masks(self, multilayer, token_ratio, masks_layer, starting_nodes):
        """
        Generate masks for every layer of the embedding matrix.

        Parameters:
        multilayer (torch.Tensor): A 3D tensor representing the multilayer network.
        token_ratio (float): Ratio to determine the size of the masks.
        masks_layer (int): Number of masks for each layer.
        starting_nodes (list of lists): A list containing lists of starting nodes.

        Returns:
        list: A list of masks generated.
        """

        masks_length = int(multilayer.shape[1] * token_ratio)

        masks = []

        for layer in range(multilayer.shape[0]):
            starting_nodes_layer = starting_nodes[layer]

            adj_matrix = multilayer[layer]

            adj_matrix.fill_diagonal_(0)

            for current_rw in range(masks_layer):
                starting_node_current_rw = starting_nodes_layer[current_rw]

                rw_mask = self.calculate_masks_layer(
                    adj_matrix.clone(),
                    masks_length,
                    starting_node_current_rw.item()
                )

                masks.append(rw_mask)

        return masks

    def get_insertion_deletion(self, patch_perc, heatmap, image, baseline, label):
        """
        Generates confidence scores for insertion and deletion for the specif baseline and every patch_perc.

        Args:
            patch_perc (list): List of patch percentages to consider.
            heatmap (torch.Tensor): Original heatmap.
            image (torch.Tensor): Original image tensor.
            baseline (str): Baseline image type ('black', 'blur', 'random', or 'mean').
            label: True label of the image.

        Returns:
            dict: Dictionary containing confidence scores for 'insertion' and 'deletion' operations.
        """

        image = self.image_processor(images=image, return_tensors="pt")

        heatmap = heatmap.reshape((1, 1, 14, 14))
        gaussian_heatmap = interpolate(heatmap, size=(224, 224), mode='nearest')
        gaussian_heatmap = gaussian_heatmap[0, 0, :, :].to('cpu').detach()

        confidences = {}

        for operation in ['insertion', 'deletion']:
            batch_modified = []
            for percentage in patch_perc:
                modified_image = self.modify_image(operation=operation, heatmap=gaussian_heatmap, image=image,
                                                   percentage=percentage / 100, baseline=baseline, device=self.device)
                batch_modified.append(modified_image)

            batch_modified = torch.stack(batch_modified, dim=0).to(self.device)
            confidences[operation] = self.predict(batch_modified, label)

        return confidences

    def predict(self, obscured_inputs, true_class_index):
        """
        Predicts the class probabilities for the true class for a list of obscured inputs.

        Args:
            obscured_inputs (torch.Tensor): Batch of obscured images.
            true_class_index (int): True class index for the original image.

        Returns:
            list: List of predicted probabilities for the true class in each obscured input.
        """
        outputs = self.model(obscured_inputs)
        if hasattr(outputs, 'logits'):
            probabilities = F.softmax(outputs.logits, dim=1)
        else:
            probabilities = F.softmax(outputs, dim=1)

        true_class_probs = probabilities[:, true_class_index]

        return true_class_probs.tolist()