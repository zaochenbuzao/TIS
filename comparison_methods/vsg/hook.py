import torch

class VIT_Hook:
    def __init__(self, model):
        self.model = model
        self.outputs = []

    def sampling_hook(self, token_indices):
        def hook(module, input, output):
            cls_token = output[:, 0:1, :]
            token_embeddings = output[:, 1:, :]
            sampled_tokens = token_embeddings[:, token_indices, :]
            new_output = torch.cat([cls_token, sampled_tokens], dim=1)
            return new_output
        return hook

    def output_hook(self, module, input, output):
        self.outputs.append(output)

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        class_probabilities = []
        for token_indices in token_indices_list:
            hook = self.model.vit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
            outputs = self.model(**inputs)
            hook.remove()
            predictions = outputs.logits.softmax(dim=-1)[0]
            true_class_probability = predictions[class_index]
            class_probabilities.append(true_class_probability)
        return class_probabilities

    def classify_and_capture_outputs(self, inputs, output_attentions = False):
        self.outputs = []
        hooks = []
        for layer in self.model.vit.encoder.layer:
            hooks.append(layer.intermediate.dense.register_forward_hook(self.output_hook))
        outputs = self.model(**inputs, output_attentions = output_attentions)
        for h in hooks:
            h.remove()
        return outputs.logits, outputs.attentions, torch.stack(self.outputs, dim = 0)



class DEIT_Hook:
    def __init__(self, model):
        self.model = model

    def sampling_hook(self, token_indices):
        def hook(module, input, output):
            cls_token = output[:, 0:1, :]
            dist_token = output[:, 1:2, :]
            token_embeddings = output[:, 2:, :]
            sampled_tokens = token_embeddings[:, token_indices, :]
            new_output = torch.cat([cls_token, dist_token, sampled_tokens], dim=1)
            return new_output
        return hook

    def output_hook(self, module, input, output):
        self.outputs.append(output)

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        class_probabilities = []
        for token_indices in token_indices_list:
            hook = self.model.deit.embeddings.dropout.register_forward_hook(self.sampling_hook(token_indices))
            outputs = self.model(**inputs)
            hook.remove()
            predictions = outputs.logits.softmax(dim=-1)[0]
            true_class_probability = predictions[class_index]
            class_probabilities.append(true_class_probability)
        return class_probabilities

    def classify_and_capture_outputs(self, inputs, output_attentions = False):
        self.outputs = []
        hooks = []
        for layer in self.model.deit.encoder.layer:
            hooks.append(layer.intermediate.dense.register_forward_hook(self.output_hook))
        outputs = self.model(**inputs, output_attentions = output_attentions)
        for h in hooks:
            h.remove()
        return outputs.logits, outputs.attentions, torch.stack(self.outputs, dim = 0)


class TimmViT_Hook:
    def __init__(self, model):
        self.model = model
        self.outputs = []
        self.attentions = []

    def sampling_hook(self, token_indices):
        def hook(module, input, output):
            cls_token = output[:, 0:1, :]
            token_embeddings = output[:, 1:, :]
            sampled_tokens = token_embeddings[:, token_indices, :]
            new_output = torch.cat([cls_token, sampled_tokens], dim=1)
            return new_output
        return hook

    def output_hook(self, module, input, output):
        self.outputs.append(output)

    def attention_hook(self, module, input, output):
        self.attentions.append(output)

    def classify_with_sampled_tokens(self, inputs, token_indices_list, class_index):
        class_probabilities = []
        original_forward = self.model.forward

        def custom_forward(x):
            return original_forward(x, output_attentions=True)

        for token_indices in token_indices_list:
            hook_handle = self.model.head.dropout.register_forward_hook(self.sampling_hook(token_indices))
            outputs = custom_forward(inputs['pixel_values'])
            hook_handle.remove()

            if hasattr(outputs, 'logits'):
                predictions = outputs.logits.softmax(dim=-1)[0]
            else:
                predictions = outputs.softmax(dim=-1)[0]
            true_class_probability = predictions[class_index]
            class_probabilities.append(true_class_probability)

        return class_probabilities

    def classify_and_capture_outputs(self, inputs, output_attentions=False):
        self.outputs = []
        self.attentions = []

        hooks = []
        for block in self.model.blocks:
            hooks.append(block.register_forward_hook(self.output_hook))
            if output_attentions:
                hooks.append(block.attn.register_forward_hook(self.attention_hook))

        outputs = self.model.forward(inputs['pixel_values'], output_attentions=output_attentions)

        for h in hooks:
            h.remove()

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        attentions = self.attentions if self.attentions else None

        if self.outputs:
            embeddings_stacked = torch.stack(self.outputs, dim=0)
        else:
            embeddings_stacked = torch.zeros(1)

        return logits, attentions, embeddings_stacked