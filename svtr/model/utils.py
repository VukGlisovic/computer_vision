import torch


def print_model_parameters(model):
    nr_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nr_non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Number of trainable/non-trainable parameters: {nr_trainable_params:,} / {nr_non_trainable_params:,}")


def save_model(model, path):
    torch.save(model, path)


def load_model(path):
    return torch.load(path)
