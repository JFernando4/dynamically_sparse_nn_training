import torch

@torch.no_grad()
def compute_accuracy_from_batch(predictions: torch.Tensor, labels: torch.Tensor):
    """
    Computes accuracy based on a batch of predictions and labels

    Args:
        predictions: tensor of shape (n, c) where n is the batch size and c is the number of classes
        labels: tensor of same shape as predictions

    Return:
        Scalar tensor corresponding to the accuracy, a float between 0 and 1
    """
    return torch.mean((predictions.argmax(axis=1) == labels.argmax(axis=1)).to(torch.float32))

def compute_average_gradient_magnitude(model: torch.nn.Module) -> float:
    """
    computes the average gradient magnitude of a network
    """
    grad_magnitude_summ = 0.0
    total_params = 0.0

    for p in model.parameters():
        assert p.grad is not None
        grad_magnitude_summ += p.grad.abs().sum()
        total_params += p.numel()

    return float(grad_magnitude_summ / total_params)


