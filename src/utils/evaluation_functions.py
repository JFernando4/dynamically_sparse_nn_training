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
