import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader, device=device):
    model.eval()
    correct = total = 0
    loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss += F.cross_entropy(logits, y).item() * y.size(0)  # Accumulate total loss
    loss /= total  # Compute average loss
    return correct / total, loss

def evaluate_f1_and_accuracy(model, test_loader, device=device, task = "multiclass"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if task == "multiclass":
                preds = logits.argmax(dim=1)
            elif task == "binary":
                preds = torch.sigmoid(logits).squeeze()
                preds = (preds > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    return f1, accuracy

def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)  # reshape to (batch_size, channels, height*width)
        mean += images.mean(2).sum(0)  # sum over the batch and pixels
        std += images.std(2).sum(0)  # sum over the batch and pixels
    mean /= len(loader.dataset)  # divide by the total number of samples
    std /= len(loader.dataset)  # divide by the total number of samples
    return mean, std