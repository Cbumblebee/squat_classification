import torch
from torch import nn
from neural_network import model
from fetch_data import train_loader, test_loader

learning_rate = 1e-3
batch_size = 32
epochs = 50
best_accuracy = 0.0 # in the iteration phase, saving the best accuracy to minimize overfitting.

def train_loop(dataloader, train_model, train_loss_fn, train_optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    train_model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = train_model(X)
        loss = train_loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        train_optimizer.step()
        train_optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, test_model, test_loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    test_model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = test_model(X)
            test_loss += test_loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

loss_fn = nn.CrossEntropyLoss()
optimizerSGD = torch.optim.SGD(model.parameters(), lr=learning_rate) #TODO: explain why ADAM
optimizerADAM = torch.optim.Adam(model.parameters(), lr=learning_rate) #TODO: explain why ADAM

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizerADAM)
    current_acc = test_loop(test_loader, model, loss_fn)
    if current_acc > best_accuracy:
        best_accuracy = current_acc
        # save the model
        torch.save(model.state_dict(), f"model_adam_lr1e-3.pth")
        print(f"--> New accuracy record: {100 * best_accuracy:.1f}% achieved in epoch {t+1}\n")
print("Done!")


# load the model - later
# model.load_state_dict(torch.load(f"name.pth"))
# model.eval() ?