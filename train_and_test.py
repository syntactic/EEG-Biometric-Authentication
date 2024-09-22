import torch as th

if th.backends.mps.is_available():
    device = th.device("mps")

def train_and_test_model(model, train_loader, test_loader, optimizer, criterion, epochs=500, verbose=True):
    training_losses = []
    testing_losses = []
    testing_accuracies = []
    model.train()
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if verbose and i % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
        training_losses.append(epoch_loss/len(train_loader))
        test_loss, accuracy = test_model(model, test_loader, criterion, verbose)
        testing_losses.append(test_loss)
        testing_accuracies.append(accuracy)
    return training_losses, testing_losses, testing_accuracies

def test_model(model, test_loader, criterion, verbose=True):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    accuracy = 0
    with th.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += criterion(y_pred, y).item()
            _, predicted = th.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        test_loss /= len(test_loader)
    accuracy = correct / total
    if verbose:
        print(f'Accuracy: {100 * accuracy}')
    return test_loss, accuracy
