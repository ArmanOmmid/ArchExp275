
import torch

# Seperated into another function to collect memory usage data if needed
def run_experiment(model, optimizer, criterion, dataloader, epochs, device, SAVE_WEIGHTS_FILE):
    losses = []
    dataset_size = len(dataloader.dataset)
    for e in range(epochs):
        print("===="*4, f"Epoch {e}", "===="*4)

        epoch_loss = 0
        for iter, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            model.train(True)
            optimizer.zero_grad()
            with torch.enable_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            model.train(False)

            avg_batch_loss = loss.item()
            epoch_loss += avg_batch_loss * inputs.size(0)
            print(f"BatchAvgLoss: {avg_batch_loss}")

        losses.append(epoch_loss / dataset_size)
        print("----"*16)
        print(f"Epoch: {e}")
        print(f"EpochAvgLoss: {losses[e]}")
        print(SAVE_WEIGHTS_FILE)
        model.save(SAVE_WEIGHTS_FILE)

    print("===="*4, "Done", "===="*4)
    print(SAVE_WEIGHTS_FILE)

def run_epoch(
    model,
    criterion,
    optimizer,
    dataloader,
    device,
    learn
):
    epoch_loss = 0.
    dataset_size = len(dataloader.dataset)

    for iter, (inputs, labels) in enumerate(dataloader):
        with torch.set_grad_enabled(learn):

            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            model.train(False)

            avg_batch_loss = loss.detach().item()
            epoch_loss += avg_batch_loss * batch_size

    avg_epoch_loss = epoch_loss / dataset_size
    return avg_epoch_loss
