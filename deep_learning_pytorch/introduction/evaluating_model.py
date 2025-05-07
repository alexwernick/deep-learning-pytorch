'''
Training, validation and testing

 - A dataset is typically split into three subsets:
   | Percent of data | Role                           |
   |----------------|--------------------------------|
   | Training       | 80-90%         | Adjusts model parameters    |
   | Validation     | 10-20%         | Tunes hyperparameters       |
   | Test           | 5-10%          | Evaluates final model performance |

 - Track loss and accuracy during training and validation

Calculating the loss

- For each epoch
  - sum the loss across all batches in dataloader
  - compute the mean training loss at the end of the epoch
  - compute the mean validation loss at the end of the epoch
  - Keeping tack of both helps prevent overfitting. If training loss
  continues to drop and validation loss does not or even increases
  we have overfit
  - Finally you can calculate accuracy with torch metrics
'''
import torchmetrics

# Below is just example code and is not runnable

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    training_loss = 0.0
    for inputs, labels in trainloader:
        # Run the forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Reset gradients
        
        # Calculate and sum the loss
        training_loss += loss.item()
    
    epoch_loss = training_loss / len(trainloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")



    # We then calculate validation loss for each epoch
    validation_loss = 0.0
    model.eval() # Put model in evaluation mode (more efficient)

    with torch.no_grad(): # Disable gradients for efficiency
        for inputs, labels in validationloader:
            # Run the forward pass
            outputs = model(inputs)
            # Calculate the loss
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

    epoch_loss = validation_loss / len(validationloader) # Compute mean loss
    model.train() # Switch back to training mode


    

    # Create accuracy metric for each epoch
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)

    for features, labels in dataloader:
        outputs = model(features) # Forward pass
        # Compute batch accuracy (keeping argmax for one-hot labels)
        # using labels.argmax(dim=-1) we select the class with highest probability
        metric.update(outputs, labels.argmax(dim=-1))

    # Compute accuracy over the whole epoch
    accuracy = metric.compute()

    # Reset metric for the next epoch
    metric.reset()