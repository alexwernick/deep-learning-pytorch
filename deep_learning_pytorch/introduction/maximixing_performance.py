'''
Steps to maximize performance

1. Overfit the training set
  - Can we solve the problem?
  - Set a performance baseline
2. Reduce overfitting
  - Increase performance on the validation set
3. Fine-tune the hyperparameters
  - Achieve best possible performance
'''

# Step 1: overfit training set
# - modify training loop to overfit a single data point
#   - should quickly reach 1.0 accuracy and 0 loss
# - Then scale up to entire training set
#  - Keep default hyperparameters
features, labels = next(iter(dataloader))
for i in range(1000):
    outputs = model(features)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Step 2: reduce overfitting
# - Goal: maximize the validation accuracy
# - Experiment with:
#   - Dropout
#   - Data augmentation
#   - Weight decay
#   - Reducing model capacity
# - Keep track of each hyperparameter and validation accuracy

# Step 3: fine-tine hyperparameters
# Can use grid search
for factor in range(2, 6):
    lr = 10 ** -factor 
# Can use random search
factor = np.random.uniform(2, 6)
lr = 10 ** -factor