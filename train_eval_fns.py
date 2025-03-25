import torch
import torch.nn as nn
import torch.optim as optim
import csv

# Train model
def train_model(model, train_loader, val_loader, device, criterion, lr=0.001, save=False):

    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr)

    if save:
        # Create CSV file and write header
        csv_filename = f"{model.__class__.__name__.lower()}_metrics.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 
                            'val_loss', 'val_accuracy'])

    # Training Loop with validation
    num_epochs = 20
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase TODO update this
        for inputs, labels, bboxes in train_loader:
            inputs, labels, bboxes = inputs.to(device), labels.to(device), bboxes.to(device)


        '''
        # Training phase OLD: ASSN 3
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accumulate training metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        ''' 
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        #TODO update this
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        if save:
            # Save to CSV
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1,
                    f"{avg_train_loss:.4f}",
                    f"{train_accuracy:.2f}",
                    f"{avg_val_loss:.4f}",
                    f"{val_accuracy:.2f}"
                ])
        
        print(f"{model.__class__.__name__.lower()} Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

    print('Training finished')
    return model

# Testing Function (works for both validation and test sets)
def evaluate_model(model, loader, device, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    #TODO update this
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = test_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy