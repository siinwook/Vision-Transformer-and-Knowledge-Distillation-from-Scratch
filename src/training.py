import torch
from torch import nn


def train_classifier(model, train_dataloader, train_dataset, device, epochs, lr=5e-4):
  model.train()

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=35, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[5])

  train_loss = []

  for epoch in range(epochs):
    epoch_loss = 0

    for batch, (x_train,y_train) in enumerate(train_dataloader):
      x_train, y_train = x_train.to(device), y_train.to(device)

      logits = model(x_train)

      optimizer.zero_grad()
      loss = loss_fn(logits, y_train)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    scheduler.step()
    print(f"epoch {epoch+1} loss: {epoch_loss / len(train_dataloader)}")
    train_loss.append(epoch_loss / len(train_dataloader))

  return train_loss


def train_deit(model, teacher, train_dataloader, train_dataset, device, epochs, lr=5e-4):
  model.train()
  teacher.eval()

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
  scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=35, eta_min=1e-5)
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler_linear,scheduler_cos], milestones=[5])

  train_loss = []

  for epoch in range(epochs):
    epoch_loss = 0

    for batch, (x_train,y_train) in enumerate(train_dataloader):
      x_train, y_train = x_train.to(device), y_train.to(device)

      x_cls_logits, x_dist_logits = model(x_train)
      with torch.no_grad():
        teacher_logits = teacher(x_train).argmax(dim=-1)

      optimizer.zero_grad()
      loss = 0.5 * loss_fn(x_cls_logits, y_train) + 0.5 * loss_fn(x_dist_logits, teacher_logits)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    scheduler.step()
    print(f"epoch {epoch+1} loss: {epoch_loss / len(train_dataloader)}")
    train_loss.append(epoch_loss / len(train_dataloader))

  return train_loss


def train_classifier_adam_drop(model, train_dataloader, train_dataset, device, epochs=[10,40], lrs=[1e-3,1e-4]):
  loss_fn = nn.CrossEntropyLoss()

  train_loss = []
  train_acc = []

  for i in range(len(epochs)):
    optimizer = torch.optim.Adam(model.parameters(), lr = lrs[i])

    for epoch in range(epochs[i]):
      model.train()
      epoch_loss = 0
      epoch_acc = 0

      for batch, (x_train,y_train) in enumerate(train_dataloader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        logits = model(x_train)

        optimizer.zero_grad()
        loss = loss_fn(logits,y_train)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += sum((logits.argmax(dim=-1) == y_train))

      print(f"epoch {sum(epochs[:i])+epoch+1} loss: {epoch_loss / len(train_dataloader)} acc: {epoch_acc / len(train_dataset)}")
      train_acc.append((epoch_acc / len(train_dataset)).item())
      train_loss.append(epoch_loss / len(train_dataloader))

  return train_loss, train_acc
