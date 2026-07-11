import torch
from torch import nn


def test_classifier(model, test_dataloader, test_dataset, device):
  with torch.no_grad():
    model.eval()
    epoch_acc = 0

    for batch, (x_test,y_test) in enumerate(test_dataloader):
      x_test, y_test = x_test.to(device), y_test.to(device)
      logits = model(x_test)
      epoch_acc += sum(logits.argmax(dim=-1) == y_test)

    test_acc = epoch_acc / len(test_dataset)
    print(f"test acc: {test_acc}")

  return test_acc


def test_deit(model, test_dataloader, test_dataset, device):
  with torch.no_grad():
    model.eval()
    deit_epoch_acc = 0
    cls_epoch_acc = 0
    dist_epoch_acc = 0

    softmax = nn.functional.softmax

    for batch, (x_test,y_test) in enumerate(test_dataloader):
      x_test, y_test = x_test.to(device), y_test.to(device)

      x_cls_logits, x_dist_logits = model(x_test)

      deit_epoch_acc += sum((softmax(x_cls_logits, dim=-1) + softmax(x_dist_logits, dim=-1)).argmax(dim=-1) == y_test) # late fusion
      cls_epoch_acc += sum(x_cls_logits.argmax(dim=-1) == y_test)
      dist_epoch_acc += sum(x_dist_logits.argmax(dim=-1) == y_test)

    deit_test_acc = deit_epoch_acc / len(test_dataset)
    cls_test_acc = cls_epoch_acc / len(test_dataset)
    dist_test_acc = dist_epoch_acc / len(test_dataset)

    print(f"DeiT test acc: {deit_test_acc}")
    print(f"CLS Token test acc: {cls_test_acc}")
    print(f"DIST Token test acc: {dist_test_acc}")

  return deit_test_acc, cls_test_acc, dist_test_acc
