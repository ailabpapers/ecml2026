import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import copy
import logging
import matplotlib.pyplot as plt
import math
import time
import statistics
import numpy as np
from datetime import datetime
import os, json, requests

class simpleNN(torch.nn.Module):
  def __init__(self):
      super(simpleNN, self).__init__()

      self.linear1 = torch.nn.Linear(8, 16)
      self.activation = torch.nn.ReLU()
      self.linear2 = torch.nn.Linear(16, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

model = simpleNN()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
class CSVDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

def dftoDst(df):
  feature_colmn = [col for col in df.columns if col not in ['DATE','STATION', 'FSP']]
  features = df[feature_colmn].values
  features_normalized = (features - features.min()) / (features.max() - features.min())
  targets = df['FSP'].values
  targets_normalized = (targets - targets.min()) / (targets.max() - targets.min())

  features_tensor = torch.tensor(features_normalized, dtype=torch.float)
  targets_tensor = torch.tensor(targets_normalized, dtype=torch.float)

  csv_data = CSVDataset(features_tensor, targets_tensor)
  return csv_data

df = pd.read_csv("../../../Datasets/Air Quality/AQ_DomainIL.csv", index_col=0)
df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))
df["DATE"] = pd.to_datetime(df['DATE'], format="%d/%m/%Y")
df = df.sort_values("DATE")

stations = ['YUEN LONG', 'TUNG CHUNG', 'TSUEN WAN', 'TAP MUN', 'CENTRAL', 'CAUSEWAY BAY', 'MONG KOK', 'TUEN MUN' ,'TSEUNG KWAN O']

task = {}
batch_size = 8

for i, station in enumerate(stations):
    split_date1 = pd.to_datetime("31-12-2020", dayfirst=True)
    split_date2 = pd.to_datetime("01-01-2024", dayfirst=True)
    
    df_curr = df.loc[df['STATION'] == station]

    train = df_curr[df_curr['DATE'] < split_date1]
    test = df_curr[(df_curr['DATE'] >= split_date1) & (df_curr['DATE'] <= split_date2)]
    val = df_curr[df_curr['DATE'] > split_date2]
    
    traindfDst = dftoDst(train)
    trainDL = DataLoader(traindfDst, batch_size=batch_size)
    
    testdfDst = dftoDst(test)
    testDL = DataLoader(testdfDst)
    
    valdfDst = dftoDst(val)
    valDL = DataLoader(valdfDst)

    task[i] = {
        "train": trainDL,
        "test": testDL,
        "val": valDL
    }
    
learning_rate = 0.001
loss_fn = nn.MSELoss()
min_mse = 0.5
novelty_buffer = []
familiarity_buffer = []

def get_fisher_approx(dataloader, model, loss_fn, ctxt_count):
  F_sq = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
  F_mean = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
  num_samples = 0

  model.eval()
  for x, y in dataloader:
    x = x.squeeze()
    model.zero_grad()
    pred = model(x)
    if pred.dim() > 1:
      pred = pred.squeeze()
    loss = loss_fn(pred, y)
    loss.backward()

    batch_size = x.size(0)
    num_samples += batch_size

    for name, param in model.named_parameters():
      if param.grad is not None:
        grad = param.grad.data
        F_sq[name] += (grad ** 2) * batch_size
        F_mean[name] += grad * batch_size

  F_var = {}
  for name in F_sq:
    mean_grad = F_mean[name] / num_samples
    mean_grad_sq = F_sq[name] / num_samples
    F_var[name] = mean_grad_sq - mean_grad ** 2

  total_variance = sum(torch.sum(var_tensor) for var_tensor in F_var.values())
  return total_variance

def update_buffer(buffer_type, features, targets, novelty_buffer_size):
  global familiarity_buffer, novelty_buffer
  if buffer_type == 'novelty':
    new_samples = list(zip(features, targets))
    for sample in new_samples:
      if len(novelty_buffer) <= novelty_buffer_size:
        novelty_buffer.append(sample)
  elif buffer_type == 'familiarity':
    new_samples = list(zip(features, targets))
    for sample in new_samples:
      familiarity_buffer.append(sample)
  else:
    assert False, "Bad buffer type passed"

def train(dataloader, val_data, model, loss_fn, optimizer, curr_context, prev_context, clear_alpha, novelty_buffer_size, ewc_lambda):
  global min_mse, novelty_buffer, familiarity_buffer
  size = len(dataloader)
  model.train()
  correct = 0
  running_loss = 0
  threshold = clear_alpha * min_mse

  if curr_context > 1:
    fisher_approx = get_fisher_approx(prev_context, model, loss_fn, curr_context)

  for i, (X, y) in enumerate(dataloader):
    X = X.squeeze()
    optimizer.zero_grad()

    pred = model(X)
    if pred.dim() > 1:
      pred = pred.squeeze()
    loss = loss_fn(pred, y)

    if len(novelty_buffer) >= novelty_buffer_size:
      # print("Retraining")
      for idx in range(len(novelty_buffer)):
        key, true_val = novelty_buffer[idx]
        true_val = true_val.view(-1)
        optimizer.zero_grad()
        new_pred = model(key).view(-1)
        new_loss = loss_fn(new_pred, true_val)
        new_loss.backward()
        optimizer.step()
      model.eval()
      with torch.no_grad():
        for idx in range(len(familiarity_buffer)):
          key, true_val = familiarity_buffer[idx]
          true_val = true_val.view(-1)
          new_pred = model(key).view(-1)
          fam_loss = loss_fn(new_pred, true_val).item()
          if fam_loss < min_mse:
            min_mse = fam_loss
            threshold = clear_alpha * min_mse
      model.train()
      novelty_buffer.clear()
      familiarity_buffer.clear()
      continue

    if(loss.mean() > threshold):
      if X.dim() > 1:
        update_buffer('novelty', X, y, novelty_buffer_size)
      else:
        novelty_buffer.append((X, y))
    else:
      if X.dim() > 1:
        update_buffer('familiarity', X, y, None)
      else:
        familiarity_buffer.append((X, y))

    in_range = (torch.abs((pred - y) / y)) <= 0.05
    correct += ((in_range).sum().item())

    if curr_context > 1:
      loss += ewc_lambda * 0.5 * fisher_approx
      
    running_loss += loss.item()

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      loss, current = loss.item(), i * batch_size + len(X)
  #     print(f"loss: {loss:>3f}  [{current:>3d}/{size:>3d}]")
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for data, target in val_data:
      data = data.squeeze()
      output = model(data)
      loss = loss_fn(output, target)
      val_loss += loss.item()
  val_loss /= len(val_data)
  
  # print(f"Train Loss: {(loss/size):>0.5f}, Val Loss: {(val_loss):>0.5f}")

  early_stopping(val_loss, model)
  return running_loss/size, correct*100/size

def test(dataloader, model, loss_fn, ctxt_count):
  model.eval()
  size = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X = X.squeeze()
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      in_range = (torch.abs((pred - y) / y)) <= 0.05
      correct += in_range.type(torch.float).sum().item()

    test_loss /= size
    test_acc = (correct/size) * 100
  return test_loss, test_acc

testing_ctxt = [task[i]["test"] for i in range(len(task))]

training_ctxt = [task[i]["train"] for i in range(len(task))]

val_ctxt = [task[i]["val"] for i in range(len(task))]

def test_all(model, ctxt_count):
    list_loss = []
    for test_ctxt in testing_ctxt:
        test_loss_ctxt, acc = test(test_ctxt, model, loss_fn, ctxt_count)
        list_loss.append(test_loss_ctxt)
    return list_loss

def getFR(saved_errors_list):
    forgetting_ratio = 0
    for j in range(len(saved_errors_list)):
        comp = saved_errors_list[len(saved_errors_list)-1][j] - saved_errors_list[j][j]
        forgetting_ratio += ((max(0, comp))/saved_errors_list[j][j])
    print(f"FR = {(forgetting_ratio/len(saved_errors_list)):>6f}")
    return forgetting_ratio/len(saved_errors_list)

def get_loverall(saved_errors_list):
    error = 0
    for k in range(len(saved_errors_list)):
        comp = saved_errors_list[len(saved_errors_list)-1][k] - saved_errors_list[k][k]
        error += comp
    return error/len(saved_errors_list)

runs = 20
epochs = 100

all_total_train_times = []
all_during_averages = []
all_after_averages = []
all_saved_error_lists = []
all_overall_averages = []

for run in range(runs):
    print(f"\n--- Run {run + 1} ---")

    # Reinitialize model and optimizer each run
    model = simpleNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ctxt_count = 1
    saved_errors_list = []
    total_train_time = 0
    during_averages = 0
    after_averages = 0
    fr_saved = []
    novelty_buffer.clear()
    familiarity_buffer.clear()

    for train_ctxt in training_ctxt:
        early_stopping = EarlyStopping(patience=10, delta=0.0003)
        val_idx = training_ctxt.index(train_ctxt)
        start_train_time = time.time()
        for t in range(epochs):
            if ctxt_count == 1:
                train(train_ctxt, val_ctxt[val_idx], model, loss_fn, optimizer, ctxt_count, None, 0.5, 50, 2)
                if early_stopping.early_stop:
                    print(f"epoch: {t}, Early stopping")
                    break
            else:
                train(train_ctxt, val_ctxt[val_idx], model, loss_fn, optimizer, ctxt_count, training_ctxt[ctxt_count-2], 0.5, 50, 2)
                if early_stopping.early_stop:
                    print(f"epoch: {t}, Early stopping")
                    break
        end_train_time = time.time()
        total_train_time += end_train_time - start_train_time
        
        early_stopping.load_best_model(model)

        test_on_train_ctxt = test_all(model, ctxt_count)
        during_averages += test_on_train_ctxt[ctxt_count - 1]
        saved_errors_list.append(test_on_train_ctxt)

        print(f"After training on C{ctxt_count}:")
        print("".join([f"C{itera+1}: {val:>5f}, " for itera, val in enumerate(test_on_train_ctxt[:len(training_ctxt)])])[:-2])
        ctxt_count += 1
        fr_saved.append(getFR(saved_errors_list))

    print(f"Total train time: {total_train_time:.2f} sec")
    print(f"During average: {during_averages / len(training_ctxt):.4f}")
    after_averages = statistics.mean(saved_errors_list[-1])
    print(f"After average: {after_averages:.4f}")
    overall_avg = get_loverall(saved_errors_list)
    print(f"Overall average: {overall_avg:.4f}")

    # Save results of this run
    all_total_train_times.append(total_train_time)
    all_during_averages.append(during_averages/len(training_ctxt))
    all_after_averages.append(after_averages)
    all_overall_averages.append(overall_avg)
    all_saved_error_lists.append(saved_errors_list)

# Final summary
print("\n=== Final Averages ===")
print(f"Avg Total Train Time: {statistics.mean(all_total_train_times):.4f}")
print(f"Avg During: {statistics.mean(all_during_averages):.5f}")
print(f"Std Dev During: {statistics.stdev(all_during_averages):.5f}")
print(f"Avg After: {statistics.mean(all_after_averages):.5f}")
print(f"Std Dev After: {statistics.stdev(all_after_averages):.5f}")
print(f"Avg Overall: {statistics.mean(all_overall_averages):.5f}")
print(f"Std Dev Overall: {statistics.stdev(all_overall_averages):.5f}")

error_array = np.array(all_saved_error_lists)
average_errors = np.mean(error_array, axis=0)

def getFRadjusted(saved_errors_list):
    forgetting_ratio = 0
    for j in range(len(saved_errors_list)):
        comp = saved_errors_list[len(saved_errors_list)-1][j] - saved_errors_list[j][j]
        forgetting_ratio += ((max(0, comp))/saved_errors_list[j][j])
    print(f"{(forgetting_ratio/len(saved_errors_list)):>6f}")
    return forgetting_ratio/len(saved_errors_list)

final_FR = []
for ac in range(len(average_errors)):
    subset = average_errors[:ac+1]
    fr = getFRadjusted(subset)
    final_FR.append(fr)