import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import copy
import logging
import matplotlib.pyplot as plt
import random
import math
import time
import statistics
import numpy as np
from datetime import datetime
import os, json, requests

class simpleNN(torch.nn.Module):
  def __init__(self):
      super(simpleNN, self).__init__()

      self.linear1 = torch.nn.Linear(22, 44)
      self.activation = torch.nn.ReLU()
      self.linear2 = torch.nn.Linear(44, 1)

  def forward(self, x, task_id):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

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
  feature_colmn = [col for col in df.columns if col not in ['Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI','DNI', 'GHI', 'City', 'Global Horizontal UV Irradiance (280-400nm)', 'Global Horizontal UV Irradiance (295-385nm)', 'datetime']]
  features = df[feature_colmn].values
  features_normalized = (features - features.min()) / (features.max() - features.min())
  targets = df['GHI'].values
  targets_normalized = (targets - targets.min()) / (targets.max() - targets.min())

  features_tensor = torch.tensor(features_normalized, dtype=torch.float)
  targets_tensor = torch.tensor(targets_normalized, dtype=torch.float)

  csv_data = CSVDataset(features_tensor, targets_tensor)
  return csv_data

df = pd.read_csv("../../../Datasets/Solar Radiation/nsrdb_final.csv")
df.columns = df.columns.str.strip()
df["datetime"] = pd.to_datetime(
    df[["Year", "Month", "Day", "Hour", "Minute"]],
    errors="raise"
)
cities = ["Albany", "Annapolis", "Atlanta", "Augusta", "Austin", "Baton Rouge", "Bismarck", "Boise", "Boston", "Carson City", "Charleston", "Cheyenne", "Columbia", "Columbus", "Concord", "Denver", "Des Moines", "Dover", "Frankfort", "Harrisburg", "Hartford", "Helena", "Honolulu", "Indianapolis", "Jackson", "Jefferson City", "Juneau", "Lansing", "Lincoln", "Little Rock", "Madison", "Montgomery", "Montpelier", "Nashville", "Oklahoma City", "Olympia", "Phoenix", "Pierre", "Providence", "Raleigh", "Richmond", "Sacramento", "Saint Paul", "Salem", "Salt Lake City", "Santa Fe", "Springfield", "Tallahassee", "Topeka", "Trenton"]

task = {}
batch_size = 32

for i, city in enumerate(cities):
    df_curr = df.loc[df['City'] == city.lower()]
    
    split_date1 = pd.Timestamp("2023-07-01")
    split_date2 = pd.Timestamp("2024-07-01")

    train = df_curr[df_curr['datetime'] < split_date1]
    test = df_curr[(df_curr['datetime'] >= split_date1) & (df_curr['datetime'] < split_date2)]
    val = df_curr[df_curr['datetime'] >= split_date2]
    
    traindfDst = dftoDst(train)
    trainDL = DataLoader(traindfDst, batch_size=batch_size)
    
    testdfDst = dftoDst(test)
    testDL = DataLoader(testdfDst, batch_size=batch_size)
    
    valdfDst = dftoDst(val)
    valDL = DataLoader(valdfDst, batch_size=batch_size)

    task[i] = {
        "train": trainDL,
        "test": testDL,
        "val": valDL
    }
    
learning_rate = 0.001
reservoir_buffer = []
smote_buffer = []

def update_reservoir_buffer(features, logit, reservoir_buffer_length):
  global reservoir_buffer
  eject = None
  if features.dim() > 1:
    new_samples = list(zip(features, logit))
    for sample in new_samples:
      if reservoir_buffer_length > len(reservoir_buffer):
        reservoir_buffer.append(sample)
      else:
        random_key = random.randrange(len(reservoir_buffer))
        eject = reservoir_buffer[random_key]
        reservoir_buffer[random_key] = sample
  else:
    if reservoir_buffer_length > len(reservoir_buffer):
      reservoir_buffer.append((features, logit))
    else:
      random_key = random.randrange(len(reservoir_buffer))
      eject = reservoir_buffer[random_key]
      reservoir_buffer[random_key] = (features, logit)
  return eject
  
def update_syn_buffer(features, logit, reservoir_buffer_length):
  global smote_buffer
  if reservoir_buffer_length > len(smote_buffer):
    smote_buffer.append((features, logit))
  else:
    random_key = random.randrange(len(smote_buffer))
    smote_buffer[random_key] = (features, logit)

def get_buffer_sample():
  global reservoir_buffer
  random_key = random.randrange(len(reservoir_buffer))
  feat, logit = reservoir_buffer[random_key]
  return feat, logit

def get_syn_sample():
  global smote_buffer
  random_key = random.randrange(len(smote_buffer))
  feat, logit = smote_buffer[random_key]
  return feat, logit

def euc_dist(a, b):
  sum_vals = 0
  for i in range(len(a)):
      inter = (a[i] - b[i])**2
      sum_vals += inter

  return torch.sqrt(sum_vals)
  
def convex_combination(case, x):
  new = case.clone()
  
  diff = x - case
  rand_factor = torch.rand(case.size())
  new = case + rand_factor * diff

  return new

def oversample_buffer(reservoir_buffer_length, eject):
  global reservoir_buffer
    
  random_key1 = random.randrange(len(reservoir_buffer))
  random_key2 = random.randrange(len(reservoir_buffer))
  sample1 = reservoir_buffer[random_key1]
  sample2 = reservoir_buffer[random_key2]
  syn_feat = convex_combination(sample1[0], sample2[0])
  
  d1 = euc_dist(syn_feat, sample1[0])
  rtt1 = sample1[1].clone()
  d2 = euc_dist(sample2, syn_feat)
  rtt2 = sample2[1].clone()
  syn_rtt = (d2 * rtt1 + d1 * rtt2) / (d1 + d2)
  syn_rtt = syn_rtt[:1]

  update_syn_buffer(syn_feat, syn_rtt, reservoir_buffer_length)
    
  if len(reservoir_buffer) > reservoir_buffer_length:
    print(f"lengths: reservoir: {len(reservoir_buffer)}")

def train(dataloader, val_data, model, loss_fn, optimizer, cmr_alpha, cmr_beta, reservoir_buffer_length, ctxt_count):
  global reservoir_buffer, smote_buffer
  size = len(dataloader)
  model.train()
  correct = 0
  running_loss = 0

  for i, (X, y) in enumerate(dataloader):
    X = X.squeeze()
    optimizer.zero_grad()

    pred = model(X, ctxt_count)
    eject = update_reservoir_buffer(X, pred.detach(), reservoir_buffer_length)
    
    if len(reservoir_buffer) > 1:
      oversample_buffer(reservoir_buffer_length, eject)
    
    if X.dim() > 1:
      pred = pred.squeeze(-1)
    loss = loss_fn(pred, y)

    in_range = (torch.abs((pred - y) / y)) <= 0.05
    correct += ((in_range).sum().item())

    feat1, saved_logit1 = get_buffer_sample()
    model.eval()
    buffer_pred1 = model(feat1, ctxt_count)
    if len(reservoir_buffer) > 1:
      feat2, saved_logit2 = get_syn_sample()
      buffer_pred2 = model(feat2, ctxt_count)
      loss += cmr_alpha * loss_fn(saved_logit1, buffer_pred1) + cmr_beta * loss_fn(saved_logit2, buffer_pred2)
    else:
      alpha_var = torch.sum((saved_logit1 - buffer_pred1)**2)
      
      loss += cmr_alpha * alpha_var
    model.train()
    running_loss += loss.item()

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      loss_log, current = loss.item(), i * batch_size + len(X)
  #     print(f"loss: {loss_log:>3f}  [{current:>3d}/{size:>3d}]")

  model.eval()
  val_loss = 0
  with torch.no_grad():
    for data, target in val_data:
      data = data.squeeze()
      output = model(data, ctxt_count)
      output = output.squeeze(-1)
      loss = loss_fn(output, target)
      val_loss += loss.item()
  val_loss /= len(val_data)

  early_stopping(val_loss, model)
  return running_loss/size, correct*100/size

def test(dataloader, model, loss_fn, ctxt_count):
  model.eval()
  size = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X = X.squeeze()
      pred = model(X, ctxt_count)
      pred = pred.squeeze(-1)
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
all_overall_averages = []
all_saved_error_lists = []
test_on_train_ctxt = []

for run in range(runs):
    print(f"\n--- Run {run + 1} ---")

    # Reinitialize model and optimizer each run
    model = simpleNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    ctxt_count = 0
    saved_errors_list = []
    total_train_time = 0
    during_averages = 0
    after_averages = 0
    fr_saved = []
    
    reservoir_buffer.clear()
    smote_buffer.clear()

    for train_ctxt in training_ctxt:
        early_stopping = EarlyStopping(patience=10, delta=0.0003)
        val_idx = training_ctxt.index(train_ctxt)
        start_train_time = time.time()
        for t in range(epochs):
            # print(f"epoch= {t}")
            train(train_ctxt, val_ctxt[val_idx], model, loss_fn, optimizer, 1, 0.5, 50, ctxt_count)
            if early_stopping.early_stop:
                print(f"epoch: {t}, Early stopping")
                break
        end_train_time = time.time()
        total_train_time += end_train_time - start_train_time
        
        early_stopping.load_best_model(model)

        test_on_train_ctxt = test_all(model, ctxt_count)
        during_averages += test_on_train_ctxt[ctxt_count]
        saved_errors_list.append(test_on_train_ctxt)

        print(f"After training on C{ctxt_count+1}:")
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