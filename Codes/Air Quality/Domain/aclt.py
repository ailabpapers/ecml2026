import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
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

      self.encoder = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU()
      )
      self.regressor = nn.Linear(16, 1)

  def forward(self, x):
    h = self.encoder(x)
    y = self.regressor(h)
    return y

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
loss_fn_ser = nn.MSELoss(reduction='none')
memory_main = []
soft_memory = [torch.rand(1, 1), torch.rand(1, 1)]

def update_main_buffer(features, target, logit, memory_K):
  global memory_main
  new_samples = list(zip(features, target, logit))
  for sample in new_samples:
    if memory_K > len(memory_main):
      memory_main.append(sample)
    else:
      random_key = random.randrange(len(memory_main))
      memory_main[random_key] = sample

def get_buffer_sample():
  global memory_main
  random_key = random.randrange(len(memory_main))
  feat, tar, logit = memory_main[random_key]
  return feat, tar, logit

def train(dataloader, val_data, model, loss_fn, optimizer, ser_alpha, ser_beta, ser_gamma, memory_K, soft_memory_C, prev_encoder, ctxt_count):
  global memory_main, soft_memory
  size = len(dataloader)
  model.train()
  correct = 0
  running_loss = 0

  for i, (Xt, yt) in enumerate(dataloader):
    Xt = Xt.squeeze()
    optimizer.zero_grad()
    
    if len(soft_memory[0]) >= soft_memory_C:
      x_d, y_d = soft_memory[0], soft_memory[1]
      if Xt.dim() > 1:
        Xall = torch.cat([Xt, x_d], axis=0)
      else:
        Xt = Xt.unsqueeze(0)
        Xall = torch.cat([Xt, x_d], axis=0)
      yall = torch.cat([yt, y_d], axis=0)
    else:
      Xall = Xt
      yall = yt
    
    if len(memory_main) >= memory_K:
      x_dd, y_dd, z_dd = get_buffer_sample()
      x_dd = x_dd.unsqueeze(0)
      y_dd = y_dd.unsqueeze(0)
      z_dd = z_dd.unsqueeze(0)
      for j in range(memory_K-1):
        x_inter, y_inter, z_inter = get_buffer_sample()
        x_inter = x_inter.unsqueeze(0)
        y_inter = y_inter.unsqueeze(0)
        z_inter = z_inter.unsqueeze(0)
        x_dd = torch.cat([x_dd, x_inter], axis=0)
        y_dd = torch.cat([y_dd, y_inter], axis=0)
        z_dd = torch.cat([z_dd, z_inter], axis=0)

    pred = model(Xall)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, yall)
    LP_eplsion_t_1 = loss_fn_ser(pred, yall)
    
    model.eval()
    if len(memory_main) >= memory_K:
      f_theta_x_dd = model(x_dd)
      f_theta_x_dd = f_theta_x_dd.squeeze()
      alpha_var = torch.sum((y_dd - f_theta_x_dd)**2)
      
      h_theta_x_dd = model.encoder(x_dd)
      beta_var = torch.sum((z_dd - h_theta_x_dd)**2)
      
      h_theta_xt = model.encoder(Xt)
      with torch.no_grad():
        prev_encoder_value = prev_encoder(Xt)
      gamma_var = torch.sum((prev_encoder_value - h_theta_xt)**2)
      loss += ser_alpha * alpha_var + ser_beta * beta_var + ser_gamma * gamma_var
    model.train()
    running_loss += loss.item()

    loss.backward()
    optimizer.step()
    
    update_main_buffer(Xt.detach(), yt.detach(), model.encoder(Xt).detach(), memory_K)
    
    lp_epsilon_model_inter = model(Xall)
    lp_epsilon_model_inter = lp_epsilon_model_inter.squeeze(-1)
    LP_eplsion_t = loss_fn_ser(lp_epsilon_model_inter, yall)
    LP = LP_eplsion_t - LP_eplsion_t_1
    losses_per_sample = LP.view(LP.size(0), -1).mean(dim=1)
    C = min(soft_memory_C, Xall.size(0))
    _, top_C = torch.topk(losses_per_sample, k=C, largest=False)
    X_top_C = Xall[top_C]
    y_top_C = yall[top_C]
    
    soft_memory[0] = X_top_C
    soft_memory[1] = y_top_C

    if i % 100 == 0:
      loss_log, current = loss.item(), i * batch_size + len(Xall)
  #     print(f"loss: {loss_log:>3f}  [{current:>3d}/{size:>3d}]")

  model.eval()
  val_loss = 0
  with torch.no_grad():
    for data, target in val_data:
      data = data.squeeze()
      output = model(data)
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
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      in_range = (torch.abs((pred - y) / y)) <= 0.05

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

    ctxt_count = 0
    saved_errors_list = []
    total_train_time = 0
    during_averages = 0
    after_averages = 0
    fr_saved = []
    memory_main.clear()
    soft_memory = [torch.rand(1, 1), torch.rand(1, 1)]
    previous_encoder = copy.deepcopy(model.encoder)

    for train_ctxt in training_ctxt:
        early_stopping = EarlyStopping(patience=10, delta=0.0003)
        val_idx = training_ctxt.index(train_ctxt)
        start_train_time = time.time()
        
        for t in range(epochs):
            train(train_ctxt, val_ctxt[val_idx], model, loss_fn, optimizer, 0.2, 0.2, 0.2, 50, 4, previous_encoder, ctxt_count)
            if early_stopping.early_stop:
                print(f"epoch: {t}, Early stopping")
                break
        end_train_time = time.time()
        total_train_time += end_train_time - start_train_time
        
        previous_encoder = copy.deepcopy(model.encoder)
        
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