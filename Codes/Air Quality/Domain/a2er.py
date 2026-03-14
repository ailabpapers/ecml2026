import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
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

      self.linear1 = torch.nn.Linear(8, 16)
      self.activation = torch.nn.ReLU()
      self.linear2 = torch.nn.Linear(16, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.activation(x)
    x = self.linear2(x)
    return x

model = simpleNN()

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
        
learning_rate = 0.001
loss_fn = nn.MSELoss()
reservoir_buffer = []

def compute_gamma_tau(delta_tau_vec, delta_Q, rho=0.5, eps=1e-12):
    """
    delta_tau_vec: shape (B,)  Δ_τ = 0.5 * ||h(xτ) - zτ||^2  per reservoir sample
    delta_Q      : scalar Δ_Q
    rho          : quantile
    returns      : gamma_vec of shape (B,)
    """

    if delta_tau_vec.numel() == 0:
        return torch.zeros_like(delta_tau_vec)
    delta_Q_high = delta_Q / max(rho, eps)

    if delta_Q_high - delta_Q <= eps:
        return torch.zeros_like(delta_tau_vec)

    clipped = delta_tau_vec.clamp(min=delta_Q, max=delta_Q_high)
    eta = 1.0 - (clipped - delta_Q) / (delta_Q_high - delta_Q)
    eta = eta.clamp(0.0, 1.0)

    target_delta = eta * delta_tau_vec + (1.0 - eta) * delta_Q

    safe_delta = torch.clamp(delta_tau_vec, min=eps)
    ratio = torch.clamp(target_delta / safe_delta, min=0.0)
    gamma = 1.0 - torch.sqrt(ratio)

    return gamma.clamp(0.0, 1.0)

def update_reservoir_buffer(features, logit, reservoir_buffer_length):
  global reservoir_buffer
  if features.dim() > 1:
    new_samples = list(zip(features, logit, [1] * len(features)))
    for sample in new_samples:
      if reservoir_buffer_length > len(reservoir_buffer):
        reservoir_buffer.append(sample)
      else:
        random_key = random.randrange(len(reservoir_buffer))
        reservoir_buffer[random_key] = sample
  else:
    if reservoir_buffer_length > len(reservoir_buffer):
      reservoir_buffer.append((features, logit, 1))
    else:
      random_key = random.randrange(len(reservoir_buffer))
      reservoir_buffer[random_key] = (features, logit, 1)

def get_buffer_sample():
  global reservoir_buffer
  random_key = random.randrange(len(reservoir_buffer))
  feat, logit, gamma_bar = reservoir_buffer[random_key]
  return feat, logit, random_key

def train(dataloader, val_data, model, loss_fn, optimizer, alpha_raw, beta_logit, N_RS, rho, eps, lambda_priority, delta_Q, task_id):
  global reservoir_buffer
  size = len(dataloader)
  model.train()
  correct = 0
  running_loss = 0

  for i, (X, y) in enumerate(dataloader):
    # print(f"Sample idx i:{i}")
    X = X.squeeze()
    optimizer.zero_grad()

    pred = model(X)
    if pred.dim() > 1:
      pred = pred.squeeze(-1)
    update_reservoir_buffer(X, pred.detach(), N_RS)
    fifo_loss = loss_fn(pred, y)

    in_range = (torch.abs((pred - y) / y)) <= 0.05
    correct += ((in_range).sum().item())
    
    if len(reservoir_buffer) > 0:
      feat_rs, saved_logit_rs, idx_rs = get_buffer_sample()
      feat_rs = feat_rs.unsqueeze(0)
      saved_logit_rs = saved_logit_rs.view(1)
      model.eval()
      with torch.no_grad():
        buffer_pred_rs = model(feat_rs)
        buffer_pred_rs = buffer_pred_rs.squeeze(-1)
      diff = buffer_pred_rs - saved_logit_rs
      delta_tau_vec = 0.5 * diff.view(diff.shape[0], -1).pow(2).sum(dim=1)

      batch_size_rs = delta_tau_vec.shape[0]
      gamma_batch = batch_size_rs / float(N_RS)

      q_batch = torch.quantile(delta_tau_vec, rho).item()
      delta_Q = (1.0 - gamma_batch) * delta_Q + gamma_batch * q_batch

      gamma_vec = compute_gamma_tau(delta_tau_vec, delta_Q, rho=rho, eps=eps)

      correction_factor = (1.0 - gamma_vec)
      correction_factor_sq = correction_factor ** 2

      feature_reg_term = 0.5 * (diff ** 2) * correction_factor_sq
      feature_reg = feature_reg_term.mean()

      model.train()
      buffer_pred_rs = model(feat_rs)
      buffer_pred_rs = buffer_pred_rs.squeeze(-1)

      rs_loss = loss_fn(buffer_pred_rs, saved_logit_rs)

      alpha = F.softplus(alpha_raw)
      beta = torch.sigmoid(beta_logit)

      main_loss = (1.0 - beta) * fifo_loss + beta * rs_loss + alpha * feature_reg

      fifo_loss_det = fifo_loss.detach()
      rs_loss_det = rs_loss.detach()
      feature_reg_det = feature_reg.detach()

      beta_constraint = -beta * (rs_loss_det - fifo_loss_det)
      alpha_constraint = -alpha * (feature_reg_det - delta_Q)

      loss = main_loss + beta_constraint + alpha_constraint

      with torch.no_grad():
        old_feat, old_logit, gamma_bar_old = reservoir_buffer[idx_rs]

        gamma_scalar = gamma_vec.item()
        gamma_bar_new = (1.0 - lambda_priority) * gamma_bar_old + \
                        lambda_priority * (1.0 - gamma_scalar)

        old_logit = old_logit.view(1)
        new_logit = (1.0 - gamma_scalar) * old_logit + \
                    gamma_scalar * buffer_pred_rs.detach()
        new_logit = new_logit.view(())

        reservoir_buffer[idx_rs] = (old_feat, new_logit, gamma_bar_new)
    else:
      alpha = F.softplus(alpha_raw)
      beta = torch.sigmoid(beta_logit)
      loss = fifo_loss
      main_loss = loss
    
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
      output = model(data)
      loss = loss_fn(output, target)
      val_loss += loss.item()
  val_loss /= len(val_data)

  early_stopping(val_loss, model)
  return running_loss/size, correct*100/size

def test(dataloader, model, loss_fn, task_id):
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

def test_all(model, task_id):
    list_loss = []
    for test_ctxt in testing_ctxt:
        test_loss_ctxt, acc = test(test_ctxt, model, loss_fn, task_id)
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
    
    # --- A2ER hyperparameters ---
    rho = 0.5
    lambda_priority = 0.5
    delta_Q = 0.0 
    eps = 1e-12

    alpha_raw = torch.nn.Parameter(torch.tensor(1.0))
    beta_logit = torch.nn.Parameter(torch.tensor(0.5))
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [alpha_raw, beta_logit],
        lr=learning_rate
    )

    ctxt_count = 0
    saved_errors_list = []
    total_train_time = 0
    during_averages = 0
    after_averages = 0
    fr_saved = []
    reservoir_buffer.clear()

    for train_ctxt in training_ctxt:
        early_stopping = EarlyStopping(patience=10, delta=0.0003)
        val_idx = training_ctxt.index(train_ctxt)
        start_train_time = time.time()
        for t in range(epochs):
            train(train_ctxt, val_ctxt[val_idx], model, loss_fn, optimizer, alpha_raw, beta_logit, 50, rho, eps, lambda_priority, delta_Q, ctxt_count)
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