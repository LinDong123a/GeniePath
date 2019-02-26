import time
import numpy as np
import torch
from torch import optim

from models import GeniePath
from utils import (
        adj_to_bias, load_data, preprocess_features)

dataset = 'cora'

# training params
batch_size = 1
n_epochs = 10000
patience = 200
lr = 0.005
l2_coef = 0.0005
attn_dropout = 0.4
ff_dropout = 0.4
hidden_units = 128
n_layer = 3
nonlinearity = torch.tanh
model = GeniePath

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: {}'.format(lr))
print('l2_coef: {}'.format(l2_coef) )
print('feed forward dropout: {}'.format(ff_dropout))
print('attention dropout: {}'.format(attn_dropout))
print('patience: {}'.format(patience))
print('----- Archi. hyperparams -----')
print('no. layers: {}'.format(n_layer))
print('no. hidden units: {}'.format(hidden_units))
print('nonlinearity: {}'.format(nonlinearity))
print('model: {}'.format(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
features, spars = preprocess_features(features)

n_node = features.shape[0]
ft_size = features.shape[1]
n_class = y_train.shape[1]

adj = adj.todense()

features = torch.from_numpy(features)
y_train = torch.from_numpy(y_train)
y_val = torch.from_numpy(y_val)
y_test = torch.from_numpy(y_test)
train_mask = torch.from_numpy(np.array(train_mask, dtype=np.uint8))
val_mask = torch.from_numpy(np.array(val_mask, dtype=np.uint8))
test_mask = torch.from_numpy(np.array(test_mask, dtype=np.uint8))

bias_mtx = torch.from_numpy(adj_to_bias(adj[np.newaxis], [n_node], n_neigh=1))[0]
bias_mtx = bias_mtx.type(torch.FloatTensor)

model = GeniePath(ft_size, n_class, hidden_units, n_layer, 
                  attn_dropout=attn_dropout, ff_dropout=ff_dropout)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
loss_function = torch.nn.CrossEntropyLoss()

_, y_train = y_train.max(dim=-1)
_, y_val = y_val.max(dim=-1)
_, y_test = y_test.max(dim=-1)

print('---------- Training Info --------------')
print('Training node number: %d' % train_mask.sum().item())
print('Validation node number: %d' % val_mask.sum().item())
print('Test node number: %d' % test_mask.sum().item())

#for epoch in range(n_epochs):
val_loss_min = 100.0
val_accu_max = 0.0
n_waiting_step = 0

for epoch in range(n_epochs):
    model.zero_grad()
    optimizer.zero_grad()

    class_pred = model(features, n_node, train_mask, bias_mtx=bias_mtx)

    train_pred = class_pred[train_mask, :]
    train_label = y_train[train_mask]

    train_loss = loss_function(train_pred, train_label)

    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        # presave train related label
        train_label = model.predict()

        val_class_pred = model(features, n_node, val_mask, bias_mtx=bias_mtx, training=False)

        val_pred = class_pred[val_mask, :]
        val_label = y_val[val_mask]
        
        val_loss = loss_function(val_pred, val_label)

        class_label = model.predict()
        val_accu = model.masked_accu(class_label, y_val, val_mask)
        val_loss = val_loss.item()
        print('Epoch: %d, train_loss, %.5f, train_accu: %.5f, val_loss: %.5f, val_accu: %.5f' % (
            epoch+1, train_loss.item(), model.masked_accu(train_label, y_train, train_mask),
            val_loss, val_accu))
        
        if val_accu >= val_accu_max or val_loss <= val_loss_min:
            if val_accu >= val_accu_max and val_loss <= val_loss_min:
                print('best one, saved')
                torch.save(model.state_dict(), './pretrained_model/genie.pt')
                torch.save(model, './pretrained_model/entire_model.pt')
            val_accu_max = max([val_accu_max, val_accu])
            val_loss_min = min([val_loss_min, val_loss])
            n_waiting_step = 0
        else:
            n_waiting_step += 1
            if n_waiting_step == patience:
                print('Early Stop! Epoch: %d, max val_accu: %.5f, min val_loss: %.5f' % (
                    epoch+1, val_accu_max, val_loss_min))
                break

with torch.no_grad():
    class_pred = model(features, n_node, test_mask, bias_mtx=bias_mtx, training=False)

    test_pred = class_pred[test_mask, :]
    test_label = y_test[test_mask]
    
    test_loss = loss_function(test_pred, test_label)

    class_label = model.predict()
    print('test_loss: %.4f, accu: %.4f' % (
        test_loss.item(), model.masked_accu(class_label, y_test, test_mask)))
    
"""
model = GeniePath(ft_size, n_class, hidden_units, n_layer)
model.load_state_dict(torch.load('./pretrained_model/genie.pt'))
model.eval()
"""
model = torch.load('./pretrained_model/entire_model.pt')

with torch.no_grad():
    class_pred = model(features, n_node, test_mask, bias_mtx=bias_mtx, training=False)

    test_pred = class_pred[test_mask, :]
    test_label = y_test[test_mask]
    
    test_loss = loss_function(test_pred, test_label)

    class_label = model.predict()
    print('test_loss: %.4f, accu: %.4f' % (
        test_loss.item(), model.masked_accu(class_label, y_test, test_mask)))
    

