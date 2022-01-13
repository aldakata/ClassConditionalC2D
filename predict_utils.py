import json
import torch
import numpy as np


def pred_test(loader, net1, net2, predicted_file):
    net1.eval()
    net2.eval()
    final_prediction = np.asarray([])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
            final_prediction=np.concatenate((final_prediction,predicted.tolist()))
    # Save predictions
    with open(predicted_file, 'w') as f:
        json.dump([int(pred) for pred in final_prediction], f)


def pred_train(loader, net1, net2, predicted_file):
    net1.eval()
    net2.eval()
    final_prediction = np.asarray([])
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(loader):
            inputs = inputs.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)
            final_prediction=np.concatenate((final_prediction,predicted.tolist()))
    # Save predictions
    with open(predicted_file, 'w') as f:
        json.dump([int(pred) for pred in final_prediction], f)