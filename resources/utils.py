import os
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from IPython.display import clear_output


def prepare_trainable_params(model, exceptions):
   for param_name, param in model.named_parameters():
      param.requires_grad = False

   trainable_parameters = []
   for param_name, param in model.named_parameters():
      # skip exceptions:
      for exception in exceptions:
         if exception in param_name:
            continue

      if 'weight_quantizer.step' in param_name:
         param.requires_grad = True
         trainable_parameters.append(param)

      elif 'weight_quantizer.offset' in param_name:
         param.requires_grad = True
         trainable_parameters.append(param)

   return trainable_parameters


def plot_loss(train_losses, learning_rates=None, figsize=(6, 3)):
    clear_output(wait=True)

    fig, ax = plt.subplots(figsize=figsize)
    lines = []

    line_loss = ax.plot(train_losses, label='loss', color='blue')[0]
    ax.set_xlabel('time')
    ax.set_ylabel('Loss')
    lines.append(line_loss)
    
    if learning_rates is not None:
       ax_lr = ax.twinx()
       ax_lr.set_ylabel('Learning rate')
       line_lr = ax_lr.plot(learning_rates, label='lr', color='red')[0]
       lines.append(line_lr)

    #labels = []
    ax.legend(lines, [line.get_label() for line in lines])

    plt.title('Training Loss Over Iterations')
    plt.grid()
    plt.show()


def print_file_size(path_to_file):
    file_size = os.path.getsize(path_to_file)/2**20
    print(f'{file_size:.03f} Mb')
