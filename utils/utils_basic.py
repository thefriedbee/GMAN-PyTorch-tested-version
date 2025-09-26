import torch
import matplotlib.pyplot as plt


def set_device(device_str: str | None = None):
    if device_str is not None:
        return torch.device(device_str)
    # if device is not specified, use the best available device
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    return torch.device(device_str)


def set_device_for_tensor(t: torch.Tensor, device: torch.device | None = None):
    if device is None:
        device = set_device()
    return t.to(device)


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def metric(pred, label, y_mask = None, print_percent_masked = True):
    if y_mask is None:
        print("No mask provided, using all values")
        y_mask = torch.ones_like(label, device=pred.device).to(torch.bool)
    
    # get percent of masked values
    percent_masked = torch.sum(y_mask) / y_mask.numel()
    if print_percent_masked:
        print(f"Percent of observed values: {percent_masked*100:.2f}%")

    mae = torch.abs(torch.sub(pred, label)).type(torch.float32)
    rmse = mae ** 2
    mape = mae / (label + 1e-6)

    # set loss to nan if masked places
    mae = torch.where(y_mask, mae, torch.nan)
    rmse = torch.where(y_mask, rmse, torch.nan)
    mape = torch.where(y_mask, mape, torch.nan)

    # compute metric considering masks
    mae = torch.nanmean(mae)
    rmse = torch.sqrt(torch.nanmean(rmse))
    mape = torch.nanmean(mape)
    return mae, rmse, mape


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The following function can be replaced by
#  'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss

def plot_train_val_loss(
    train_total_loss: list[float|torch.Tensor], 
    val_total_loss: list[float|torch.Tensor], 
    file_path: str
):
    train_total_loss = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t 
                        for t in train_total_loss]
    val_total_loss = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t 
                      for t in val_total_loss]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1),
             train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1),
             val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)
