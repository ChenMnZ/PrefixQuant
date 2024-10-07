import torch


def clamp_mse(outlier_threshold=50):
    def func(output, label):
        output = output.clamp(min=-outlier_threshold, max=outlier_threshold)               
        label = label.clamp(min=-outlier_threshold, max=outlier_threshold)               
        return torch.nn.functional.mse_loss(output, label)
    return func

def skip_mse(skip_num=3):
    def func(output, label):
        # adaptive select the outlier token
        mean_value =  label.abs().mean(dim=-1)
        select_index =  mean_value.topk(skip_num,dim=-1)[1]
        mask = torch.ones_like(mean_value)
        batch_indices = torch.arange(mean_value.shape[0]).unsqueeze(1)
        mask[batch_indices, select_index] = 0
        mask = mask.unsqueeze(-1)
        return ((output-label)**2*mask).mean()
        # else:
        #     return ((output[:, prefix_length:]-label[:, prefix_length:])**2).mean()
            
    return func

def normalized_mse():
    def func(output, label):
        max_value = label.abs().max(dim=2,keepdim=True)[0]
        return torch.nn.functional.mse_loss(output/max_value, label/max_value)
    return func


def cosine_loss():
    def func(output, label):
        cosine = torch.nn.functional.cosine_similarity(output, label, -1)
        return  1 - cosine.mean()
    return  func


def get_recon_loss(loss_type, prefixed=False, prefix_length=0):
    if loss_type == "mse":
        loss_func = torch.nn.MSELoss()
    elif loss_type == "clamp_mse":
        loss_func = clamp_mse(10)
    elif loss_type == "skip_mse":
        loss_func = skip_mse()
    elif loss_type == "normalized_mse":
        loss_func = normalized_mse()
    elif loss_type == "cosine":
        loss_func = cosine_loss()
    else:
        raise NotImplementedError
    return loss_func