import time

import torch
from utils.augmentation import apply_augmentation


def _residual_target(model, load, mark, era, img, target, mode, config=None):
    if mode not in {'pretrain_weather', 'pretrain_img'}:
        return target

    config = config or {}
    # Residual labels must be deterministic. Keep the residual expert in train
    # mode, but force the backbone into eval mode while producing base forecasts.
    backbone_was_training = model.backbone.training
    model.backbone.eval()
    with torch.no_grad():
        base_pred = model(load, mark, era, img, mode='pretrain_backbone')
    model.backbone.train(backbone_was_training)

    residual = target - base_pred.detach()
    if config.get('residual_target_clip', True):
        clip_value = config.get('aux_delta_scale', config.get('cross_delta_scale', 0.1))
        residual = residual.clamp(min=-clip_value, max=clip_value)
    return residual


def _eval_frozen_modules(model):
    """Disable dropout/batchnorm in modules whose parameters are frozen."""
    for module in model.children():
        params = list(module.parameters(recurse=True))
        if params and not any(param.requires_grad for param in params):
            module.eval()
    if hasattr(model, 'branch_dropout') and not any(
        ('weather_proj' in name or 'image_proj' in name) and param.requires_grad
        for name, param in model.named_parameters()
    ):
        model.branch_dropout.eval()


def   train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config=None, mode='finetune'):
    config = config or {}
    model.train()
    epoch_loss = 0.0
    proc_loss = 0.0
    proc_size = 0
    log_interval = config.get('log_interval', 30)
    lambda_sparse = config.get('lambda_sparse', 0.0)
    lambda_delta = config.get('lambda_delta', 0.0)
    start_time = time.time()

    for i_batch, (batch_X, batch_Y, batch_X_mark, _) in enumerate(train_loader):
        _eval_frozen_modules(model)
        load, era, img = [x.to(device) for x in batch_X]
        target = batch_Y.to(device)
        mark = batch_X_mark.to(device)

        load, era, img = apply_augmentation(
            load,
            era,
            img,
            noise_std=config.get('augment_noise_std', 0.0),
            jitter_max=config.get('augment_jitter_max', 0),
        )

        optimizer.zero_grad()
        output = model(load, mark, era, img, mode=mode)

        if 'pretrain' in mode:
            supervised_target = _residual_target(model, load, mark, era, img, target, mode, config=config)
            loss = criterion(output, supervised_target)
        else:
            final_pred, w_weather, w_image, weather_bias, image_bias = output
            loss_main = criterion(final_pred, target)
            loss_sparsity = w_weather.mean() + w_image.mean()
            loss_delta = weather_bias.abs().mean() + image_bias.abs().mean()
            loss = loss_main + lambda_sparse * loss_sparsity + lambda_delta * loss_delta
            if i_batch == 0 and epoch == 0:
                print(
                    f"[diagnostic] mean router weights: weather={w_weather.mean().item():.6f}, "
                    f"image={w_image.mean().item():.6f}"
                )
                print(
                    f"[diagnostic] mean abs residual corrections: weather={weather_bias.abs().mean().item():.6f}, "
                    f"image={image_bias.abs().mean().item():.6f}"
                )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip', 0.5))
        optimizer.step()

        batch_size = load.size(0)
        epoch_loss += loss.item() * batch_size
        proc_loss += loss.item() * batch_size
        proc_size += batch_size
        if i_batch % log_interval == 0 and i_batch > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            print(
                'Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.format(
                    epoch, i_batch, len(train_loader), elapsed_time * 1000 / log_interval, avg_loss
                )
            )
            proc_loss, proc_size = 0.0, 0
            start_time = time.time()

    return epoch_loss / len(train_loader.dataset)


def evaluate(model, loader, criterion, device, mode='finetune', config=None):
    config = config or {}
    model.eval()
    total_loss = 0.0
    results, truths = [], []

    with torch.no_grad():
        for batch_X, batch_Y, batch_X_mark, _ in loader:
            load, era, img = [x.to(device) for x in batch_X]
            target = batch_Y.to(device)
            mark = batch_X_mark.to(device)

            if 'pretrain' in mode:
                preds = model(load, mark, era, img, mode=mode)
                supervised_target = _residual_target(model, load, mark, era, img, target, mode, config=config)
                loss = criterion(preds, supervised_target)
            else:
                preds, _, _, _, _ = model(load, mark, era, img, mode='finetune')
                loss = criterion(preds, target)

            total_loss += loss.item() * load.size(0)
            results.append(preds.cpu())
            truths.append(target.cpu())

    return total_loss / len(loader.dataset), torch.cat(results), torch.cat(truths)
