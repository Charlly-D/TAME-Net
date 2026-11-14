import torch


def _add_noise(x, noise_std):
    if noise_std <= 0:
        return x
    scale = x.std().clamp_min(1e-6)
    return x + torch.randn_like(x) * noise_std * scale


def _time_jitter(x, shifts):
    if shifts is None:
        return x
    out = []
    for i in range(x.size(0)):
        out.append(torch.roll(x[i], shifts=int(shifts[i].item()), dims=0))
    return torch.stack(out, dim=0)


def apply_augmentation(load_seq, era_seq, img_seq, noise_std=0.0, jitter_max=0):
    load_seq = _add_noise(load_seq, noise_std)
    era_seq = _add_noise(era_seq, noise_std)
    if img_seq is not None:
        img_seq = _add_noise(img_seq, noise_std)

    if jitter_max > 0:
        device = load_seq.device
        shifts = torch.randint(-jitter_max, jitter_max + 1, (load_seq.size(0),), device=device)
        load_seq = _time_jitter(load_seq, shifts)
        era_seq = _time_jitter(era_seq, shifts)
        if img_seq is not None:
            t_load = load_seq.size(1)
            t_img = img_seq.size(1)
            ratio = t_img / max(t_load, 1)
            shifts_img = torch.round(shifts.float() * ratio).to(torch.long)
            img_seq = _time_jitter(img_seq, shifts_img)

    return load_seq, era_seq, img_seq
