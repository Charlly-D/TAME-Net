import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features

ZONE_CENTERS = {
    "NEMASSBO": (42.36, -71.06),
    "SEMASS": (41.90, -70.90),
    "WCMASS": (42.27, -71.80),
    "RI": (41.82, -71.41),
    "CT": (41.60, -72.70),
    "VT": (44.00, -72.70),
    "NH": (43.20, -71.50),
    "ME": (44.50, -69.50),
}

CSV_COLUMN_MAP = {
    "ME": "ISNE - Maine",
    "NH": "ISNE - New Hampshire",
    "VT": "ISNE - Vermont",
    "CT": "ISNE - Connecticut",
    "RI": "ISNE - Rhode Island",
    "SEMASS": "ISNE - Southeast Mass.",
    "WCMASS": "ISNE - Western/Central Mass.",
    "NEMASSBO": "ISNE - Northeast Mass.",
}

DEFAULT_ERA_VARS = ["t2m", "u10", "v10", "sp", "tp", "d2m"]

# Geographic extent used when the ERA5/GOES files were generated.
LAT_MAX = 47.5
LAT_MIN = 41.0
LON_MIN = -73.5
LON_MAX = -67.0
RESOLUTION = 0.25
IMG_W = 256
IMG_H = 128


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _handle_nan(array):
    if np.isnan(array).any():
        return np.nan_to_num(array, nan=np.nanmean(array))
    return array


def _load_era5(path, era_vars=None):
    era_vars = era_vars or DEFAULT_ERA_VARS
    with xr.open_dataset(path) as ds:
        return np.stack([ds[v].values for v in era_vars], axis=-1).astype(np.float32)


def _time_features(df_load, border1, border2, timeenc, freq):
    df_stamp = df_load[["datetime"]][border1:border2].copy()
    if timeenc == 0:
        df_stamp["month"] = df_stamp.datetime.dt.month
        df_stamp["day"] = df_stamp.datetime.dt.day
        df_stamp["weekday"] = df_stamp.datetime.dt.weekday
        df_stamp["hour"] = df_stamp.datetime.dt.hour
        data_stamp = df_stamp.drop(["datetime"], axis=1).values
    else:
        data_stamp = time_features(pd.to_datetime(df_stamp["datetime"].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp.astype(np.float32)


def crop_era5_zone(full_data, zone_name, crop_size):
    """Crop ERA5 data around a zone center.

    full_data shape: [T, H, W, C]. Returned shape: [T, crop_size, crop_size, C].
    """
    if zone_name not in ZONE_CENTERS:
        raise ValueError(f"Unknown zone_name={zone_name}. Available: {sorted(ZONE_CENTERS)}")

    center_lat, center_lon = ZONE_CENTERS[zone_name]
    lat_idx = int(round((LAT_MAX - center_lat) / RESOLUTION))
    lon_idx = int(round((center_lon - LON_MIN) / RESOLUTION))

    max_h, max_w = full_data.shape[1], full_data.shape[2]
    crop_size = min(crop_size, max_h, max_w)
    half = crop_size // 2

    lat_start = max(0, lat_idx - half)
    lat_start = min(lat_start, max_h - crop_size)
    lon_start = max(0, lon_idx - half)
    lon_start = min(lon_start, max_w - crop_size)

    return full_data[:, lat_start:lat_start + crop_size, lon_start:lon_start + crop_size, :]


def crop_goes_zone(full_img, zone_name, deg_span=2.0):
    """Crop GOES images around a zone center.

    full_img shape: [T, H, W]. Returned shape is variable and can be resized later.
    """
    if zone_name not in ZONE_CENTERS:
        raise ValueError(f"Unknown zone_name={zone_name}. Available: {sorted(ZONE_CENTERS)}")

    center_lat, center_lon = ZONE_CENTERS[zone_name]
    ratio_y = (LAT_MAX - center_lat) / (LAT_MAX - LAT_MIN)
    ratio_x = (center_lon - LON_MIN) / (LON_MAX - LON_MIN)
    center_y = int(round(ratio_y * (IMG_H - 1)))
    center_x = int(round(ratio_x * (IMG_W - 1)))

    ppd_y = IMG_H / (LAT_MAX - LAT_MIN)
    ppd_x = IMG_W / (LON_MAX - LON_MIN)
    crop_h = max(1, int(round(deg_span * ppd_y)))
    crop_w = max(1, int(round(deg_span * ppd_x)))
    half_h = crop_h // 2
    half_w = crop_w // 2

    y_start = max(0, center_y - half_h)
    y_start = min(y_start, IMG_H - crop_h)
    x_start = max(0, center_x - half_w)
    x_start = min(x_start, IMG_W - crop_w)

    return full_img[:, y_start:y_start + crop_h, x_start:x_start + crop_w]


class BaseLoadDataset(Dataset):
    def __init__(self, data_paths: dict, config: dict, flag="train", stats=None):
        self.seq_len = config.get("seq_len", 96)
        self.label_len = config.get("label_len", 48)
        self.pred_len = config.get("pred_len", 96)
        self.scale = config.get("scale", True)
        self.timeenc = config.get("timeenc", 0)
        self.freq = config.get("freq", "h")
        self.goes_ratio = config.get("goes_time_ratio", 4)
        self.config = config

        self.load_csv = data_paths["load"]
        self.era5_nc = data_paths["era5"]
        self.goes_imgs_npy = data_paths["goes"]

        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.stats = stats

    def _split_borders(self, total_len):
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        if self.set_type == 0:
            return 0, train_end, train_end, val_end
        if self.set_type == 1:
            return train_end, val_end, train_end, val_end
        return val_end, total_len, train_end, val_end

    def _slice_goes(self, raw_goes, border1, border2):
        buffer = self.seq_len // self.goes_ratio + 1
        goes_slice_start = border1 // self.goes_ratio
        if self.set_type == 2:
            goes_slice_end = len(raw_goes)
        else:
            goes_slice_end = min(len(raw_goes), border2 // self.goes_ratio + buffer)
        return raw_goes[goes_slice_start:goes_slice_end], goes_slice_start

    def _make_stats(self, raw_goes_train):
        return {
            "load_mean": _as_list(np.mean(self.load_data, axis=0)),
            "load_std": _as_list(np.std(self.load_data, axis=0) + 1e-5),
            "era_mean": _as_list(np.mean(self.era_data, axis=(0, 1, 2))),
            "era_std": _as_list(np.std(self.era_data, axis=(0, 1, 2)) + 1e-5),
            "goes_mean": float(np.mean(raw_goes_train)),
            "goes_std": float(np.std(raw_goes_train) + 1e-5),
        }

    def _normalize(self, raw_goes_train):
        if not self.scale:
            return
        if self.flag == "train":
            self.stats = self._make_stats(raw_goes_train)
        elif self.stats is None:
            raise ValueError(f"train stats must be passed when building the {self.flag} dataset")

        self.load_data = (self.load_data - np.array(self.stats["load_mean"])) / np.array(self.stats["load_std"])
        self.era_data = (self.era_data - np.array(self.stats["era_mean"])) / np.array(self.stats["era_std"])
        self.goes_data = (self.goes_data - self.stats["goes_mean"]) / self.stats["goes_std"]

    def __len__(self):
        length_by_load = len(self.load_data) - self.seq_len - self.pred_len + 1
        expected_goes_len = self.seq_len // self.goes_ratio
        last_goes_bucket = self.goes_slice_start + len(self.goes_data) - expected_goes_len
        max_abs_idx_by_goes = (last_goes_bucket + 1) * self.goes_ratio - 1
        length_by_goes = max_abs_idx_by_goes - self.border1 + 1
        return max(0, min(length_by_load, length_by_goes))

    def _get_goes_seq(self, idx):
        abs_idx = self.border1 + idx
        goes_start = (abs_idx // self.goes_ratio) - self.goes_slice_start
        goes_end = goes_start + (self.seq_len // self.goes_ratio)
        goes_seq = self.goes_data[goes_start:goes_end]
        expected_len = self.seq_len // self.goes_ratio
        if goes_seq.shape[0] != expected_len:
            raise IndexError(
                f"GOES sequence length mismatch: got {goes_seq.shape[0]}, expected {expected_len}, "
                f"idx={idx}, abs_idx={abs_idx}, split={self.flag}"
            )
        return goes_seq

    def __getitem__(self, idx):
        load_seq = self.load_data[idx: idx + self.seq_len]
        era_seq = self.era_data[idx: idx + self.seq_len]
        goes_seq = self._get_goes_seq(idx)
        target_seq = self.load_data[idx + self.seq_len: idx + self.seq_len + self.pred_len]

        load_tensor = torch.from_numpy(load_seq).float()
        era_tensor = torch.from_numpy(era_seq).float().permute(0, 3, 1, 2).contiguous()
        goes_tensor = torch.from_numpy(goes_seq).float().unsqueeze(1).contiguous()

        image_size = self.config.get("regional_img_size", None)
        if image_size is not None:
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            goes_tensor = F.interpolate(goes_tensor, size=tuple(image_size), mode="bilinear", align_corners=False)

        y = torch.from_numpy(target_seq).float()

        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x_mark = torch.from_numpy(self.data_stamp[s_begin:s_end]).float()
        seq_y_mark = torch.from_numpy(self.data_stamp[r_begin:r_end]).float()

        return (load_tensor, era_tensor, goes_tensor), y, seq_x_mark, seq_y_mark


class CustomDataset(BaseLoadDataset):
    """System-level ISO-NE load dataset."""

    def __init__(self, data_paths: dict, config: dict, flag="train", stats=None):
        super().__init__(data_paths, config, flag=flag, stats=stats)
        self.__read_data__()

    def __read_data__(self):
        df_load = pd.read_csv(self.load_csv, parse_dates=["datetime"])
        df_load["value"] = df_load["value"].interpolate()
        raw_load = df_load["value"].values.reshape(-1, 1).astype(np.float32)
        raw_era = _load_era5(self.era5_nc, self.config.get("era_vars"))
        raw_goes = np.load(self.goes_imgs_npy).astype(np.float32)

        raw_era = _handle_nan(raw_era)
        raw_goes = _handle_nan(raw_goes)

        total_len = len(raw_load)
        border1, border2, train_end, _ = self._split_borders(total_len)
        self.border1 = border1
        self.border2 = border2
        self.load_data = raw_load[border1:border2]
        self.era_data = raw_era[border1:border2]
        self.goes_data, self.goes_slice_start = self._slice_goes(raw_goes, border1, border2)
        raw_goes_train = raw_goes[: train_end // self.goes_ratio]

        self._normalize(raw_goes_train)
        self.data_stamp = _time_features(df_load, border1, border2, self.timeenc, self.freq)


class RegionalDataset(BaseLoadDataset):
    """ISO-NE zone-level load dataset with zone-specific ERA5 and GOES crops."""

    def __init__(self, data_paths: dict, config: dict, flag="train", stats=None):
        super().__init__(data_paths, config, flag=flag, stats=stats)
        self.zone_name = config.get("zone_name")
        if not self.zone_name:
            raise ValueError("RegionalDataset requires config['zone_name']")
        self.zone_load_column = config.get("zone_load_column", CSV_COLUMN_MAP.get(self.zone_name))
        if not self.zone_load_column:
            raise ValueError(f"No load column mapping configured for zone_name={self.zone_name}")
        self.__read_data__()

    def __read_data__(self):
        df_load = pd.read_csv(self.load_csv, parse_dates=["datetime"])
        if self.zone_load_column not in df_load.columns:
            raise ValueError(
                f"Load column '{self.zone_load_column}' was not found in {self.load_csv}. "
                f"Available columns: {list(df_load.columns)}"
            )
        raw_load = df_load[self.zone_load_column].interpolate().values.reshape(-1, 1).astype(np.float32)

        raw_era = _load_era5(self.era5_nc, self.config.get("era_vars"))
        crop_size = self.config.get("zone_weather_crop_size", self.config.get("weather_spatial_size", 27))
        raw_era = crop_era5_zone(raw_era, self.zone_name, crop_size=crop_size)

        raw_goes = np.load(self.goes_imgs_npy).astype(np.float32)
        raw_goes = crop_goes_zone(raw_goes, self.zone_name, deg_span=self.config.get("zone_goes_deg_span", 2.0))

        raw_era = _handle_nan(raw_era)
        raw_goes = _handle_nan(raw_goes)

        total_len = len(raw_load)
        border1, border2, train_end, _ = self._split_borders(total_len)
        self.border1 = border1
        self.border2 = border2
        self.load_data = raw_load[border1:border2]
        self.era_data = raw_era[border1:border2]
        self.goes_data, self.goes_slice_start = self._slice_goes(raw_goes, border1, border2)
        raw_goes_train = raw_goes[: train_end // self.goes_ratio]

        self._normalize(raw_goes_train)
        self.data_stamp = _time_features(df_load, border1, border2, self.timeenc, self.freq)


def _select_dataset(config):
    dataset_mode = config.get("dataset_mode", "system")
    if dataset_mode == "regional" or config.get("zone_name"):
        return RegionalDataset
    return CustomDataset


def build_dataloaders(data_paths: dict, config: dict, save_dir: str):
    """Build train/validation/test dataloaders and persist train-only scaler stats."""
    os.makedirs(save_dir, exist_ok=True)
    dataset_cls = _select_dataset(config)
    print(f">>> Building Training Data with {dataset_cls.__name__}...")
    train_dataset = dataset_cls(data_paths, config, flag="train")

    stats_save_path = os.path.join(save_dir, "scaler_stats.json")
    with open(stats_save_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.stats, f, indent=4)
    print(f">>> Training statistics saved to: {stats_save_path}")

    print(">>> Building Validation/Test Data...")
    val_dataset = dataset_cls(data_paths, config, flag="val", stats=train_dataset.stats)
    test_dataset = dataset_cls(data_paths, config, flag="test", stats=train_dataset.stats)

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
