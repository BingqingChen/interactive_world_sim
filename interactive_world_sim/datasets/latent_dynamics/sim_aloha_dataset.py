import concurrent.futures
import copy
import gc
import glob
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import cv2
import h5py
import numpy as np
import psutil
import torch
import zarr
import zarr.storage
from filelock import FileLock
from imgaug import augmenters as iaa
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from yixuan_utilities.draw_utils import center_crop

from interactive_world_sim.utils.imagecodecs_numcodecs import Jpeg2k, register_codecs
from interactive_world_sim.utils.normalizer import (
    LinearNormalizer,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from interactive_world_sim.utils.pytorch_util import dict_apply
from interactive_world_sim.utils.replay_buffer import ReplayBuffer
from interactive_world_sim.utils.sampler import SequenceSampler

from .base_dataset import BaseImageDataset

register_codecs()


# convert raw hdf5 data to replay buffer, which is used for diffusion policy training
def _convert_real_to_dp_replay(
    store: zarr.storage.Store,
    shape_meta: dict,
    dataset_dir: str,
    ctrl_mode: str = "bimanual_push",
    n_workers: Optional[int] = None,
    max_inflight_tasks: Optional[int] = None,
) -> ReplayBuffer:
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        if type == "depth":
            depth_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    episodes_paths = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
    episodes_stem_name = [Path(path).stem for path in episodes_paths]
    episodes_idx = [int(stem_name.split("_")[-1]) for stem_name in episodes_stem_name]
    episodes_idx = sorted(episodes_idx)

    episode_ends = list()
    prev_end = 0
    lowdim_data_dict: dict = dict()
    rgb_data_dict: dict = dict()
    depth_data_dict: dict = dict()
    for epi_idx in tqdm(episodes_idx, desc="Loading episodes"):
        dataset_path = os.path.join(dataset_dir, f"episode_{epi_idx}.hdf5")
        with h5py.File(dataset_path) as file:
            # count total steps
            episode_length = file["action"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)

            # save lowdim data to lowedim_data_dict
            if "action" not in lowdim_data_dict:
                lowdim_data_dict["action"] = list()
            ee_pos = file["obs"]["ee_pos"][()]  # (T, N_arms, 4, 4)
            if ctrl_mode == "bimanual_push":
                n_arms = ee_pos.shape[1]
                action_data = np.concatenate(
                    [ee_pos[:, i, :2, 3] for i in range(n_arms)], axis=1
                )  # (T, 2*N_arms)
            elif ctrl_mode == "single_grasp":
                joint_pos = file["obs"]["joint_pos"][()]  # (T, 7)
                ee_xyz = ee_pos[:, 0, :3, 3]  # (T, 3)
                gripper = joint_pos[:, -1:]  # (T, 1)
                action_data = np.concatenate([ee_xyz, gripper], axis=1)  # (T, 4)
            elif ctrl_mode == "single_grasp_cmd":
                # Like single_grasp but the 4th dim is the actual gripper open/close COMMAND
                # (stored at obs/gripper), not joint_pos[:,-1] (the wrist joint, which carries no
                # grasp signal). Required for tasks where grasping matters, e.g. robosuite Lift.
                ee_xyz = ee_pos[:, 0, :3, 3]  # (T, 3)
                gripper = file["obs"]["gripper"][()]  # (T, 1) open/close command in [-1, 1]
                action_data = np.concatenate([ee_xyz, gripper], axis=1)  # (T, 4)
            else:
                raise NotImplementedError(
                    f"ctrl_mode '{ctrl_mode}' is not implemented for SimAlohaDataset"
                )
            lowdim_data_dict["action"].append(action_data)

            for key in rgb_keys:
                if key not in rgb_data_dict:
                    rgb_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)
                assert imgs[0].shape == (h, w, c)
                rgb_data_dict[key].append(imgs)

            for key in depth_keys:
                if key not in depth_data_dict:
                    depth_data_dict[key] = list()
                imgs = file["obs"]["images"][key][()]
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                crop_imgs = [center_crop(img, (h, w)) for img in imgs]
                resize_imgs = [
                    cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    for img in crop_imgs
                ]
                imgs = np.stack(resize_imgs, axis=0)[..., None]
                imgs = np.clip(imgs, 0, 1000).astype(np.uint16)
                assert imgs[0].shape == (h, w, c)
                depth_data_dict[key].append(imgs)

    def img_copy(
        zarr_arr: zarr.Array, zarr_idx: int, hdf5_arr: np.ndarray, hdf5_idx: int
    ) -> bool:
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # dump data_dict
    print("Dumping meta data")
    n_steps = episode_ends[-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    print("Dumping lowdim data")
    for key, data in lowdim_data_dict.items():
        data = np.concatenate(data, axis=0)
        _ = data_group.array(
            name=key,
            data=data,
            shape=data.shape,
            chunks=data.shape,
            compressor=None,
            dtype=data.dtype,
        )

    print("Dumping rgb data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: set = set()
        for key, data in rgb_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    print("Dumping depth data")
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for key, data in depth_data_dict.items():
            hdf5_arr = np.concatenate(data, axis=0)
            shape = tuple(shape_meta["obs"][key]["shape"])
            c, h, w = shape
            this_compressor = Jpeg2k(level=50)
            img_arr = data_group.require_dataset(
                name=key,
                shape=(n_steps, h, w, c),
                chunks=(1, h, w, c),
                compressor=this_compressor,
                dtype=np.uint16,
            )
            for hdf5_idx in tqdm(range(hdf5_arr.shape[0])):
                if len(futures) >= max_inflight_tasks:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                zarr_idx = hdf5_idx
                futures.add(
                    executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx)
                )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode image!")

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def _merge_replay_buffers(buffers: list) -> ReplayBuffer:
    """Concatenate several ReplayBuffers into one MemoryStore-backed buffer.

    Frame-chunked arrays (chunks[0] == 1, i.e. the compressed image streams) are
    merged by copying raw compressed chunks with renumbered keys — no decode /
    re-encode. Everything else (lowdim, single-chunk) is concatenated in memory.
    """

    def chunk_key(arr: zarr.Array, coords: tuple) -> str:
        try:
            return arr._chunk_key(coords)
        except AttributeError:
            sep = getattr(arr, "_dimension_separator", None) or "."
            return arr.path + "/" + sep.join(map(str, coords))

    out_store = zarr.MemoryStore()
    out_root = zarr.group(out_store)
    out_data = out_root.require_group("data", overwrite=True)
    out_meta = out_root.require_group("meta", overwrite=True)

    episode_ends = []
    offset = 0
    for buffer in buffers:
        ends = buffer.episode_ends[:]
        episode_ends.append(ends + offset)
        offset += int(ends[-1])
    episode_ends = np.concatenate(episode_ends)
    out_meta.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    keys = sorted(buffers[0].keys())
    for buffer in buffers[1:]:
        assert sorted(buffer.keys()) == keys, (
            f"replay buffers have mismatched keys: {sorted(buffer.keys())} vs {keys}"
        )
    n_total = int(episode_ends[-1])
    for key in keys:
        arrs = [buffer.data[key] for buffer in buffers]
        first = arrs[0]
        for arr in arrs[1:]:
            assert arr.shape[1:] == first.shape[1:] and arr.dtype == first.dtype, key
        if first.chunks[0] == 1:
            for arr in arrs[1:]:
                assert arr.chunks == first.chunks, key
                assert (arr.compressor is None) == (first.compressor is None) and (
                    arr.compressor is None
                    or arr.compressor.get_config() == first.compressor.get_config()
                ), f"compressor mismatch for '{key}' — caches built with different codecs"
            out_arr = out_data.require_dataset(
                name=key,
                shape=(n_total, *first.shape[1:]),
                chunks=first.chunks,
                compressor=first.compressor,
                dtype=first.dtype,
            )
            tail = (0,) * (first.ndim - 1)
            t0 = 0
            for arr in arrs:
                src_store = arr.store
                for i in range(arr.shape[0]):
                    src_key = chunk_key(arr, (i, *tail))
                    if src_key in src_store:  # fill-value chunks may be absent
                        out_store[chunk_key(out_arr, (t0 + i, *tail))] = src_store[
                            src_key
                        ]
                t0 += arr.shape[0]
        else:
            data = np.concatenate([arr[:] for arr in arrs], axis=0)
            _ = out_data.array(
                name=key,
                data=data,
                shape=data.shape,
                chunks=data.shape,
                compressor=None,
                dtype=data.dtype,
            )
    return ReplayBuffer(out_root)


def load_replay_buffer(
    dataset_dir: str, use_cache: bool, shape_meta: dict, ctrl_mode: str = "bimanual_push",
) -> ReplayBuffer:
    if not isinstance(dataset_dir, str):
        dirs = [str(d) for d in dataset_dir]
        if len(dirs) > 1:
            print(f"Merging {len(dirs)} replay buffers: {dirs}")
            return _merge_replay_buffers(
                [
                    load_replay_buffer(d, use_cache, shape_meta, ctrl_mode=ctrl_mode)
                    for d in dirs
                ]
            )
        dataset_dir = dirs[0]
    replay_buffer = None
    if use_cache:
        cache_info_str = ""
        cache_zarr_path = os.path.join(dataset_dir, f"cache{cache_info_str}.zarr.zip")
        cache_lock_path = cache_zarr_path + ".lock"
        print("Acquiring lock on cache.")
        with FileLock(cache_lock_path):
            if not os.path.exists(cache_zarr_path):
                try:
                    print("Cache does not exist. Creating!")
                    # store = zarr.DirectoryStore(cache_zarr_path)
                    replay_buffer = _convert_real_to_dp_replay(
                        store=zarr.MemoryStore(),
                        shape_meta=shape_meta,
                        dataset_dir=dataset_dir,
                        ctrl_mode=ctrl_mode,
                    )
                    print("Saving cache to disk.")
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                except Exception as e:
                    shutil.rmtree(cache_zarr_path)
                    raise e
            else:
                print("Loading cached ReplayBuffer from Disk.")
                with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                    replay_buffer = ReplayBuffer.copy_from_store(
                        src_store=zip_store, store=zarr.MemoryStore()
                    )
                print("Loaded!")
    else:
        replay_buffer = _convert_real_to_dp_replay(
            store=zarr.MemoryStore(),
            shape_meta=shape_meta,
            dataset_dir=dataset_dir,
            ctrl_mode=ctrl_mode,
        )
    return replay_buffer


class SimAlohaDataset(BaseImageDataset):
    """A dataset for the real-world data collected on Aloha robot."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        # assign config
        shape_meta = cfg.shape_meta
        dataset_dir = cfg.dataset_dir
        horizon = cfg.horizon * cfg.skip_frame
        pad_before = cfg.pad_before
        pad_after = cfg.pad_after
        use_cache = cfg.use_cache
        self.val_horizon = (
            cfg.val_horizon * cfg.skip_frame if "val_horizon" in cfg else horizon
        )
        self.skip_idx = cfg.skip_idx if "skip_idx" in cfg else 1
        self.aug_mode = cfg.aug_mode
        if cfg.aug_mode == "img_aug":
            self.aug = iaa.Sequential(
                [
                    iaa.Affine(
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-30, 30),
                        mode="edge",
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05), per_channel=0.5
                    ),
                    iaa.MultiplyHueAndSaturation(
                        mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)
                    ),
                    iaa.MultiplyBrightness(mul=(0.8, 1.2)),
                ]
            )
        elif cfg.aug_mode == "none":
            self.aug = None
        else:
            raise ValueError(f"Invalid augmentation mode: {cfg.aug_mode}")

        self.action_mode = cfg.action_mode if "action_mode" in cfg else "bimanual_push"

        # dataset_dir may be a single root or a list of roots, each containing
        # train/ and val/ subdirectories; multiple roots are merged.
        if isinstance(dataset_dir, str):
            train_dir = os.path.join(dataset_dir, "train")
        else:
            train_dir = [os.path.join(str(d), "train") for d in dataset_dir]
        self.replay_buffer = load_replay_buffer(
            train_dir, use_cache, shape_meta, ctrl_mode=self.action_mode
        )

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "depth":
                depth_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        train_mask = np.ones((self.replay_buffer.n_episodes,), dtype=bool)
        all_keys = list(self.replay_buffer.keys())

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            goal_sample=cfg.goal_sample,
            keys=all_keys,
            skip_frame=cfg.skip_frame,
            keys_to_keep_intermediate=["action"],
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_dir = dataset_dir
        self.skip_frame = cfg.skip_frame
        self.goal_sample = cfg.goal_sample
        self.use_cache = use_cache
        self.resolution = cfg.resolution

    def get_normalizer(self, mode: str = "none", **kwargs: dict) -> LinearNormalizer:
        """Return a normalizer for the dataset."""
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])
        this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                # this_normalizer = get_range_normalizer_from_stat(stat)
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("vel"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        for key in self.depth_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        if self.is_val:
            # the number of episodes in the validation set
            return self.replay_buffer.n_episodes // self.skip_idx
        else:
            return len(self.sampler)

    def get_validation_dataset(self) -> "BaseImageDataset":
        """Return a validation dataset."""
        val_set = copy.copy(self)
        val_set.is_val = True
        if isinstance(self.dataset_dir, str):
            val_dir = os.path.join(self.dataset_dir, "val")
        else:
            val_dir = [os.path.join(str(d), "val") for d in self.dataset_dir]
        shape_meta = self.shape_meta
        use_cache = self.use_cache
        val_set.replay_buffer = load_replay_buffer(
            val_dir, use_cache, shape_meta, ctrl_mode=self.action_mode
        )
        val_mask = np.ones((val_set.replay_buffer.n_episodes,), dtype=bool)
        val_set.sampler = SequenceSampler(
            replay_buffer=val_set.replay_buffer,
            sequence_length=self.val_horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=val_mask,
            skip_idx=self.skip_idx,
            goal_sample=self.goal_sample,
            skip_frame=self.skip_frame,
            keys_to_keep_intermediate=["action"],
        )
        val_set.train_mask = val_mask
        return val_set

    def _sample_to_data(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        final_dict = dict()

        # Apply augmentation with 0.2 probability
        apply_aug = np.random.random() < 0.2 if self.aug_mode == "img_aug" else False

        # skip_start = np.random.randint(0, self.skip_frame) + self.skip_frame
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_images = sample[key].astype(np.uint8)
            final_images = sample[f"{key}_final"].astype(np.uint8)

            # Apply augmentation if probability condition is met
            if apply_aug:
                aug_det = self.aug.to_deterministic()
                combined = [*obs_images, final_images]
                # apply the deterministic augmenter separately to each image
                combined_aug = [aug_det.augment_image(img) for img in combined]
                obs_images = np.stack(combined_aug[:-1], axis=0)
                final_images = combined_aug[-1]  # Last image

            obs_dict[key] = np.moveaxis(obs_images, -1, 1).astype(np.float32) / 255.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(final_images, -1, 0).astype(np.float32) / 255.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint16 image to float32
            obs_dict[key] = np.moveaxis(sample[key], -1, 1).astype(np.float32) / 1000.0
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = (
                np.moveaxis(sample[f"{key}_final"], -1, 0).astype(np.float32) / 1000.0
            )
            del sample[f"{key}_final"]
            # T,C,H,W
            del sample[key]
        for key in self.lowdim_keys:
            obs_dict[key] = sample[key].astype(np.float32)
            # obs_dict[key] = obs_dict[key][skip_start :: self.skip_frame]
            final_dict[key] = sample[f"{key}_final"].astype(np.float32)
            del sample[f"{key}_final"]
            del sample[key]

        actions = sample["action"].astype(np.float32)
        # action_dim = actions.shape[-1]
        # downsample_horizon = actions.shape[0] // self.skip_frame - 1
        # action_len = downsample_horizon * self.skip_frame
        # action_start = skip_start - self.skip_frame
        # actions = actions[action_start : action_start + action_len]
        # actions = actions.reshape(downsample_horizon, self.skip_frame, action_dim)
        # actions = actions.reshape(downsample_horizon, self.skip_frame * action_dim)
        data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "goal": dict_apply(final_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "is_early_stop": torch.from_numpy(np.array([sample["is_early_stop"]])),
            "rel_stop_idx": torch.from_numpy(np.array([sample["rel_stop_idx"]])),
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_val:
            epi_idx = idx * self.skip_idx
            epi_start = (
                self.replay_buffer.episode_ends[epi_idx - 1] if epi_idx > 0 else 0
            )
            epi_end = self.replay_buffer.episode_ends[epi_idx]
            epi_len = epi_end - epi_start
            rng = np.random.default_rng(idx)
            offsets = rng.choice(
                epi_len, size=self.val_horizon, replace=epi_len < self.val_horizon
            )
            idxs = epi_start + np.sort(offsets)
            val_horizon = self.val_horizon
            sample = dict()
            for key in self.sampler.keys:
                sample[key] = self.replay_buffer[key][epi_start:epi_end][np.sort(offsets)]
                if sample[key].shape[0] < val_horizon:
                    pad_len = val_horizon - sample[key].shape[0]
                    pad_shape = (pad_len, *np.ones_like(sample[key].shape[1:]).tolist())
                    sample_pad = np.tile(sample[key][-1:], pad_shape)
                    sample[key] = np.concatenate([sample[key], sample_pad], axis=0)
                if key in self.sampler.keys_to_keep_intermediate:
                    inter_frames = sample[key].shape[0] // self.skip_frame
                    sample_shape = list(sample[key].shape[1:])
                    sample_shape[0] = sample_shape[0] * self.skip_frame
                    sample[key] = sample[key].reshape(
                        inter_frames, self.skip_frame, *sample[key].shape[1:]
                    )
                    sample[key] = sample[key].reshape(-1, *sample_shape)
                else:
                    sample[key] = sample[key][:: self.skip_frame]
                sample[f"{key}_final"] = sample[key][-1]
                sample["is_early_stop"] = False
                sample["rel_stop_idx"] = len(idxs) - 1 # note I think this value is not used and wrong
        else:
            sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data


def test_sim_aloha_dataset() -> None:
    config_path = "configurations/dataset/sim_aloha_dataset.yaml"
    cfg = OmegaConf.load(config_path)
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/single_arm_transfer_cube_0407_v3"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_test_0414"  # noqa
    # cfg.dataset_dir = "/media/yixuan/Extreme SSD/projects/diffusion-forcing/data/sim_aloha/pusht_0407"  # noqa
    cfg.dataset_dir = "data/scripted_sim_aloha_10000"
    cfg.horizon = 10
    cfg.shape_meta.action.shape = (4,)
    cfg.skip_frame = 1
    cfg.skip_idx = 4
    cfg.val_horizon = 200
    cfg.goal_sample = "aggressive"
    cfg.resolution = 128
    dataset = SimAlohaDataset(cfg)
    print(len(dataset))

    p = psutil.Process(os.getpid())

    def rss() -> float:
        return p.memory_info().rss / 1e9

    print(f"START RSS: {rss():.3f} GB")
    k = min(2000, len(dataset))  # enough iterations to see creep
    for i in range(k):
        _ = dataset[i]  # exercises __getitem__
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    cpu_count = os.cpu_count()
    assert cpu_count is not None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=min(cpu_count, 16),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        prefetch_factor=1,
    )
    i = 0
    for _ in dataloader:
        i += 1
        if (i + 1) % 50 == 0:
            gc.collect()
            print(f"i={i+1} RSS={rss():.3f} GB")

    val_dataset = dataset.get_validation_dataset()
    print(len(val_dataset))
    for i in range(len(val_dataset)):
        data = val_dataset[i]
    print(data)
    print("validation dataset success!")


if __name__ == "__main__":
    test_sim_aloha_dataset()
