import math
import os
from collections import defaultdict
from collections.abc import Iterator
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rootutils
import torch
import torch.distributed as dist
from pycocotools import mask as mask_util
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, DistributedSampler

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.common.data.base import DataModuleBase, TorchDataset
from src.common.data.classes import TARGET_CLASS_LIST
from src.tasks.pretrain.utils import ToTensor
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TorchSelfSupDataset(TorchDataset):
    """Self-supervised learning dataset class.

    Args:
        root (str): Root directory path of the dataset. Defaults to "your_dataset_root".
        image_folder_name (str): Image folder name. Defaults to "images".
        label_file_name (str): Label file name. Defaults to "labels".
        suffix (Optional[str]): File suffix. Defaults to None.
        split (str): Data split ("train", "val", "test"). Defaults to "train".
        transform (Optional[A.Compose]): Transformations. Defaults to None.
        val_ratio (float): Validation data ratio. Defaults to 0.2.
    """

    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        val_ratio: float = 0.2,
        global_context: Optional[bool] = False,
        global_context_name: Optional[str] = None,
        train_subsample_ratio: Optional[float] = None,
    ):
        self.dataset_name = Path(root).name
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            split=split,
            transform=transform,
            val_ratio=val_ratio,
        )

        # Apply subsampling for train split only
        if split == "train" and train_subsample_ratio is not None:
            assert (
                0 < train_subsample_ratio < 1
            ), "train_subsample_ratio must be between 0 and 1 (exclusive)"
            original_size = len(self.dataframe)
            subsample_size = int(original_size * train_subsample_ratio)
            self.dataframe = self.dataframe.sample(n=subsample_size, random_state=42).reset_index(
                drop=True
            )
            log.info(
                f"Subsampled train dataset from {original_size} to {len(self.dataframe)} samples (ratio: {train_subsample_ratio})"
            )

        self.transform = transform
        self.split = split
        self.global_context = global_context
        self.global_context_latents_path = Path(root) / f"latents_{global_context_name}"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sample for the given index.

        Args:
            idx (int): index of sample

        Returns:
            Dict[str, Any]: sample data
        """
        # Extract necessary data from dataframe (Ensure consistent order)
        data = self.dataframe.iloc[idx]
        sample_id = data["sample_id"]

        # Image processing
        img = cv2.cvtColor(cv2.imread(data["filepath"]), cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = ToTensor(img)
        if img.max() > 1:
            img = img / 255

        # Label processing
        cls_columns = [f"cls_{c}" for c in self.CLASS_NAME_LIST]
        label = torch.tensor(data[cls_columns].values.astype(np.float32), dtype=torch.float32)
        label = None if torch.isnan(label).all() else label

        # Process class name list
        class_name_list = self.CLASS_NAME_LIST if self.CLASS_NAME_LIST else None

        global_context_latents = None
        if self.global_context:
            global_context_latents_path = self.global_context_latents_path / (
                Path(data["filepath"]).stem + ".pt"
            )
            global_context_latents = torch.load(global_context_latents_path, weights_only=True)

        return {
            "img": img,
            "label": label,
            "global_context_latents": global_context_latents,
            "class_name_list": class_name_list,
            "dataset_name": self.dataset_name,
            "filepath": data["filepath"],
            "info": {},
        }

    def train_val_split(
        self, dataframe: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
    ):
        """Split data into train and validation set using random split.

        Args:
            dataframe (pd.DataFrame): dataframe to split
            val_ratio (float, optional): validation ratio. Defaults to 0.2.
            random_state (int, optional): random state. Defaults to 42.
        """
        # Extract only training data
        train_df = dataframe[dataframe["split"] == "train"]
        if len(train_df) == 0:
            log.info(f"[Data Loading] No train data in {self.dataset_name}")
            return dataframe

        # If validation ratio is 0, treat all data as training data
        if val_ratio == 0:
            log.info(f"[Data Loading] No validation split for {self.dataset_name}")
            return dataframe

        # Perform random split
        log.info(f"[Data Loading] Using random split for {self.dataset_name} dataset")
        val_indices = np.random.RandomState(random_state).choice(
            len(train_df), size=int(len(train_df) * val_ratio), replace=False
        )
        val_df_indices = train_df.iloc[val_indices].index
        dataframe.loc[val_df_indices, "split"] = "val"

        return dataframe


class SelfSupMultiDMBase(DataModuleBase):
    """Class for creating a data module for self-supervised learning by combining multiple
    datasets.

    Attributes:
        dataset_list (Dict[str, List[str]]): List of dataset names to use
        data_train (TorchSelfSupDataset): Training dataset
        data_val (TorchSelfSupDataset): Validation dataset
        data_test (TorchSelfSupDataset): Test dataset
    """

    def __init__(
        self,
        dataset_list: Dict[str, List[str]],
        root: str = "your_path/DATASET_UNIFIED_V1",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        val_ratio: float = 0.2,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = None,
        train_subsample_ratio: Optional[float] = None,
    ):
        """Initialize the data module.

        Args:
            root (str): Root directory path for datasets
            dataset_list (Dict[str, List[str]]): List of dataset names to use
            image_folder_name (str): Name of image folder
            label_file_name (str): Name of label file
            suffix (Optional[str]): File suffix
            train_batch_size (int): Batch size for training
            val_batch_size (int): Batch size for validation
            test_batch_size (int): Batch size for testing
            num_workers (int): Number of workers for data loading
            pin_memory (bool): Whether to pin memory to GPU
            persistent_workers (bool): Whether to use persistent workers
            prefetch_factor (int): Prefetch factor for data loading
            val_ratio (float): Validation data ratio
            transforms (Dict[str, Callable[..., Any]]): Dictionary of transform functions
            fill_nan (Number | None): Value to fill NaN values
            train_subsample_ratio (Optional[float]): Ratio for subsampling training data (0 < ratio < 1)
        """
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            transforms=transforms,
            fill_nan=fill_nan,
            val_ratio=val_ratio,
        )
        self.train_subsample_ratio = train_subsample_ratio
        self.dataset_list = dataset_list
        self.target_class = TARGET_CLASS_LIST["finding"] + TARGET_CLASS_LIST["detail_dx"]
        # 데이터셋 리스트 초기화
        self.data_train = []
        self.data_val = []
        self.data_test = []
        self.class_name_list = None
        self.target_idx = {}


class SelfSupMultiDM(SelfSupMultiDMBase):
    """Self-supervised learning data module class.

    Integrates multiple datasets.

    Args:
        dataset_list (Dict[str, List[str]]): List of datasets for train/val/test
        root (str): Root directory path of the dataset. Defaults to "your_path/DATASET_UNIFIED_V1_PREPROC_512x512".
        image_folder_name (str): Image folder name. Defaults to "images".
        label_file_name (str): Label file name. Defaults to "labels".
        suffix (Optional[str]): File suffix. Defaults to None.
        train_batch_size (int): Training batch size. Defaults to 8.
        val_batch_size (int): Validation batch size. Defaults to 8.
        test_batch_size (int): Test batch size. Defaults to 8.
        num_workers (int): Number of data loading workers. Defaults to 0.
        pin_memory (bool): Whether to pin memory. Defaults to False.
        persistent_workers (bool): Whether to keep workers persistent. Defaults to False.
        prefetch_factor (int): Prefetch factor. Defaults to 2.
        val_ratio (float): Validation data ratio. Defaults to 0.1.
        transforms (Optional[Dict[str, Callable]]): Dictionary of data transformation functions. Defaults to dict().
        fill_nan (Optional[Number]): Value to replace NaN with. Defaults to None.
        train_subsample_ratio (Optional[float]): Training data subsampling ratio (0 < ratio < 1). Defaults to None.
    """

    def __init__(
        self,
        dataset_list: Dict[str, List[str]],
        root: str = "your_path/DATASET_UNIFIED_V1_PREPROC_512x512",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        val_ratio: float = 0.1,
        transforms: Optional[Dict[str, Callable]] = dict(),
        fill_nan: Optional[Number] = None,
        global_context: Optional[bool] = False,
        global_context_name: Optional[str] = None,
        train_subsample_ratio: Optional[float] = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            dataset_list=dataset_list,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            val_ratio=val_ratio,
            transforms=transforms,
            fill_nan=fill_nan,
            train_subsample_ratio=train_subsample_ratio,
        )
        self.global_context = global_context
        if global_context_name is not None:
            global_context_names = [
                "flair",
                "dinov2",
                "keepfit",
                "urfound",
                "retfound",
                "retclip",
                "GMI",
                "HMI",
            ]
            able_global_context_name = False
            for name in global_context_names:
                if name in global_context_name:
                    able_global_context_name = True
                    break
            assert (
                able_global_context_name
            ), f"global_context_name must be in {global_context_names}, but got {global_context_name}"
        self.global_context_name = global_context_name

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing.

        Args:
            stage (Optional[str]): Stage identifier ('fit' or 'test')
        """

        class_name_list = set()
        if stage == "fit" or stage is None:
            log.info("==> Dataset Loading for Fit Stage")

            # Train 데이터셋 로드
            num_total_train = 0
            for dataset_name in self.dataset_list["train"]:
                validation_ratio = (
                    self.hparams.val_ratio
                    if dataset_name in self.dataset_list.get("val", [])
                    else 0
                )
                _data_train = TorchSelfSupDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    suffix=self.hparams.suffix,
                    split="train",
                    transform=self.hparams.transforms.get("train", None),
                    val_ratio=validation_ratio,
                    global_context=self.global_context,
                    global_context_name=self.global_context_name,
                    train_subsample_ratio=self.train_subsample_ratio,
                )
                self.data_train.append(_data_train)
                num_total_train += len(_data_train)
                log.info(
                    f"[Data Loading] ==> {dataset_name} train dataset loaded ({len(_data_train)})"
                )

                class_name_list.update(_data_train.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_train.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_train.CLASS_NAME_LIST
                }

            log.info(f"Total train dataset size: {num_total_train}")
            # Load Val dataset
            num_total_val = 0
            for dataset_name in self.dataset_list.get("val", []):
                validation_ratio = (
                    self.hparams.val_ratio
                    if dataset_name in self.dataset_list.get("val", [])
                    else 0
                )
                _data_val = TorchSelfSupDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    suffix=self.hparams.suffix,
                    split="val",
                    transform=self.hparams.transforms.get("val", None),
                    val_ratio=validation_ratio,
                    global_context=self.global_context,
                    global_context_name=self.global_context_name,
                )
                self.data_val.append(_data_val)
                num_total_val += len(_data_val)
                log.info(
                    f"[Data Loading] ==> {dataset_name} val dataset loaded ({len(_data_val)})"
                )

                class_name_list.update(_data_val.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_val.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_val.CLASS_NAME_LIST
                }

            log.info(f"Total val dataset size: {num_total_val}")
            # Merge datasets
            # Concatenate train datasets
            if self.data_train:
                self.data_train = sum(self.data_train[1:], self.data_train[0])

            if self.data_val:
                self.data_val = sum(self.data_val[1:], self.data_val[0])

        if stage == "test" or stage is None:
            log.info("==> Dataset Loading for Test Stage")

            # Load Test dataset
            num_total_test = 0
            for dataset_name in self.dataset_list.get("test", []):
                _data_test = TorchSelfSupDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    suffix=self.hparams.suffix,
                    split="test",
                    transform=self.hparams.transforms.get("test", None),
                )
                self.data_test.append(_data_test)
                num_total_test += len(_data_test)
                log.info(
                    f"[Data Loading] ==> {dataset_name} test dataset loaded ({len(_data_test)})"
                )
                class_name_list.update(_data_test.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_test.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_test.CLASS_NAME_LIST
                }

            if self.data_test:
                self.data_test = sum(self.data_test[1:], self.data_test[0])

            log.info(f"Total test dataset size: {num_total_test}")

        # Update class info
        self.class_name_list = list(class_name_list)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for self-supervised learning.

        Args:
            batch (List[Dict[str, Any]]): Batch of data

        Returns:
            Dict[str, Any]: Collated data
        """
        # Process image and metadata separately
        images = []
        dataset_names = []
        labels = []
        class_name_list = {}
        info = None
        filepaths = []
        global_context_latents = []

        for sample in batch:
            img = sample["img"]
            if img.dim() != 3:
                img = img.unsqueeze(0)
            images.append(img)

            if "info" in sample:
                info = sample["info"]
            if "dataset_name" in sample:
                dataset_names.append(sample["dataset_name"])
            if "label" in sample:
                labels.append(self.get_target_label(sample["label"], sample["dataset_name"]))
            if "class_name_list" in sample:
                class_name_list[sample["dataset_name"]] = sample["class_name_list"]
            if "filepath" in sample:
                filepaths.append(sample["filepath"])
            if ("global_context_latents" in sample) and (
                sample["global_context_latents"] is not None
            ):
                global_context_latents.append(sample["global_context_latents"])

        # Construct result dictionary
        out = {
            "img": torch.stack(images),
            "dataset_name": dataset_names,
            "info": info,
            "class_name_list": class_name_list,
            "filepath": filepaths,
            "global_context_latents": (
                torch.stack(global_context_latents) if global_context_latents else None
            ),
        }

        label_tensor = torch.stack(labels)
        label_new, valid_mask = self._process_labels(label_tensor)
        out.update({"label": label_new, "valid_mask": valid_mask})
        return out

    def _process_labels(self, label_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Label processing and valid mask generation.

        Args:
            label_target (torch.Tensor): Target label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: New label and valid mask
        """
        # Process only if fill_nan is not None
        if self.hparams.fill_nan is not None:
            # Fill NaN values and generate valid_mask at once
            label_target = torch.nan_to_num(label_target, nan=self.hparams.fill_nan)
            valid_mask = (
                torch.ones_like(label_target)
                if self.hparams.fill_nan >= 0
                else ~torch.isnan(label_target)
            )
        else:
            valid_mask = ~torch.isnan(label_target)

        return label_target, valid_mask.float()

    def get_target_label(
        self, raw_label: Optional[torch.Tensor] = None, dataset_name: str = None
    ) -> torch.Tensor:
        """Generate target label.

        Args:
            raw_label (torch.Tensor): Raw label
            dataset_name (str): Dataset name
        Returns:
            torch.Tensor: Target label
        """
        # Handle case where target_idx is missing
        if (
            not hasattr(self, "target_idx")
            or not self.target_idx[dataset_name]
            or raw_label is None
        ):
            return torch.full((len(self.target_class),), torch.nan, dtype=torch.float)

        # Initialize result tensor - Keep batch size of raw_label
        label = torch.full((len(self.target_class),), torch.nan, dtype=torch.float)

        # Create mapping between target_class and target_idx
        target_map = {
            l: i for i, l in enumerate(self.target_class) if l in self.target_idx[dataset_name]
        }

        if target_map:
            # Extract mapped indices
            target_indices = []
            source_indices = []
            for target_class, target_idx in target_map.items():
                target_indices.append(target_idx)
                source_indices.append(self.target_idx[dataset_name][target_class])

            # Assign values using indexing at once
            label[target_indices] = raw_label[source_indices]

        return label


class TorchTransferLearningDataset(TorchSelfSupDataset):
    def __init__(
        self,
        root: str,
        image_folder_name: str,
        label_file_name: str,
        target_class: List[str],
        suffix: Optional[str] = None,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            split=split,
            transform=transform,
            val_ratio=0,
        )
        self.CLASS_NAME_LIST = None
        self.target_class = sorted(target_class)
        self.drop_and_proc_dataframe()

    def proc_labels(self):
        """Process labels."""
        self.dataframe.columns = [col.replace("cls_", "") for col in self.dataframe.columns]

    def drop_and_proc_dataframe(self):
        """Drop samples with NaN class and process dataframe."""
        # Get lesion columns
        # process diabetic retinopathy class
        self.dataframe = self._proc_diabetic_retinopathy(dataframe=self.dataframe)
        self.dataframe = self._proc_glaucoma(dataframe=self.dataframe)
        self.dataframe = self._proc_retinal_vein_occlusion(dataframe=self.dataframe)
        self.dataframe = self._proc_age_related_macular_degeneration(dataframe=self.dataframe)
        lesion_cols = [
            col for col in self.dataframe.columns if col.split("_")[-1] in self.target_class
        ]
        self.CLASS_NAME_LIST = [col.split("_")[-1] for col in lesion_cols]
        # Drop samples with NaN class
        self.dataframe = self.dataframe[self.dataframe[lesion_cols].notna().all(axis=1)][
            [c for c in self.dataframe.columns if not c.startswith("cls_")] + lesion_cols
        ]

        for c in lesion_cols:
            log.info(
                f"[Data Loading] ==> {self.dataset_name} loads {c.split('_')[-1]} ({(self.dataframe[c] == 1).sum()})"
            )

    def _get_lesion_cols(self, dataframe: pd.DataFrame) -> List[str]:
        """Get lesion columns."""
        return [
            c
            for c in dataframe.columns
            if (
                (("DiabeticRetinopathy" in c) and (c != "cls_DiabeticRetinopathy_0"))
                or (
                    (c.startswith("cls_Glaucoma"))
                    and (not c.startswith("cls_GlaucomatousDiscChange"))
                )
                or ("RetinalVeinOcclusion" in c)
                or ("Age-relatedMacularDegeneration" in c)
            )
        ]

    def _proc_age_related_macular_degeneration(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process Age-relatedMacularDegeneration class."""
        lesion_cols = self._get_lesion_cols(dataframe)
        if "cls_Age-relatedMacularDegeneration" in lesion_cols:
            amd_cols_to_drop = [
                col
                for col in lesion_cols
                if ("Age-relatedMacularDegeneration" in col)
                and (col != "cls_Age-relatedMacularDegeneration")
            ]
            return dataframe.drop(columns=amd_cols_to_drop) if amd_cols_to_drop else dataframe

        amd_cols = [col for col in lesion_cols if ("Age-relatedMacularDegeneration" in col)]
        if amd_cols:
            dataframe["cls_Age-relatedMacularDegeneration"] = pd.NA

            has_lesion = (dataframe[amd_cols] == 1).any(axis=1)
            no_lesion = (dataframe[amd_cols] == 0).all(axis=1)

            dataframe.loc[has_lesion, "cls_Age-relatedMacularDegeneration"] = 1
            dataframe.loc[no_lesion, "cls_Age-relatedMacularDegeneration"] = 0

        amd_cols_to_drop = [
            col
            for col in lesion_cols
            if ("Age-relatedMacularDegeneration" in col)
            and (col != "cls_Age-relatedMacularDegeneration")
        ]
        if amd_cols_to_drop:
            dataframe = dataframe.drop(columns=amd_cols_to_drop)

        return dataframe

    def _proc_retinal_vein_occlusion(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process RetinalVeinOcclusion class."""
        lesion_cols = self._get_lesion_cols(dataframe)
        if "cls_RetinalVeinOcclusion" in lesion_cols:
            rvo_cols_to_drop = [
                col
                for col in lesion_cols
                if ("RetinalVeinOcclusion" in col) and (col != "cls_RetinalVeinOcclusion")
            ]
            return dataframe.drop(columns=rvo_cols_to_drop) if rvo_cols_to_drop else dataframe

        rvo_cols = [col for col in lesion_cols if ("RetinalVeinOcclusion" in col)]
        if rvo_cols:
            dataframe["cls_RetinalVeinOcclusion"] = pd.NA

            has_lesion = (dataframe[rvo_cols] == 1).any(axis=1)
            no_lesion = (dataframe[rvo_cols] == 0).all(axis=1)

            dataframe.loc[has_lesion, "cls_RetinalVeinOcclusion"] = 1
            dataframe.loc[no_lesion, "cls_RetinalVeinOcclusion"] = 0

        rvo_cols_to_drop = [
            col
            for col in lesion_cols
            if ("RetinalVeinOcclusion" in col) and (col != "cls_RetinalVeinOcclusion")
        ]
        if rvo_cols_to_drop:
            dataframe = dataframe.drop(columns=rvo_cols_to_drop)

        return dataframe

    def _proc_glaucoma(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process Glaucoma class."""
        lesion_cols = self._get_lesion_cols(dataframe)

        # If aggregated retinopathy column already exists
        if "cls_GlaucomaSuspect" in lesion_cols:
            gl_cols_to_drop = [
                col
                for col in lesion_cols
                if ("Glaucoma" in col) and (col != "cls_GlaucomaSuspect")
            ]
            return dataframe.drop(columns=gl_cols_to_drop) if gl_cols_to_drop else dataframe

        gl_cols = [
            col
            for col in lesion_cols
            if ("Glaucoma" in col) and (not col.startswith("cls_GlaucomatousDiscChange"))
        ]
        if gl_cols:
            dataframe["cls_GlaucomaSuspect"] = pd.NA

            has_lesion = (dataframe[gl_cols] == 1).any(axis=1)
            no_lesion = (dataframe[gl_cols] == 0).all(axis=1)

            dataframe.loc[has_lesion, "cls_GlaucomaSuspect"] = 1
            dataframe.loc[no_lesion, "cls_GlaucomaSuspect"] = 0

        gl_cols_to_drop = [
            col for col in lesion_cols if ("Glaucoma" in col) and (col != "cls_GlaucomaSuspect")
        ]
        if gl_cols_to_drop:
            dataframe = dataframe.drop(columns=gl_cols_to_drop)

        return dataframe

    def _proc_diabetic_retinopathy(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process Diabetic Retinopathy class.

        Args:
            dataframe (pd.DataFrame): Dataframe to process

        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Extract columns related to Diabetic Retinopathy
        lesion_cols = self._get_lesion_cols(dataframe)

        # If aggregated Diabetic Retinopathy column already exists
        if "cls_DiabeticRetinopathy" in lesion_cols:
            # Drop unnecessary columns
            dr_cols_to_drop = [
                col
                for col in lesion_cols
                if ("DiabeticRetinopathy" in col) and (col != "cls_DiabeticRetinopathy")
            ]
            return dataframe.drop(columns=dr_cols_to_drop) if dr_cols_to_drop else dataframe

        # Consolidate individual lesion columns
        dr_cols = [
            col
            for col in lesion_cols
            if ("DiabeticRetinopathy" in col) and (col != "cls_DiabeticRetinopathy_0")
        ]
        if dr_cols:
            # Initialize consolidated column
            dataframe["cls_DiabeticRetinopathy"] = pd.NA

            # Set to 1 if any lesion exists, 0 if all are absent
            has_lesion = (dataframe[dr_cols] == 1).any(axis=1)
            no_lesion = (dataframe[dr_cols] == 0).all(axis=1)

            dataframe.loc[has_lesion, "cls_DiabeticRetinopathy"] = 1
            dataframe.loc[no_lesion, "cls_DiabeticRetinopathy"] = 0

        # Drop individual lesion columns
        dr_cols_to_drop = [
            col
            for col in lesion_cols
            if ("DiabeticRetinopathy" in col) and (col != "cls_DiabeticRetinopathy")
        ]
        if dr_cols_to_drop:
            dataframe = dataframe.drop(columns=dr_cols_to_drop)

        return dataframe

    def train_val_split(
        self, dataframe: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
    ):
        """Split data into train and validation set.

        First, get indices of samples with lesions.
        Then, split samples with lesions.
        Finally, sample equal number of no-lesion samples.

        Args:
            dataframe (pd.DataFrame): dataframe
            val_ratio (float, optional): validation ratio. Defaults to 0.2.
            random_state (int, optional): random state. Defaults to 42.
        """
        if not val_ratio:
            return

        train_df = dataframe[dataframe["split"] == "train"]

        # Get lesion columns
        lesion_cols = self._get_lesion_cols(train_df)

        # Get indices of samples with lesions
        lesion_df = train_df[train_df[lesion_cols].sum(axis=1) > 0].copy()

        # Filter out samples with a single class
        lesion_df["num_class"] = lesion_df[lesion_cols].sum(axis=1)

        # Filter out samples with a single class
        value_counts = lesion_df["num_class"].value_counts()
        single_class = value_counts[value_counts == 1].index
        lesion_df = lesion_df[~lesion_df["num_class"].isin(single_class)]

        # Split samples with lesions
        split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        val_indices = next(split.split(lesion_df, lesion_df["num_class"]))[1]

        # Get indices of samples without lesions
        no_lesion_df = train_df.loc[
            ~train_df.index.isin(lesion_df.index) & train_df[lesion_cols].notnull().all(axis=1)
        ]

        # Sample equal number of no-lesion samples
        sampled_no_lesion_indices = no_lesion_df.sample(
            n=len(val_indices), random_state=random_state
        ).index

        # Combine indices and update split column
        val_indices = list(set(lesion_df.index[val_indices]).union(set(sampled_no_lesion_indices)))
        dataframe.loc[val_indices, "split"] = "val"


class TorchTLSegmentationDataset(TorchSelfSupDataset):
    """Transfer Learning Segmentation Dataset.

    This dataset handles loading and processing datasets for transfer learning tasks.
    It supports multiple datasets, stratified batch sampling, and customizable transformations.
    """

    def __init__(
        self,
        root: str,
        image_folder_name: str,
        label_file_name: str,
        target_class: List[str],
        suffix: Optional[str] = None,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            split=split,
            transform=transform,
            val_ratio=0.2,
        )
        self.target_class = TARGET_CLASS_LIST["qunat"]
        self.CLASS_NAME_LIST = None
        self.target_class = sorted(target_class)
        self.drop_and_proc_dataframe()

    def proc_labels(self):
        """Process labels."""
        self.dataframe.columns = [
            col.replace("mask_counts_", "").replace("mask_size_", "")
            for col in self.dataframe.columns
        ]

    def _get_lesion_cols(self, dataframe: pd.DataFrame) -> List[str]:
        """Get lesion columns."""
        return [
            c
            for c in dataframe.columns
            if (
                ("mask_counts_CottonWoolPatch" in c)
                or ("mask_counts_HardExudate" in c)
                or ("mask_counts_Hemorrhage" in c)
                or ("mask_counts_Microaneurysm" in c)
                or ("mask_counts_Vessel" in c)
            )
        ]

    def train_val_split(
        self, dataframe: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
    ):
        """Split data into train and validation set.

        First, get indices of samples with lesions.
        Then, split samples with lesions.
        Finally, sample equal number of no-lesion samples.

        Args:
            dataframe (pd.DataFrame): dataframe
            val_ratio (float, optional): validation ratio. Defaults to 0.2.
            random_state (int, optional): random state. Defaults to 42.
        """
        if not val_ratio:
            return

        train_df = dataframe[dataframe["split"] == "train"]
        # Get lesion columns
        lesion_cols = self._get_lesion_cols(train_df)
        # Get indices of samples with lesions
        lesion_df = train_df[train_df[lesion_cols].notna().any(axis=1)].copy()
        # Randomly sample validation set
        val_size = int(len(lesion_df) * val_ratio)
        val_indices = lesion_df.sample(n=val_size, random_state=random_state).index
        dataframe.loc[val_indices, "split"] = "val"

    def drop_and_proc_dataframe(self):
        """Drop samples with NaN class and process dataframe."""
        # Get lesion columns
        lesion_cols = [
            col
            for col in self.dataframe.columns
            if (col.split("_")[-1] in self.target_class) and (col.startswith("mask_counts_"))
        ]
        self.CLASS_NAME_LIST = self.target_class
        # Drop samples with NaN class
        self.dataframe = self.dataframe[self.dataframe[lesion_cols].notna().any(axis=1)]

        for c in lesion_cols:
            log.info(
                f"[Data Loading] ==> {self.dataset_name} loads {c.split('_')[-1]} ({(self.dataframe[c].notna()).sum()})"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sample for the given index.

        Args:
            idx (int): index of sample

        Returns:
            Dict[str, Any]: sample data
        """
        # 데이터프레임에서 필요한 데이터 추출
        sample_id = list(self.dataframe["sample_id"].values)[idx]
        data = self.dataframe.set_index("sample_id").loc[sample_id]

        # 이미지 처리
        img = cv2.cvtColor(cv2.imread(data["filepath"]), cv2.COLOR_BGR2RGB)
        # mask
        c = self.CLASS_NAME_LIST[0]
        masks = []
        valid_masks = []
        for c in self.CLASS_NAME_LIST:
            mask_size = data[f"mask_size_{c}"]
            mask_counts = data[f"mask_counts_{c}"]
            # Convert RLE to binary mask using pycocotools
            if pd.isna(mask_counts):
                mask = np.ones((img.shape[0], img.shape[1])) * -1
                valid_mask = torch.tensor(0)
            else:
                mask = mask_util.decode({"size": mask_size, "counts": mask_counts})
                valid_mask = torch.tensor(1)
            masks.append(mask)
            valid_masks.append(valid_mask)

        mask = np.stack(masks, axis=0)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
        else:
            transformed = {"image": img, "mask": mask}

        img = ToTensor(transformed["image"])
        mask = torch.from_numpy(transformed["mask"]).long()
        valid_mask = torch.stack(valid_masks, axis=0)

        if img.max() > 1:
            img = img / 255

        # 클래스 이름 리스트 처리
        class_name_list = self.CLASS_NAME_LIST if self.CLASS_NAME_LIST else None

        return {
            "img": img,
            "mask": mask,
            "valid_mask": valid_mask,
            "class_name_list": class_name_list,
            "dataset_name": self.dataset_name,
            "filepath": data["filepath"],
            "info": {},
        }


class TransferLearningDMBase(DataModuleBase):
    """Base class for transfer learning data modules.

    This class provides the foundation for transfer learning data modules,
    handling dataset configuration, target class management, and data splitting.
    """

    def __init__(
        self,
        dataset_list: Dict[str, List[str]],
        root: str = "your_path/DATASET_UNIFIED_V1",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        target_class: List[str] = ["DiabeticRetinopathy"],
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = None,
    ):
        """Initialize the TransferLearningDMBase.

        Args:
            dataset_list: Dictionary mapping split names to dataset names
            root: Root directory for datasets
            image_folder_name: Name of the folder containing images
            label_file_name: Name of the label file
            suffix: Optional suffix for dataset paths
            target_class: List of target classes to predict
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in data loader
            persistent_workers: Whether to maintain worker processes between iterations
            prefetch_factor: Number of samples to prefetch per worker
            val_ratio: Ratio of validation data
            transforms: Dictionary of transformation functions
            fill_nan: Value to fill NaN entries with, or None to keep NaNs
        """
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            transforms=transforms,
            fill_nan=fill_nan,
        )
        self.dataset_list = dataset_list
        self.target_class = target_class
        self.class_name_list = None
        self.target_idx = {}
        self.data_train = {}
        self.data_val = {}
        self.data_test = {}


class TransferLearningDM(TransferLearningDMBase):
    """Transfer Learning Data Module.

    This data module handles loading and processing datasets for transfer learning tasks.
    It supports multiple datasets, stratified batch sampling, and customizable transformations.

    Attributes:
        stratified_batch_sampling (bool): Whether to use stratified batch sampling
        dataset_list (Dict[str, List[str]]): Dictionary mapping split names to dataset names
        target_class (List[str]): List of target classes to predict
        class_name_list (List[str]): List of all class names across datasets
        target_idx (Dict[str, Dict[str, int]]): Mapping of dataset names to target class indices
        data_train (Dict[str, Dataset]): Training datasets
        data_val (Dict[str, Dataset]): Validation datasets
        data_test (Dict[str, Dataset]): Test datasets
    """

    def __init__(
        self,
        dataset_list: Dict[str, List[str]],
        root: str = "your_path/DATASET_UNIFIED_V1",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        target_class: List[str] = ["DiabeticRetinopathy"],
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = -1,
        stratified_batch_sampling: bool = True,
    ):
        """Initialize the TransferLearningDM.

        Args:
            dataset_list: Dictionary mapping split names to dataset names
            root: Root directory for datasets
            image_folder_name: Name of the folder containing images
            label_file_name: Name of the label file
            suffix: Optional suffix for dataset paths
            target_class: List of target classes to predict
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loading
            persistent_workers: Whether to use persistent workers
            prefetch_factor: Prefetch factor for data loading
            val_ratio: Ratio of validation data
            transforms: Dictionary of transformations for each split
            fill_nan: Value to fill NaN values with
            stratified_batch_sampling: Whether to use stratified batch sampling
        """
        super().__init__(
            dataset_list=dataset_list,
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            target_class=target_class,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            transforms=transforms,
            fill_nan=fill_nan,
        )
        self.stratified_batch_sampling = stratified_batch_sampling

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing.

        Args:
            stage (Optional[str]): Stage identifier ('fit' or 'test')
        """

        class_name_list = set()
        if stage == "fit" or stage is None:
            log.info("==> Dataset Loading for Fit Stage")

            # Load train datasets
            num_total_train = 0
            for dataset_name in self.dataset_list["train"]:
                _data_train = TorchTransferLearningDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    target_class=self.target_class,
                    suffix=self.hparams.suffix,
                    split="train",
                    transform=self.hparams.transforms.get("train", None),
                )
                self.data_train[dataset_name] = _data_train
                num_total_train += len(_data_train)
                log.info(
                    f"[Data Loading] ==> {dataset_name} train dataset loaded ({len(_data_train)})"
                )

                class_name_list.update(_data_train.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_train.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_train.CLASS_NAME_LIST
                }
            log.info(f"Total train dataset size: {num_total_train}")

        if stage == "test" or stage is None:
            log.info("==> Dataset Loading for Test Stage")

            # Load test datasets
            num_total_test = 0
            for dataset_name in self.dataset_list.get("test", []):
                _data_test = TorchTransferLearningDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    target_class=self.target_class,
                    suffix=self.hparams.suffix,
                    split="test",
                    transform=self.hparams.transforms.get("test", None),
                )
                self.data_test[dataset_name] = _data_test
                num_total_test += len(_data_test)
                log.info(
                    f"[Data Loading] ==> {dataset_name} test dataset loaded ({len(_data_test)})"
                )

                class_name_list.update(_data_test.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_test.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_test.CLASS_NAME_LIST
                }

            log.info(f"Total test dataset size: {num_total_test}")

        # Update class information
        self.class_name_list = list(class_name_list)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for self-supervised learning.

        Args:
            batch (List[Dict[str, Any]]): Batch of data

        Returns:
            Dict[str, Any]: Collated data
        """
        # Process images and metadata separately
        images = []
        dataset_names = []
        labels = []
        class_name_list = {}
        info = None
        filepaths = []

        for sample in batch:
            img = sample["img"]
            if img.dim() != 3:
                img = img.unsqueeze(0)
            images.append(img)

            if "info" in sample:
                info = sample["info"]
            if "dataset_name" in sample:
                dataset_names.append(sample["dataset_name"])
            if "label" in sample:
                labels.append(self.get_target_label(sample["label"], sample["dataset_name"]))
            if "class_name_list" in sample:
                class_name_list[sample["dataset_name"]] = sample["class_name_list"]
            if "filepath" in sample:
                filepaths.append(sample["filepath"])

        # Construct result dictionary
        out = {
            "img": torch.stack(images),
            "dataset_name": dataset_names,
            "info": info,
            "class_name_list": class_name_list,
            "filepath": filepaths,
        }

        label_tensor = torch.stack(labels).nan_to_num(nan=self.hparams.fill_nan)
        label_new, valid_mask = self._process_labels(label_tensor)
        out.update({"label": label_new, "valid_mask": valid_mask})
        return out

    def _process_labels(self, label_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process labels and create valid mask.

        Args:
            label_target (torch.Tensor): Target labels

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed labels and valid mask
        """
        # Process only if fill_nan is not None
        if self.hparams.fill_nan is not None:
            # Fill nan values and create valid_mask in one step
            label_target = torch.nan_to_num(label_target, nan=self.hparams.fill_nan)
            valid_mask = (
                torch.ones_like(label_target)
                if self.hparams.fill_nan >= 0
                else ~torch.isnan(label_target)
            )
        else:
            valid_mask = ~torch.isnan(label_target)

        return label_target, valid_mask.float()

    def get_target_label(
        self, raw_label: Optional[torch.Tensor] = None, dataset_name: str = None
    ) -> torch.Tensor:
        """Generate target labels.

        Args:
            raw_label (torch.Tensor): Original labels
            dataset_name (str): Dataset name
        Returns:
            torch.Tensor: Target labels
        """
        # Handle case when target_idx is not available
        if (
            not hasattr(self, "target_idx")
            or not self.target_idx[dataset_name]
            or raw_label is None
        ):
            return torch.full((len(self.target_class),), torch.nan, dtype=torch.float)

        # Initialize result tensor - maintain batch size of raw_label
        label = torch.full((len(self.target_class),), torch.nan, dtype=torch.float)

        # Create mapping between target_class and target_idx
        target_map = {
            l: i for i, l in enumerate(self.target_class) if l in self.target_idx[dataset_name]
        }

        if target_map:
            # Extract mapped indices
            target_indices = []
            source_indices = []
            for target_class, target_idx in target_map.items():
                target_indices.append(target_idx)
                source_indices.append(self.target_idx[dataset_name][target_class])

            # Assign values using indexing at once
            label[target_indices] = raw_label[source_indices]

        return label

    def train_dataloader(self):
        datasets = []
        labels = []
        for v in self.data_train.values():
            for d in v:
                datasets.append(d)
                labels.append(d["label"])

        labels = torch.cat(labels)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Stratified Sampler 사용
        if self.stratified_batch_sampling:
            sampler = StratifiedDistributedSampler(
                dataset=datasets,
                labels=labels.numpy(),
                batch_size=self.hparams.train_batch_size,
                pos_ratio=None,
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
            )
        else:
            sampler = DistributedSampler(
                dataset=datasets,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )

        return DataLoader(
            datasets,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            worker_init_fn=self.worker_init_fn,
            sampler=sampler,
            batch_size=self.hparams.train_batch_size,
            shuffle=False,
        )

    def val_dataloader(self) -> Dict[str, torch.utils.data.DataLoader]:
        return

    def test_dataloader(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Test dataloader.

        Returns:
            Dict[str, torch.utils.data.DataLoader]: test dataloader
        """
        return {
            k: DataLoader(
                v,
                collate_fn=self.collate_fn,
                batch_size=self.hparams.test_batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                persistent_workers=self.hparams.persistent_workers,
                prefetch_factor=self.hparams.prefetch_factor,
                worker_init_fn=self.worker_init_fn,
                shuffle=False,
            )
            for k, v in self.data_test.items()
        }


class TorchTLSegmentationDM(TransferLearningDM):
    """Transfer Learning Segmentation Data Module.

    This data module handles loading and processing datasets for transfer learning tasks.
    It supports multiple datasets, stratified batch sampling, and customizable transformations.
    """

    def __init__(
        self,
        dataset_list: Dict[str, List[str]],
        root: str = "your_path/DATASET_UNIFIED_V1",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        target_class: List[str] = ["DiabeticRetinopathy"],
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = -1,
    ):
        super().__init__(
            dataset_list=dataset_list,
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            target_class=target_class,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            transforms=transforms,
            fill_nan=fill_nan,
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation and testing.

        Args:
            stage (Optional[str]): Stage identifier ('fit' or 'test')
        """

        class_name_list = set()
        if stage == "fit" or stage is None:
            log.info("==> Dataset Loading for Fit Stage")

            # Load train datasets
            num_total_train = 0
            for dataset_name in self.dataset_list["train"]:
                _data_train = TorchTLSegmentationDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    target_class=self.target_class,
                    suffix=self.hparams.suffix,
                    split="train",
                    transform=self.hparams.transforms.get("train", None),
                )
                self.data_train[dataset_name] = _data_train
                num_total_train += len(_data_train)
                log.info(
                    f"[Data Loading] ==> {dataset_name} train dataset loaded ({len(_data_train)})"
                )

                class_name_list.update(_data_train.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_train.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_train.CLASS_NAME_LIST
                }

            log.info(f"Total train dataset size: {num_total_train}")
            # Load validation datasets
            num_total_val = 0
            for dataset_name in self.dataset_list.get("val", []):
                _data_val = TorchTLSegmentationDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    target_class=self.target_class,
                    suffix=self.hparams.suffix,
                    split="val",
                    transform=self.hparams.transforms.get("val", None),
                )
                self.data_val[dataset_name] = _data_val
                num_total_val += len(_data_val)
                log.info(
                    f"[Data Loading] ==> {dataset_name} val dataset loaded ({len(_data_val)})"
                )

                class_name_list.update(_data_val.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_val.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_val.CLASS_NAME_LIST
                }

            log.info(f"Total val dataset size: {num_total_val}")

        if stage == "test" or stage is None:
            log.info("==> Dataset Loading for Test Stage")

            # Load test datasets
            num_total_test = 0
            for dataset_name in self.dataset_list.get("test", []):
                _data_test = TorchTLSegmentationDataset(
                    root=os.path.join(self.hparams.root, dataset_name),
                    image_folder_name=self.hparams.image_folder_name,
                    label_file_name=self.hparams.label_file_name,
                    target_class=self.target_class,
                    suffix=self.hparams.suffix,
                    split="test",
                    transform=self.hparams.transforms.get("test", None),
                )
                self.data_test[dataset_name] = _data_test
                num_total_test += len(_data_test)
                log.info(
                    f"[Data Loading] ==> {dataset_name} test dataset loaded ({len(_data_test)})"
                )

                class_name_list.update(_data_test.CLASS_NAME_LIST)
                self.target_idx[dataset_name] = {
                    c: _data_test.CLASS_NAME_LIST.index(c)
                    for c in self.target_class
                    if c in _data_test.CLASS_NAME_LIST
                }

            log.info(f"Total test dataset size: {num_total_test}")

        # Update class information
        self.class_name_list = list(class_name_list)

    def train_dataloader(self):
        datasets = []
        for v in self.data_train.values():
            for d in v:
                datasets.append(d)

        return DataLoader(
            datasets,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            worker_init_fn=self.worker_init_fn,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        datasets = []
        for v in self.data_val.values():
            for d in v:
                datasets.append(d)

        return DataLoader(
            datasets,
            collate_fn=self.collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            worker_init_fn=self.worker_init_fn,
            batch_size=self.hparams.train_batch_size,
            shuffle=False,
        )

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for self-supervised learning.

        Args:
            batch (List[Dict[str, Any]]): Batch of data

        Returns:
            Dict[str, Any]: Collated data
        """
        # Process images and metadata separately
        images = []
        dataset_names = []
        masks = []
        valid_masks = []
        class_name_list = {}
        info = None
        filepaths = []

        for sample in batch:
            img = sample["img"]
            if img.dim() != 3:
                img = img.unsqueeze(0)
            images.append(img)

            if "info" in sample:
                info = sample["info"]
            if "dataset_name" in sample:
                dataset_names.append(sample["dataset_name"])
            if "mask" in sample:
                masks.append(sample["mask"])
            if "valid_mask" in sample:
                valid_masks.append(sample["valid_mask"])
            if "class_name_list" in sample:
                class_name_list[sample["dataset_name"]] = sample["class_name_list"]
            if "filepath" in sample:
                filepaths.append(sample["filepath"])

        # Construct result dictionary
        out = {
            "img": torch.stack(images),
            "mask": torch.stack(masks),
            "valid_mask": torch.stack(valid_masks),
            "dataset_name": dataset_names,
            "info": info,
            "class_name_list": class_name_list,
            "filepath": filepaths,
        }
        return out


class StratifiedDistributedSampler(DistributedSampler):
    """Stratified sampler for distributed training that maintains class balance.

    This sampler ensures that each batch contains a specified ratio of positive samples,
    and distributes the batches evenly across all processes in distributed training.
    """

    def __init__(
        self,
        dataset: Any,
        labels: np.ndarray,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        pos_ratio: Optional[float] = None,
        drop_last: bool = False,
    ) -> None:
        """Initialize the stratified distributed sampler.

        Args:
            dataset: Dataset to sample from
            labels: Array of labels for each sample
            batch_size: Number of samples per batch
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process
            shuffle: Whether to shuffle the indices
            seed: Random seed for reproducibility
            pos_ratio: Target ratio of positive samples in each batch (defaults to dataset ratio)
            drop_last: Whether to drop the last incomplete batch
        """
        assert isinstance(labels, np.ndarray), "labels must be a numpy array"
        self.labels = labels
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        self.pos_ratio = pos_ratio or (len(self.pos_indices) / len(labels))
        self.batch_size = batch_size
        self.drop_last = drop_last

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[int]:
        """Create an iterator over the indices.

        Returns:
            Iterator over the indices for the current process
        """
        if self.shuffle:
            rng = np.random.default_rng(self.epoch + self.seed)
            pos_indices = rng.permutation(self.pos_indices)
            neg_indices = rng.permutation(self.neg_indices)
        else:
            pos_indices = self.pos_indices
            neg_indices = self.neg_indices

        pos_per_batch = max(1, math.ceil(self.batch_size * self.pos_ratio))
        neg_per_batch = max(1, self.batch_size - pos_per_batch)  # 최소 한개는 보장

        if self.drop_last:
            total_batches = min(
                len(pos_indices) // pos_per_batch,
                len(neg_indices) // neg_per_batch,
            )
        else:
            total_batches = math.ceil(
                max(len(pos_indices) / pos_per_batch, len(neg_indices) / neg_per_batch)
            )

        needed_pos = total_batches * pos_per_batch
        needed_neg = total_batches * neg_per_batch

        if len(pos_indices) < needed_pos:
            num_additional = needed_pos - len(pos_indices)
            additional = np.random.choice(pos_indices, size=num_additional, replace=True)
            pos_indices = np.concatenate([pos_indices, additional])
        if len(neg_indices) < needed_neg:
            num_additional = needed_neg - len(neg_indices)
            additional = np.random.choice(neg_indices, size=num_additional, replace=True)
            neg_indices = np.concatenate([neg_indices, additional])

        pos_indices = pos_indices[:needed_pos]
        neg_indices = neg_indices[:needed_neg]

        batches = []
        for i in range(total_batches):
            batch = []
            batch.extend(pos_indices[i * pos_per_batch : (i + 1) * pos_per_batch])
            batch.extend(neg_indices[i * neg_per_batch : (i + 1) * neg_per_batch])
            rng.shuffle(batch)
            batches.append(batch)

        # 모든 배치 flatten
        all_indices = [idx for batch in batches for idx in batch]

        # total_size를 num_replicas로 나누어떨어지도록 보정
        total_size = math.ceil(len(all_indices) / self.num_replicas) * self.num_replicas

        if len(all_indices) < total_size:
            all_indices += all_indices[: (total_size - len(all_indices))]

        # rank별로 정확히 균등 분배
        indices_rank = all_indices[self.rank : total_size : self.num_replicas]

        return iter(indices_rank)

    def __len__(self) -> int:
        """Calculate the length of the sampler for the current process.

        Returns:
            Number of samples in the dataset accessible by the current process
        """
        pos_per_batch = max(1, math.ceil(self.batch_size * self.pos_ratio))
        neg_per_batch = max(1, self.batch_size - pos_per_batch)

        if self.drop_last:
            total_batches = min(
                len(self.pos_indices) // pos_per_batch,
                len(self.neg_indices) // neg_per_batch,
            )
        else:
            total_batches = math.ceil(
                max(
                    len(self.pos_indices) / pos_per_batch,
                    len(self.neg_indices) / neg_per_batch,
                )
            )

        total_size = total_batches * self.batch_size
        total_size = math.ceil(total_size / self.num_replicas) * self.num_replicas

        return total_size // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


if __name__ == "__main__":
    # ["CottonWoolPatch", "HardExudate", "Hemorrhage", "Microaneurysm", "Vessel"],
    import logging

    # 로거 설정
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 포맷터 설정
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    log.addHandler(console_handler)

    config = {
        "dataset_list": {
            # CottonWoolPatch
            # "train": ["IDRiD"],
            # "val": ["IDRiD"],
            # "test": ["IDRiD"],
            # HardExudate
            # "train": ["IDRiD", "ADAM"],
            # "val": ["IDRiD", "ADAM"],
            # "test": ["IDRiD", "ADAM"],
            # Hemorrhage
            # "train": ["IDRiD", "ADAM"],
            # "val": ["IDRiD", "ADAM"],
            # "test": ["IDRiD", "ADAM"],
            # Microaneurysm
            # "train": ["IDRiD", "ADAM"],
            # "val": ["IDRiD", "ADAM"],
            # "test": ["IDRiD", "ADAM"],
            # Vessel
            "train": ["FIVES"],
            "val": ["FIVES"],
            "test": ["CHASE", "DRIVE", "FIVES"],
        },
        "root": "your_path/DATASET",
        "image_folder_name": "images",
        "label_file_name": "labels-duplicate_rm",
        "suffix": "base_proc-224x224",
        "target_class": [
            # "CottonWoolPatch",
            # "HardExudate",
            # "Hemorrhage",
            # "Microaneurysm",
            "Vessel",
        ],
        "num_workers": 2,
        "prefetch_factor": 1,
        "train_batch_size": 16,
    }
    dm = TorchTLSegmentationDM(**config)
    dm.setup("fit")
    print(dm.target_idx)
    dm.setup("test")
    print(dm.target_idx)

    cnt = 0
    for d in dm.val_dataloader():
        cnt += 1
