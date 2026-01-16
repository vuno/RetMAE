import json
import os
from collections import defaultdict
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rootutils
import torch
from albumentations.pytorch.transforms import ToTensorV2
from lightning import pytorch as pl
from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = [
    "COCOData",
    "COCODataset",
    "Dataframe",
    "TorchDataset",
    "TorchClassificationDataset",
]


@dataclass
class COCOData:
    """Dataclass for COCO dataset.

    The following information is stored:
        - img_info: image information
        - anno_info: annotation information
        - cat_info: category information
        - dataset_info: dataset information
        - license_info: license information
    """

    img_info: Dict[str, Any]
    anno_info: Dict[str, Any]
    cat_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    license_info: Dict[str, Any]


class COCODataset(Dataset):
    """Class for loading COCO dataset.

    The following methods are implemented:
        - __len__: return the number of images in the dataset
        - __getitem__: return COCOData object for the given index
        - load_coco: load COCO dataset
        - transform_img_root: transform image root path
        - load_dataset_info: load dataset information
        - load_license_info: load license information
        - load_coco_image: load COCO image
        - load_coco_annos: load COCO annotations
        - load_all_categories: load all categories
        - load_all_images: load all images
        - load_all_annos: load all annotations
        - getAnnIds: get annotation ids
        - getCatIds: get category ids
        - getImgIds: get image ids
        - loadAnns: load annotations
        - loadCats: load categories
        - loadImgs: load images
    """

    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
    ):
        self.root = root
        self.image_folder_name = image_folder_name
        self.label_file_name = label_file_name
        if suffix is not None:
            self.image_folder_name = f"{image_folder_name}-{suffix}"
            self.label_file_name = f"{label_file_name}-{suffix}"

        self.coco_data = self.load_coco(Path(root) / (self.label_file_name + ".json"))

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            int: number of images in the dataset
        """
        return len(self.coco_data.getImgIds())

    def __getitem__(self, index: int) -> COCOData:
        """Return COCOData object for the given index.

        Args:
            index (int): index of image

        Returns:
            COCOData: COCO data
        """
        return self.get_coco_data(index)

    def get_coco_data(self, index: int) -> COCOData:
        """Get COCO data.

        Args:
            index (int): index of image

        Returns:
            COCOData: COCO data
        """
        coco_img = self.load_coco_image(index)
        img_info = self.transform_img_root(coco_img)

        return COCOData(
            img_info=img_info,
            anno_info=self.load_coco_annos(coco_img["id"]),
            cat_info=self.load_all_categories(),
            dataset_info=self.load_dataset_info(),
            license_info=self.load_license_info(),
        )

    def load_coco(self, label_path: Path) -> COCO:
        """Load COCO dataset.

        Args:
            label_path (Path): path to COCO label file

        Returns:
            COCO: COCO dataset
        """
        assert label_path.exists(), f"{label_path} does not exist."
        return COCO(str(label_path))

    def transform_img_root(self, coco_img: Dict[str, Any]) -> Dict[str, Any]:
        """Transform image root path.

        Args:
            coco_img (Dict[str, Any]): COCO image information

        Returns:
            Dict[str, Any]: transformed image information
            The following information is stored:
            - id
            - file_name
            - split
            - height
            - width

        Raises:
            KeyError: raise KeyError if the key is not found.
        """

        try:
            out = {
                "id": coco_img["id"],
                "file_name": f"{self.root}/{self.image_folder_name}/{coco_img['file_name']}",
                "split": coco_img["split"],
                "height": coco_img.get("height", np.nan),
                "width": coco_img.get("width", np.nan),
                "contrast": coco_img.get("contrast", np.nan),
                "redness": coco_img.get("redness", np.nan),
            }
            return out

        except KeyError as e:
            print(f"KeyError: {self.root}, {coco_img['file_name']}")
            raise e

    def load_dataset_info(self) -> Dict[str, Any]:
        """Load dataset information.

        Returns:
            Dict[str, Any]: dataset information
        """
        return self.coco_data.dataset["info"]

    def load_license_info(self) -> Dict[str, Any]:
        """Load license information.

        Returns:
            Dict[str, Any]: license information
        """
        return self.coco_data.dataset["licenses"][0]

    def load_coco_image(self, ids: int) -> Dict[int, str]:
        """Load COCO image.

        Args:
            ids (int): image id

        Returns:
            Dict[int, str]: COCO image information
        """
        return self.loadImgs([ids])[0]

    def load_coco_annos(self, img_ids: int) -> Dict[int, str]:
        """Load COCO annotations.

        Args:
            img_ids (int): image id

        Returns:
            Dict[int, str]: COCO annotation information
        """
        ann_ids = self.getAnnIds(imgIds=img_ids)
        return self.loadAnns(ann_ids)

    def load_all_categories(self) -> List[Dict[int, str]]:
        """Load all categories.

        Returns:
            List[Dict[int, str]]: list of categories
        """
        return self.coco_data.dataset["categories"]

    def load_all_images(self) -> List[Dict[int, str]]:
        """Load all images.

        Returns:
            List[Dict[int, str]]: list of images
        """
        return self.coco_data.dataset["images"]

    def load_all_annos(self) -> List[Dict[int, str]]:
        """Load all annotations.

        Returns:
            List[Dict[int, str]]: list of annotations
        """
        return self.coco_data.dataset["annotations"]

    def getAnnIds(
        self,
        imgIds: List[int] = [],
        catIds: List[int] = [],
        areaRng: List[int] = [],
        iscrowd: int = None,
    ) -> List[int]:
        """Get annotation ids.

        Args:
            imgIds (List[int], optional): List of images id. Defaults to [].
            catIds (List[int], optional): List of category ids. Defaults to [].
            areaRng (List[int], optional): List of areaRngs. Defaults to [].
            iscrowd (int, optional): Defaults to None.

        Returns:
            List[int]: list of annotation ids
        """
        return self.coco_data.getAnnIds(
            imgIds=imgIds, catIds=catIds, areaRng=areaRng, iscrowd=iscrowd
        )

    def getCatIds(
        self, catNms: List[str] = [], supNms: List[str] = [], catIds: List[int] = []
    ) -> List[int]:
        """Get category ids.

        Args:
            catNms (List[str], optional): List of category names. Defaults to [].
            supNms (List[str], optional): List of super-category names. Defaults to [].
            catIds (List[int], optional): List of category ids. Defaults to [].

        Returns:
            List[int]: list of category ids
        """
        return self.coco_data.getCatIds(catNms=catNms, supNms=supNms, catIds=catIds)

    def getImgIds(self, imgIds: List[int] = [], catIds: List[int] = []) -> List[int]:
        """Get image ids.

        Args:
            imgIds (List[int], optional): List of image ids. Defaults to [].
            catIds (List[int], optional): List of category ids. Defaults to [].

        Returns:
            List[int]: list of image ids
        """
        return self.coco_data.getImgIds(imgIds=imgIds, catIds=catIds)

    def loadAnns(self, ids: List[int] = []) -> List[Dict[int, str]]:
        """Load annotations.

        Args:
            ids (List[int], optional): List of annotation ids. Defaults to [].

        Returns:
            List[Dict[int, str]]: list of annotations
        """
        return self.coco_data.loadAnns(ids)

    def loadCats(self, ids: List[int] = []) -> List[Dict[int, str]]:
        """Load categories.

        Args:
            ids (List[int], optional): List of category ids. Defaults to [].

        Returns:
            List[Dict[int, str]]: list of categories
        """
        return self.coco_data.loadCats(ids)

    def loadImgs(self, ids: List[int] = []) -> List[Dict[int, str]]:
        """Load images.

        Args:
            ids (List[int], optional): List of image ids. Defaults to [].

        Returns:
            List[Dict[int, str]]: list of images
        """
        return self.coco_data.loadImgs(ids)


class Dataframe(COCODataset):
    """Class for creating dataframe from COCO dataset.

    The following methods are implemented:
        - __len__: return the number of images in the dataset
        - __getitem__: return pd.Series for the given index
        - sampling: sample data
        - train_val_split: split data into train and validation set
        - load_labels: load label file
        - get_is_mask: get mask information
        - add_dataset_info: add dataset information
        - add_image_info: add image information
        - add_anno_info: add annotation information
        - find_label_file_name: find label file name
    """

    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
        )

        self.CLASS_NAME_LIST = []
        self.LOC_CLASS_LIST = ["Fovea"]

        self.is_mask = self.get_is_mask()
        self.dataframe = self.load_labels(Path(root) / (self.label_file_name + ".json"))
        # drop if "Ungradable" is 1
        if self.dataframe.get("Ungradable") is not None:
            log.info(
                f"Number of samples before dropping ungradable samples: {len(self.dataframe)}"
            )
            self.dataframe = self.dataframe[self.dataframe["Ungradable"] != 1]
            log.info(f"Number of samples after dropping ungradable samples: {len(self.dataframe)}")

        # get class names
        for c in self.dataframe.columns:
            if c.startswith("cls_"):
                self.CLASS_NAME_LIST.append("_".join(c.split("_")[1:]))

            elif "mask_counts" in c and c.split("_")[-1] in [
                "HardExudate",
                "Microaneurysm",
                "Drusen",
                "Hemorrhage",
                "CottonWoolPatch",
                "Vessel",
            ]:
                self.CLASS_NAME_LIST.append(c.split("_")[-1])
                self.dataframe[f"cls_{c.split('_')[-1]}"] = self.dataframe[c].apply(
                    lambda x: float(isinstance(x, str))
                )

        self.CLASS_NAME_LIST.sort()
        self.coco_data = None

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            int: number of images in the dataset
        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> pd.Series:
        """Return pd.Series for the given index.

        Args:
            idx (int): index of image

        Returns:
            pd.Series: image data
        """
        return self.dataframe.iloc[idx]

    def load_labels(self, label_file_path: str) -> pd.DataFrame:
        """Load label file.

        Args:
            label_file_path (str): label file path

        Returns:
            pd.DataFrame: label data
        """

        with open(label_file_path) as f:
            json_data = json.load(f)

        data = defaultdict(list)
        for i in range(len(json_data["images"])):
            # add image info
            self.add_image_info(data, i)
            # add anno info
            self.add_anno_info(data, i)
            # add dataset info
            self.add_dataset_info(data, i)

        return pd.DataFrame(data)

    def get_is_mask(self) -> Dict[str, bool]:
        """Get mask information.

        Returns:
            Dict[str, bool]: whether the category is mask or not
        """
        is_mask = {}
        for cat in self.load_all_categories():
            cat_name = cat["name"]
            if cat_name.startswith("No_"):
                cat_name = "_".join(cat_name.split("_")[1:])
                is_mask[cat_name] = False

            elif cat_name == "Ungradable":
                is_mask[cat_name] = False

            else:
                is_mask[cat_name] = True

        return is_mask

    def add_dataset_info(self, target: Dict[str, List[Any]], sample_index: int):
        """Add dataset information.

        Args:
            target (Dict[str, List[Any]]): target dictionary
            sample_index (int): index of sample
        """
        # add private or public info
        for k, v in self.get_coco_data(sample_index).dataset_info.items():
            if k == "private":
                target["private"].append(v)
        # add license info
        for k, v in self.get_coco_data(sample_index).license_info.items():
            if k == "name":
                target["license"].append(v)

    def add_image_info(self, target: Dict[str, List[Any]], sample_index: int):
        """Add image information.

        Args:
            target (Dict[str, List[Any]]): target dictionary
            sample_index (int): index of sample
        """
        img_info = self.get_coco_data(sample_index).img_info
        img_info_dict = {}
        img_info_dict["sample_id"] = img_info["id"]
        img_info_dict["filepath"] = str(
            Path(self.root) / self.image_folder_name / img_info["file_name"]
        )
        img_info_dict["split"] = img_info["split"]
        img_info_dict["width"] = img_info["width"]
        img_info_dict["height"] = img_info["height"]
        img_info_dict["contrast"] = img_info["contrast"]
        img_info_dict["redness"] = img_info["redness"]
        for k, v in img_info_dict.items():
            target[k].append(v)

    def add_anno_info(self, target: Dict[str, List[Any]], sample_index: int):
        """Add annotation information.

        Args:
            target (Dict[str, List[Any]]): target dictionary
            sample_index (int): index of sample
        """
        anno_info_list = self.get_coco_data(sample_index).anno_info
        anno_info_dict = defaultdict(lambda: np.nan)
        for anno_info in anno_info_list:
            cat_id = anno_info["category_id"]
            cat_name = self.loadCats(cat_id)[0]["name"]

            if cat_name.startswith("No_"):
                cat_name = "_".join(cat_name.split("_")[1:])
                anno_info_dict[f"cls_{cat_name}"] = 0

            elif cat_name == "Ungradable":
                anno_info_dict[cat_name] = 1

            else:  # MASK / LOC annotation 처리
                if self.is_mask[cat_name]:
                    # if cat_name in self.LOC_CLASS_LIST:
                    #     try:
                    #         anno_info_dict[f"loc_x_{cat_name}"] = anno_info["loc"][0]
                    #         anno_info_dict[f"loc_y_{cat_name}"] = anno_info["loc"][1]
                    #     except KeyError:
                    #         anno_info_dict[f"loc_size_{cat_name}"] = anno_info["loc"]["size"]
                    #         anno_info_dict[f"loc_counts_{cat_name}"] = anno_info["loc"]["counts"]
                    # else:
                    #     anno_info_dict[f"mask_size_{cat_name}"] = anno_info["segmentation"]["size"]
                    #     anno_info_dict[f"mask_counts_{cat_name}"] = anno_info["segmentation"][
                    #         "counts"
                    #     ]
                    pass
                else:
                    anno_info_dict[f"cls_{cat_name}"] = 1

        # for k, v in self.is_mask.items():
        #     if v:
        #         if k in self.LOC_CLASS_LIST:
        #             try:
        #                 target[f"loc_x_{k}"].append(anno_info_dict[f"loc_x_{k}"])
        #                 target[f"loc_y_{k}"].append(anno_info_dict[f"loc_y_{k}"])
        #             except KeyError:
        #                 target[f"loc_size_{k}"].append(anno_info_dict[f"loc_size_{k}"])
        #                 target[f"loc_counts_{k}"].append(anno_info_dict[f"loc_counts_{k}"])
        #         else:
        #             target[f"mask_size_{k}"].append(anno_info_dict[f"mask_size_{k}"])
        #             target[f"mask_counts_{k}"].append(anno_info_dict[f"mask_counts_{k}"])
        #     else:
        #         if k == "Ungradable":
        #             target[k].append(anno_info_dict[k])
        #         else:
        #             target[f"cls_{k}"].append(anno_info_dict[f"cls_{k}"])


class TorchDataset(Dataframe):
    """Class for creating torch dataset from dataframe.

    The following methods are implemented:
        - __getitem__: return sample for the given index
        - __len__: return the number of samples in the dataset
        - train_val_split: split data into train and validation set
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
    ):
        assert split in ["train", "val", "test"]
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
        )
        # split data into train and validation set
        if split in ["train", "val"]:
            self.train_val_split(self.dataframe, val_ratio=val_ratio, random_state=42)

        self.dataframe = self.dataframe[self.dataframe["split"] == split]

        # set transforms
        self.transform = transform
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: number of samples in the dataset
        """
        return len(self.dataframe["sample_id"].values)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sample for the given index.

        Args:
            idx (int): index of sample

        Returns:
            Dict[str, Any]: sample data
        """
        # get data from dataframe
        data = self.dataframe.iloc[idx]

        # load image
        img = cv2.imread(data["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image transform
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = self.to_tensor(image=img)["image"]

        return {
            "img": img,
        }

    def train_val_split(
        self, dataframe: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42
    ):
        """Split data into train and validation set.

        First, quantize redness.
        Then, split samples with lesions.
        Finally, get indices of validation samples and update split

        Args:
            dataframe (pd.DataFrame): dataframe
            val_ratio (float, optional): validation ratio. Defaults to 0.2.
            random_state (int, optional): random state. Defaults to 42.
        """

        train_df = dataframe[dataframe["split"] == "train"]

        # 무작위로 validation 데이터 선택
        val_indices = np.random.RandomState(random_state).choice(
            train_df.index, size=int(len(train_df) * val_ratio), replace=False
        )

        # validation 데이터의 split 값 업데이트
        dataframe.loc[val_indices, "split"] = "val"

    def log_num_samples_per_class(self, split: str):
        """Log number of samples per class."""

        log.info("-=" * 20)
        for c in self.CLASS_NAME_LIST:
            num_samples = self.dataframe[f"cls_{c}"].sum()
            log.info(f"Number of samples for class {c}: {int(num_samples)}")

        log.info(f"Number of samples in {split} set: {len(self.dataframe)}")


class TorchClassificationDataset(TorchDataset):
    """Class for creating classification dataset from TorchDataset.

    The following methods are implemented:
        - __getitem__: return sample for the given index
        - __len__: return the number of samples in the dataset
        - train_val_split: split data into train and validation set
        - collate_fn: collate function for torch DataLoader
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
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            split=split,
            transform=transform,
            val_ratio=val_ratio,
        )
        self.log_num_samples_per_class(split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return sample for the given index.

        Args:
            idx (int): index of sample

        Returns:
            Dict[str, Any]: sample data
        """
        # get data from dataframe
        data = self.dataframe.iloc[idx]

        # load image
        img = cv2.imread(data["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image transform
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = self.to_tensor(image=img)["image"]

        # get labels with sorted order
        label_data = data[[f"cls_{c}" for c in self.CLASS_NAME_LIST]]

        # get labels
        label = torch.tensor(label_data.values.astype(np.float32), dtype=torch.float32)

        return {
            "img": img,
            "label": label,
            "filepath": data["filepath"],
            "info": {
                "class_name_list": self.CLASS_NAME_LIST,
            },
        }

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

        train_df = dataframe[dataframe["split"] == "train"]

        # Get lesion columns
        lesion_cols = [
            col
            for col in dataframe.columns
            if col.startswith("cls") and col != "cls_DiabeticRetinopathy_0"
        ]

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


class DataModuleBase(pl.LightningDataModule):
    """DataModuleBase."""

    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        dataset_name: str = "BH",
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
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transform_train = transforms.get("train", None)
        self.transform_val = transforms.get("val", None)
        self.transform_test = transforms.get("test", None)
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage=None):
        """Setup data for train, val, test.

        if stage is train, val, test, only setup data for that stage
        if stage is test, setup data for test stage only

        Args:
            stage (str, optional): fit, test. Defaults to None.

        Raises:
            NotImplementedError: 각 테스크마다 다른 setup method를 구현해야 합니다.
        """
        raise NotImplementedError("setup method must be implemented")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Train dataloader.

        Returns:
            torch.utils.data.DataLoader: train dataloader
        """
        return DataLoader(
            self.data_train,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=True,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Validation dataloader.

        Returns:
            torch.utils.data.DataLoader: validation dataloader
        """
        return DataLoader(
            self.data_val,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Test dataloader.

        Returns:
            torch.utils.data.DataLoader: test dataloader
        """
        return DataLoader(
            self.data_test,
            collate_fn=self.collate_fn,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
            worker_init_fn=self.worker_init_fn,
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function.

        Torch.utils.data.DataLoader의 collate_fn으로 사용됩니다.

        Args:
            batch (List[Dict[str, torch.Tensor]]): batch data

        Raises:
            NotImplementedError: 각 테스크마다 다른 collate function을 구현해야 합니다.
        """
        raise NotImplementedError

    def worker_init_fn(self, worker_id):
        os.sched_setaffinity(0, range(os.cpu_count()))


class ImageOnlyDataModuleBase(DataModuleBase):
    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        dataset_name: str = "BH",
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_ratio: float = 0.1,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            dataset_name=dataset_name,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_ratio=val_ratio,
            transforms=transforms,
            fill_nan=fill_nan,
        )

    def setup(self, stage=None):
        """Setup data for train, val, test.

        if stage is train, val, test, only setup data for that stage
        if stage is test, setup data for test stage only

        Args:
            stage (str, optional): fit, test. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.data_train = TorchDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="train",
                transform=self.transform_train,
                val_ratio=self.hparams.val_ratio,
            )

            self.data_val = TorchDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="val",
                transform=self.transform_val,
                val_ratio=self.hparams.val_ratio,
            )

        if stage == "test" or stage is None:
            self.data_test = TorchDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="test",
                transform=self.transform_test,
            )


class CLSDataModuleBase(DataModuleBase):
    """Classification DataModuleBase."""

    def __init__(
        self,
        root: str = "your_dataset_root",
        image_folder_name: str = "images",
        label_file_name: str = "labels",
        suffix: Optional[str] = None,
        dataset_name: str = "BH",
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        val_ratio: float = 0.1,
        transforms: Dict[str, Callable[..., Any]] = dict(),
        fill_nan: Number | None = None,
    ):
        super().__init__(
            root=root,
            image_folder_name=image_folder_name,
            label_file_name=label_file_name,
            suffix=suffix,
            dataset_name=dataset_name,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_ratio=val_ratio,
            transforms=transforms,
            fill_nan=fill_nan,
        )

        self.class_name_list = None
        self.target_class = None
        self.target_idx = None

    def setup(self, stage=None):
        """Setup data for train, val, test.

        if stage is train, val, test, only setup data for that stage
        if stage is test, setup data for test stage only

        Args:
            stage (str, optional): fit, test. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.data_train = TorchClassificationDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="train",
                transform=self.transform_train,
                val_ratio=self.hparams.val_ratio,
            )

            self.data_val = TorchClassificationDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="val",
                transform=self.transform_val,
                val_ratio=self.hparams.val_ratio,
            )

            self.class_name_list = self.data_train.CLASS_NAME_LIST
            self.target_idx = {
                c: self.data_train.CLASS_NAME_LIST.index(c)
                for c in self.target_class
                if c in self.class_name_list
            }

        if stage == "test" or stage is None:
            self.data_test = TorchClassificationDataset(
                root=os.path.join(self.hparams.root, self.hparams.dataset_name),
                image_folder_name=self.hparams.image_folder_name,
                label_file_name=self.hparams.label_file_name,
                suffix=self.hparams.suffix,
                split="test",
                transform=self.transform_test,
            )

            self.class_name_list = self.data_test.CLASS_NAME_LIST
            self.target_idx = {
                c: self.data_test.CLASS_NAME_LIST.index(c)
                for c in self.target_class
                if c in self.class_name_list
            }
