"""Backbone models package with loader functions for different model architectures."""

from typing import Dict, Union

# Import all model classes
from .base import BackboneBase, TimmViTBackbone
from .falir import FLAIR, KeepFIT
from .mae import RetMAE
from .mi_estimator import MINE, InfoNCE
from .retclip import RETClipBackbone
from .retfound import RETFound
from .unimedclip import UniMedCLIP
from .urfound import UrFound
from .vilref import VilRef


# ============================ Loader Functions ============================
def get_unimedclip() -> UniMedCLIP:
    """UniMedCLIP 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        UniMedCLIP: 초기화된 UniMedCLIP 모델 인스턴스
    """
    return UniMedCLIP()


def get_vilref() -> VilRef:
    """VilRef 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        VilRef: 초기화된 VilRef 모델 인스턴스
    """
    return VilRef()


def get_flair() -> FLAIR:
    """FLAIR 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        FLAIR: 초기화된 FLAIR 모델 인스턴스
    """
    return FLAIR()


def get_urfound() -> UrFound:
    """UrFound 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        UrFound: 초기화된 UrFound 모델 인스턴스
    """
    return UrFound()


def get_dinov2() -> TimmViTBackbone:
    """DINOv2 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        TimmViTBackbone: 초기화된 DINOv2 모델 인스턴스
    """
    return TimmViTBackbone(
        model_name="timm/vit_base_patch14_reg4_dinov2.lvd142m",
        standardize=True,
    )


def get_eva02() -> TimmViTBackbone:
    """EVA-02 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        TimmViTBackbone: 초기화된 EVA-02 모델 인스턴스
    """
    return TimmViTBackbone(
        model_name="timm/eva02_small_patch14_224.mim_in22k",
        standardize=True,
    )


def get_keepfit() -> KeepFIT:
    """KeepFIT 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        KeepFIT: 초기화된 KeepFIT 모델 인스턴스
    """
    return KeepFIT()


def get_retclip() -> RETClipBackbone:
    """RET-CLIP 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        RETClipBackbone: 초기화된 RET-CLIP 모델 인스턴스
    """
    return RETClipBackbone(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=512,
    )


def get_retfound() -> RETFound:
    """RETFound 모델의 인스턴스를 생성하여 반환합니다.

    Returns:
        RETFound: 초기화된 RETFound 모델 인스턴스. 디코더와 마스크 토큰을 제외한
            사전학습된 가중치가 로드됩니다.
    """
    return RETFound()


def get_retmae(**kwargs) -> RetMAE:
    """RetMAE 모델의 인스턴스를 생성하여 반환합니다.

    Args:
        **kwargs: RetMAE에 전달할 추가 인자들

    Returns:
        RetMAE: 초기화된 RetMAE 모델 인스턴스
    """
    return RetMAE(**kwargs)


def get_mine(x_dim: int, y_dim: int, hidden_dim: int = 1024) -> MINE:
    """MINE (Mutual Information Neural Estimation) 모델의 인스턴스를 생성하여 반환합니다.

    Args:
        x_dim: 첫 번째 입력 변수의 차원
        y_dim: 두 번째 입력 변수의 차원
        hidden_dim: 크리틱 네트워크의 은닉 차원

    Returns:
        MINE: 초기화된 MINE 모델 인스턴스
    """
    return MINE(x_dim=x_dim, y_dim=y_dim, hidden_dim=hidden_dim)


def get_infonce(temperature: float = 0.07, x_dim: int = 768, y_dim: int = 512) -> InfoNCE:
    """InfoNCE 모델의 인스턴스를 생성하여 반환합니다.

    Args:
        temperature: 온도 파라미터
        x_dim: 첫 번째 입력 변수의 차원
        y_dim: 두 번째 입력 변수의 차원

    Returns:
        InfoNCE: 초기화된 InfoNCE 모델 인스턴스
    """
    return InfoNCE(temperature=temperature, x_dim=x_dim, y_dim=y_dim)


# ============================ Model Registry ============================
MODEL_REGISTRY: Dict[str, callable] = {
    "flair": get_flair,
    "urfound": get_urfound,
    "dinov2": get_dinov2,
    "eva02": get_eva02,
    "keepfit": get_keepfit,
    "retclip": get_retclip,
    "retfound": get_retfound,
    "retmae": get_retmae,
    "mine": get_mine,
    "infonce": get_infonce,
    "vilref": get_vilref,
    "unimedclip": get_unimedclip,
}


def get_model(model_name: str, **kwargs) -> Union[
    FLAIR,
    UrFound,
    TimmViTBackbone,
    KeepFIT,
    RETClipBackbone,
    RETFound,
    RetMAE,
    MINE,
    InfoNCE,
    VilRef,
    UniMedCLIP,
]:
    """모델 이름으로 모델을 생성하는 범용 함수입니다.

    Args:
        model_name: 모델 이름 (MODEL_REGISTRY에 등록된 이름)
        **kwargs: 모델에 전달할 추가 인자들

    Returns:
        요청된 모델의 인스턴스

    Raises:
        ValueError: 지원하지 않는 모델 이름인 경우
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available_models}")

    return MODEL_REGISTRY[model_name](**kwargs)


def list_available_models() -> list:
    """사용 가능한 모델 목록을 반환합니다.

    Returns:
        list: 사용 가능한 모델 이름들의 리스트
    """
    return list(MODEL_REGISTRY.keys())


# ============================ Exports ============================
__all__ = [
    # Base classes
    "BackboneBase",
    # Model classes
    "RetMAE",
    "TimmViTBackbone",
    "FLAIR",
    "KeepFIT",
    "UrFound",
    "RETClipBackbone",
    "RETFound",
    "MINE",
    "InfoNCE",
    "VilRef",
    "UniMedCLIP",
    # Loader functions
    "get_flair",
    "get_urfound",
    "get_dinov2",
    "get_eva02",
    "get_keepfit",
    "get_retclip",
    "get_retfound",
    "get_retmae",
    "get_mine",
    "get_infonce",
    "get_vilref",
    "get_unimedclip",
    # Utility functions
    "get_model",
    "list_available_models",
    "MODEL_REGISTRY",
]
