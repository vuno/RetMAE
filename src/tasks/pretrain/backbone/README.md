# Backbone Models Package

이 패키지는 다양한 백본 모델들을 모듈화된 구조로 제공합니다.

## 구조

```
backbone/
├── __init__.py          # 로더 함수들과 모델 레지스트리
├── base.py              # 기본 클래스와 공통 유틸리티 (TimmViTBackbone 포함)
├── mae.py               # RetMAE 모델
├── falir.py             # FLAIR 관련 모델들 (FLAIR, KeepFIT)
├── urfound.py           # UrFound 모델
├── mi_estimator.py      # MI 추정기들 (MINE, InfoNCE)
├── retclip.py           # RET-CLIP 모델
├── retfound.py          # RETFound 모델
└── README.md            # 사용 가이드
```

## 사용법

### 1. 개별 모델 로드

```python
from src.tasks.pretrain.backbone import get_flair, get_dinov2, get_retmae

# FLAIR 모델
flair_model = get_flair()

# DINOv2 모델
dinov2_model = get_dinov2()

# RetMAE 모델
retmae_model = get_retmae(model_name="vit_base_patch16_224")
```

### 2. 범용 모델 로드 함수 사용

```python
from src.tasks.pretrain.backbone import get_model, list_available_models

# 사용 가능한 모델 목록 확인
print(list_available_models())

# 모델 이름으로 로드
model = get_model("flair")
model = get_model("dinov2")
model = get_model("retmae", model_name="vit_base_patch16_224")
```

### 3. 직접 클래스 import

```python
from src.tasks.pretrain.backbone import RetMAE, FLAIR, UrFound

# 직접 인스턴스 생성
retmae_model = RetMAE(model_name="vit_base_patch16_224")
flair_model = FLAIR()
urfound_model = UrFound()
```

## 사용 가능한 모델들

- **flair**: FLAIR 모델
- **urfound**: UrFound 모델
- **dinov2**: DINOv2 Vision Transformer
- **eva02**: EVA-02 Vision Transformer
- **keepfit**: KeepFIT 모델
- **retclip**: RET-CLIP 모델
- **retfound**: RETFound 모델
- **retmae**: RetMAE 모델
- **mine**: MINE (Mutual Information Neural Estimation)
- **infonce**: InfoNCE 모델

## 마이그레이션 가이드

기존 코드에서 새로운 구조로 마이그레이션하려면:

### Before (기존)

```python
from src.tasks.pretrain.backbone import FLAIR, UrFound, MAEVisionTransformer

flair = FLAIR()
urfound = UrFound()
mae = MAEVisionTransformer(model_name="vit_base_patch16_224")
```

### After (새로운 구조)

```python
from src.tasks.pretrain.backbone import get_flair, get_urfound, get_retmae

flair = get_flair()
urfound = get_urfound()
retmae = get_retmae(model_name="vit_base_patch16_224")
```

또는 범용 함수 사용:

```python
from src.tasks.pretrain.backbone import get_model

flair = get_model("flair")
urfound = get_model("urfound")
retmae = get_model("retmae", model_name="vit_base_patch16_224")
```

## 파일별 모델 구성

### base.py

- `BackboneBase`: 모든 백본 모델의 기본 클래스
- `TimmViTBackbone`: Timm 기반 Vision Transformer 백본 (범용)

### mae.py

- `RetMAE`: Retina 특화 MAE 모델

### falir.py

- `FLAIR`: FLAIR 모델
- `KeepFIT`: KeepFIT 모델

### urfound.py

- `UrFound`: UrFound 모델

### mi_estimator.py

- `MINE`: Mutual Information Neural Estimation
- `InfoNCE`: InfoNCE 손실 함수

### retclip.py

- `RETClipBackbone`: RET-CLIP 모델

### retfound.py

- `RETFoundBackbone`: RETFound 모델

## 장점

1. **완전한 모듈화**: 각 모델이 개별 파일로 분리되어 관리가 용이
2. **확장성**: 새로운 모델 추가가 매우 쉬움
3. **일관성**: 모든 모델이 동일한 인터페이스를 제공
4. **유연성**: 개별 클래스 또는 로더 함수 모두 사용 가능
5. **중앙화**: 모든 모델이 한 곳에서 관리됨
6. **명확한 구조**: 파일명만으로도 어떤 모델이 포함되어 있는지 알 수 있음
7. **간소화**: 불필요한 모델들을 제거하여 핵심 모델들만 유지
