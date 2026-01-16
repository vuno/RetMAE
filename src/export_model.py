import torch
import timm
from timm.models.hub import push_to_hf_hub

# ----------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------
base_model_name = 'vit_base_patch16_224.augreg_in21k'
my_num_classes = 0  # 0 = "Head(Classifier) 제거" (Encoder Only 모드)
save_path = 'your_path/save/fundus_ssl/logs/ssl-retmae_retclip-42/runs/retmae_retclip-1_hmi-0.01_m-0.3/checkpoints/encoder_epoch_0000099.pth'
hf_username = 'aanna0701'
repo_name = 'retmae.distill_retclip'
repo_id = f"{hf_username}/{repo_name}"

# ----------------------------------------------------------------
# 2. Load Model (Encoder Mode)
# ----------------------------------------------------------------
print(f"1. Creating model... ({base_model_name}, classes={my_num_classes})")

# [개선] global_pool을 명시하지 않으면 기본값은 'token' (CLS token만 반환)입니다.
# 만약 패치 전체를 반환하는 것을 기본으로 하고 싶다면 global_pool='' 로 설정해야 합니다.
# 여기서는 일반적인 Feature Extractor처럼 CLS 토큰 반환을 기본으로 설정합니다.
model = timm.create_model(
    base_model_name, 
    pretrained=False, 
    num_classes=my_num_classes
)

print(f"2. Loading weights... ({save_path})")
state_dict = torch.load(save_path, map_location='cpu', weights_only=True)

# 딕셔너리 래핑 처리
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
elif 'model' in state_dict:
    state_dict = state_dict['model']

# [중요] state_dict에서 'head' 관련 가중치가 있다면 제거 (엄격한 Encoder Only)
# num_classes=0이라 어차피 무시되지만, 경고 메시지를 없애기 위해 키를 삭제합니다.
keys_to_remove = [k for k in state_dict.keys() if k.startswith('head.') or k.startswith('fc.')]
for k in keys_to_remove:
    print(f"Removing head key: {k}")
    del state_dict[k]

model.load_state_dict(state_dict)

# ----------------------------------------------------------------
# 3. Configure Metadata (config.json)
# ----------------------------------------------------------------
# [개선] 모델 설정에 "이 모델은 분류기가 없다"는 정보를 박제합니다.
# 이렇게 하면 나중에 로드할 때 num_classes=0을 안 적어도 알아서 헤드 없이 로드됩니다.
model_config = {
    # 모델 아키텍처 필수 정보
    "architecture": base_model_name,
    "num_classes": 0,             # ★ 핵심: 로드 시 자동으로 헤드 제거
    "global_pool": "token",       # ★ 기본 출력: CLS 토큰 (벡터)
    
    # 모델 스펙 상세
    "image_size": 224,
    "patch_size": 16,
    "embed_dim": 768,             # ViT Base 기준
    
    # 데이터 전처리 (실제 학습값과 일치해야 함!)
    "input_spec": {
        "mean": [0.485, 0.456, 0.406], 
        "std": [0.229, 0.224, 0.225]
    },
    
    # 사용자 정의 메타데이터 (사람이 읽기 위한 용도)
    "finetune_info": {
        "task_type": "encoder_only",
        "algorithm": "RetMAE",
        "description": "Pretrained Vision Transformer Encoder (No Classification Head)"
    }
}

# ----------------------------------------------------------------
# 4. Upload
# ----------------------------------------------------------------
print(f"3. Starting Hugging Face upload! ({repo_id})")

push_to_hf_hub(
    model=model,
    repo_id=repo_id,
    commit_message="Upload Encoder-only model (num_classes=0)",
    private=True,
    model_config=model_config # 위에서 정의한 딕셔너리 주입
)

print("\n🎉 Upload complete!")
print(f"Check link: https://huggingface.co/{repo_id}")