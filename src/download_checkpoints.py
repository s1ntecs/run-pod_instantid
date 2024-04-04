import torch
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline
from huggingface_hub import hf_hub_download


def fetch_instantid_checkpoints():
    """
    Fetches InstantID checkpoints from the HuggingFace model hub.
    """
    # hf_hub_download(
    #     repo_id="SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe",
    #     filename="depth-zoe-xl-v1.0-controlnet.safetensors",
    #     local_dir="./models/controlnet",
    #     local_dir_use_symlinks=False
    # )
    hf_hub_download(
        repo_id='InstantX/InstantID',
        filename='ControlNetModel/config.json',
        local_dir='./checkpoints',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='InstantX/InstantID',
        filename='ip-adapter.bin',
        local_dir='./checkpoints',
        local_dir_use_symlinks=False
    )
    hf_hub_download(
        repo_id='InstantX/InstantID',
        filename='ControlNetModel/diffusion_pytorch_model.safetensors',
        local_dir='./checkpoints',
        local_dir_use_symlinks=False
    )
     # DL Loras
    hf_hub_download(
        repo_id='artificialguybr/3DRedmond-V1',
        filename='3DRedmond-3DRenderStyle-3DRenderAF.safetensors',
        local_dir='./models/loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl',
        filename='PS1Redmond-PS1Game-Playstation1Graphics.safetensors',
        local_dir='./models/loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/PixelArtRedmond',
        filename='PixelArtRedmond-Lite64.safetensors',
        local_dir='./models/loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/ClayAnimationRedmond',
        filename='ClayAnimationRedm.safetensors',
        local_dir='./models/loras',
        local_dir_use_symlinks=False
    )

    hf_hub_download(
        repo_id='artificialguybr/ToyRedmond-ToyLoraForSDXL10',
        filename='ToyRedmond-FnkRedmAF.safetensors',
        local_dir='./models/loras',
        local_dir_use_symlinks=False
    )


def fetch_pretrained_model(model_name, **kwargs):
    """
    Fetches a pretrained model from the HuggingFace model hub.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return StableDiffusionXLInstantIDPipeline.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f'Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...')
            else:
                raise


def get_instantid_pipeline():
    """
    Fetches the InstantID pipeline from the HuggingFace model hub.
    """
    torch_dtype = torch.float16
# depth-zoe-xl-v1.0-controlnet.safetensors
    args = {
        'controlnet': ControlNetModel.from_pretrained(
            # './checkpoints/depth-zoe-xl-v1.0-controlnet.safetensors',
            # "SargeZT/controlnet-sd-xl-1.0-depth-16bit-zoe",
            './checkpoints/ControlNetModel',
            torch_dtype=torch_dtype),
        'torch_dtype': torch_dtype,
    }

    pipeline = fetch_pretrained_model('frankjoshua/albedobaseXL_v13', **args)

    return pipeline


if __name__ == '__main__':
    fetch_instantid_checkpoints()
    get_instantid_pipeline()

