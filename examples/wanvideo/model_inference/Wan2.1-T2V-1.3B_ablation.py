import torch
import re
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", local_model_path="/workspace/autodl-tmp/models/clora-wan"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", local_model_path="/workspace/autodl-tmp/models/clora-wan"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", local_model_path="/workspace/autodl-tmp/models/clora-wan"),
    ],
)

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# ---------------------------------------------------------
# ALL OPTIMAL ABLATION PAIRS INCLUDED HERE
# ---------------------------------------------------------
ablation_tasks =[
    # === CONTENT ABLATION (Changing identity, keeping shape/motion stable) ===
    {
        "category": "content",
        "base_prompt": "A video of a pickup truck driving.",
        "ablation_prompt": "A video of a sports car driving."
    },
    {
        "category": "content",
        "base_prompt": "A video of an astronaut walking.",
        "ablation_prompt": "A video of a robot walking."
    },

    # === STYLE ABLATION (Changing texture/color/medium, keeping shape/motion stable) ===
    {
        "category": "style",
        "base_prompt": "A video of a wooden chair.",
        "ablation_prompt": "A video of a metallic chair."
    },
    {
        "category": "style",
        "base_prompt": "A video of a blue sedan driving.",
        "ablation_prompt": "A video of a red sedan driving."
    },
    {
        "category": "style",
        "base_prompt": "A 3D CGI render of a cat sleeping.",
        "ablation_prompt": "A watercolor painting of a cat sleeping."
    },

    # === MOTION ABLATION (Changing temporal dynamics, keeping pose/structure stable) ===
    {
        "category": "motion",
        "base_prompt": "A video of a man walking forward toward the camera.",
        "ablation_prompt": "A video of a man walking backward away from the camera."
    },
    {
        "category": "motion",
        "base_prompt": "A video of a vinyl record spinning clockwise.",
        "ablation_prompt": "A video of a vinyl record spinning counter-clockwise."
    }
]

ablation_range = {
    5: range(1, 7),
    10: range(1, 4),
    3: range(1, 11),
}

def format_filename_string(text):
    """Cleans a prompt to be used safely as an underscore-connected string."""
    clean = re.sub(r'[^\w\s]', '', text.lower())
    # Cut down to 50 chars so massive prompts don't break Linux/Windows file path limits
    return "_".join(clean.split())[:50]

# Execute Ablation loops
for ablation_block_size in ablation_range.keys():
    for task in ablation_tasks:
        category = task["category"]
        base_prompt = task["base_prompt"]
        ablation_prompt = task["ablation_prompt"]
        
        prompt_str = format_filename_string(base_prompt)
        
        print(f"\n========================================")
        print(f"Starting ablation for {category.upper()}")
        print(f"Base:     {base_prompt}")
        print(f"Ablation: {ablation_prompt}")
        print(f"========================================\n")

        for ablation_block_id in ablation_range[ablation_block_size]:
            
            print(f"  -> Generating: Size {ablation_block_size} | ID {ablation_block_id}...")
            
            video = pipe(
                prompt=base_prompt,
                negative_prompt=NEGATIVE_PROMPT,
                ablation_prompt=ablation_prompt,
                ablation_block_size=ablation_block_size,
                ablation_block_id=ablation_block_id,
                seed=42,
                tiled=True,
            )
            
            # Formats precisely to: video_wan_out_<motion/style/content>-<block_size>-<ablation_block_id>-<text_prompt_with_underline_connection>.mp4
            out_name = f"video_wan_out_{category}-{ablation_block_size}-{ablation_block_id}-{prompt_str}.mp4"
            save_video(video, out_name, fps=15, quality=5)
            
            print(f"  -> Saved {out_name}")