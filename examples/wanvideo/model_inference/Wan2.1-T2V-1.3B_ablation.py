import torch
import re
import os
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

OUTPUT_DIR = "ablation_runs2"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", local_model_path="/users/erluo/scratch/models/clora-wan"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", local_model_path="/users/erluo/scratch/models/clora-wan"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", local_model_path="/users/erluo/scratch/models/clora-wan"),
    ],
)

NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# ---------------------------------------------------------
# ALL OPTIMAL ABLATION PAIRS INCLUDED HERE
# ---------------------------------------------------------
ablation_tasks = [
    # === CONTENT ABLATION (Changing identity, keeping shape/motion stable) ===
    {"category": "content",
        "base_prompt": "A video of a frog jumping.",
        "ablation_prompt": "A video of a toad jumping."
    },
    {
        "category": "content",
        "base_prompt": "A video of a monkey climbing.",
        "ablation_prompt": "A video of a squirrel climbing."
    },
    {
        "category": "content",
        "base_prompt": "A video of a lion roaring.",
        "ablation_prompt": "A video of a bear roaring."
    },
    {
        "category": "content",
        "base_prompt": "A video of a rocket launching.",
        "ablation_prompt": "A video of a spaceship launching."
    },
    {
        "category": "content",
        "base_prompt": "A video of a spider crawling.",
        "ablation_prompt": "A video of a crab crawling."
    },
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a dog running.",
    #     "ablation_prompt": "A video of a cat running."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a car driving.",
    #     "ablation_prompt": "A video of a truck driving."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a plane flying.",
    #     "ablation_prompt": "A video of a helicopter flying."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a guitar playing.",
    #     "ablation_prompt": "A video of a piano playing."
    # },
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a fish swimming.",
    #     "ablation_prompt": "A video of a turtle swimming."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a man reading.",
    #     "ablation_prompt": "A video of a woman reading."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of an astronaut walking.",
    #     "ablation_prompt": "A video of a robot walking."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a train moving.",
    #     "ablation_prompt": "A video of a bus moving."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a penguin walking.",
    #     "ablation_prompt": "A video of a duck walking."
    # }, 
    # {
    #     "category": "content",
    #     "base_prompt": "A video of a sword swinging.",
    #     "ablation_prompt": "A video of a stick swinging."
    # }, 

    # # === STYLE ABLATION (Changing texture/color/medium, keeping shape/motion stable) ===
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a black chair spinning.",
    #     "ablation_prompt": "A video of a white chair spinning."
    # }, 
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a red light flashing.",
    #     "ablation_prompt": "A video of a blue light flashing."
    # }, 
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a white bird flying.",
    #     "ablation_prompt": "A video of a black bird flying."
    # }, 
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a pink flower blooming.",
    #     "ablation_prompt": "A video of a white flower blooming."
    # }, 
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a black cat sleeping.",
    #     "ablation_prompt": "A video of a brown cat sleeping."
    # }, 
    # {
    #     "category": "style",
    #     "base_prompt": "A video of a white cup falling.",
    #     "ablation_prompt": "A video of a black cup falling."
    # }, 

    # # === MOTION ABLATION (Changing temporal dynamics, keeping pose/structure stable) ===
    {
        "category": "motion",
        "base_prompt": "A video of a ball rolling left.",
        "ablation_prompt": "A video of a ball rolling right."
    },
    {
        "category": "motion",
        "base_prompt": "A video of a car driving forward.",
        "ablation_prompt": "A video of a car driving backward."
    },
    {
        "category": "motion",
        "base_prompt": "A video of a monkey walking.",
        "ablation_prompt": "A video of a monkey climbing."
    },
    {
        "category": "motion",
        "base_prompt": "A video of a door swinging open.",
        "ablation_prompt": "A video of a door swinging closed."
    },
    {
        "category": "motion",
        "base_prompt": "A video of a person swimming forward.",
        "ablation_prompt": "A video of a person floating still."
    }
    # {
    #     "category": "motion",
    #     "base_prompt": "A video of a man walking forward toward the camera.",
    #     "ablation_prompt": "A video of a man walking backward away from the camera."
    # }, 
    # {
    #     "category": "motion",
    #     "base_prompt": "A video shot of a bird walking.",
    #     "ablation_prompt": "A video shot of a bird flying."
    # }, 
    # {
    #     "category": "motion",
    #     "base_prompt": "A video of a person jumping up.",
    #     "ablation_prompt": "A video of a person falling down."
    # }, 
    # {
    #     "category": "motion",
    #     "base_prompt": "A video of a train moving left.",
    #     "ablation_prompt": "A video of a train moving right."
    # },
    #         {
    #             "category": "motion",
    #             "base_prompt": "A video of a ball bouncing up.",
    #             "ablation_prompt": "A video of a ball rolling forward."
    #         }, 
    # {
    #     "category": "motion",
    #     "base_prompt": "A video of a dog standing still.",
    #     "ablation_prompt": "A video of a dog running fast."
    # }, 
    # {
    #     "category": "motion",
    #     "base_prompt": "A video of a man jumping up.",
    #     "ablation_prompt": "A video of a man crouching down."
    # } 
]

ablation_range = {
    5: range(1, 7),
    # 10: range(1, 4),
    # 3: range(1, 11),
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
            
            base_str = format_filename_string(base_prompt)
            ablation_str = format_filename_string(ablation_prompt)
            
            out_dir = os.path.join(OUTPUT_DIR, category, f"{ablation_block_size}-{ablation_block_id}")
            os.makedirs(out_dir, exist_ok=True)
            
            out_name = os.path.join(out_dir, f"{base_str}-{ablation_str}.mp4")
            save_video(video, out_name, fps=15, quality=5)
            
            print(f"  -> Saved {out_name}")