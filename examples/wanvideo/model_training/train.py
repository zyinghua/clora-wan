import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.utils.lora import (
    build_block_lora_target_regex,
    filter_block_lora_state_dict,
    parse_block_ids_cli,
    resolve_dit_block_indices,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_preset_lora_specs(spec: str):
    """Parse ``--preset_lora_path`` into a list of ``(path, alpha)`` pairs.

    Accepts either a single path (legacy single-preset behavior, alpha=1) or a
    comma-separated list of ``path[:alpha]`` items, e.g.
    ``"appearance.safetensors:1.0,style.safetensors:0.5"``. Each preset is
    fused additively into the base weights before the new trainable LoRA is
    attached, so subsequent items see the cumulative fused base.
    """
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            path, alpha_str = token.rsplit(":", 1)
            try:
                alpha = float(alpha_str.strip())
                path = path.strip()
            except ValueError:
                # Treat the colon as part of the path (e.g. drive letter).
                path, alpha = token, 1.0
        else:
            path, alpha = token, 1.0
        out.append((path, alpha))
    return out


class BlockGroupSplittingModelLogger(ModelLogger):
    """Saves the joint LoRA checkpoint plus one file per block group."""

    def __init__(self, output_path, block_ids, block_stride,
                 remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.block_ids = list(block_ids)
        self.block_stride = block_stride

    def _write_per_group(self, accelerator, state_dict, base_name):
        stem, ext = os.path.splitext(base_name)
        for bid in self.block_ids:
            sub = filter_block_lora_state_dict(state_dict, [bid], stride=self.block_stride)
            if not sub:
                print(f"[B-LoRA] no LoRA tensors found for group {bid} (stride {self.block_stride}); skipping per-group save.")
                continue
            path = os.path.join(self.output_path, f"{stem}_g{bid}{ext}")
            accelerator.save(sub, path, safe_serialization=True)
            print(f"[B-LoRA] saved group {bid} ({len(sub)} tensors) -> {path}")

    def _save_with_split(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            joint_path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, joint_path, safe_serialization=True)
            self._write_per_group(accelerator, state_dict, file_name)

    def on_epoch_end(self, accelerator, model, epoch_id):
        self._save_with_split(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def save_model(self, accelerator, model, file_name):
        self._save_with_split(accelerator, model, file_name)


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        lora_block_ids=None, lora_block_stride=1,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Multi-preset LoRA fusion: load 0..N existing LoRAs into the base
        # weights before any new LoRA adapters are attached. This enables
        # workflows like "freeze appearance, train motion": pass an appearance
        # LoRA via --preset_lora_path and the new motion LoRA will train on top
        # of the fused base. Single-path strings without colons keep their
        # legacy alpha=1 behavior; multi-preset uses path[:alpha] comma-separated.
        if preset_lora_path:
            preset_specs = _parse_preset_lora_specs(preset_lora_path)
            if preset_specs:
                if preset_lora_model is None:
                    raise ValueError("--preset_lora_path requires --preset_lora_model.")
                target = getattr(self.pipe, preset_lora_model, None)
                if target is None:
                    raise ValueError(f"pipe has no `{preset_lora_model}` attribute to fuse the preset LoRA into.")
                for path, alpha in preset_specs:
                    print(f"[preset-LoRA] fusing {path} into pipe.{preset_lora_model} with alpha={alpha}")
                    self.pipe.load_lora(target, path, alpha=alpha)
            # Suppress parent's single-preset reload path (we just did it ourselves).
            preset_lora_path = None

        # B-LoRA-style block scoping: translate (block_ids, stride, per-block targets)
        # into a PEFT-compatible regex and hand it to the standard LoRA path.
        if lora_block_ids and lora_base_model is not None:
            block_ids = parse_block_ids_cli(lora_block_ids)
            base_model = getattr(self.pipe, lora_base_model, None)
            num_layers = len(base_model.blocks) if base_model is not None and hasattr(base_model, "blocks") else None
            if num_layers is None:
                raise ValueError(
                    f"--lora_block_ids is set but `pipe.{lora_base_model}` has no `.blocks` attribute; "
                    "block-scoped LoRA is only supported for DiT-like models."
                )
            indices = resolve_dit_block_indices(num_layers, block_ids, lora_block_stride)
            if not indices:
                raise ValueError(f"lora_block_ids={lora_block_ids} with stride={lora_block_stride} resolved to zero DiT indices (num_layers={num_layers}).")
            per_block_targets = [t.strip() for t in (lora_target_modules or "").split(",") if t.strip()]
            if not per_block_targets:
                raise ValueError("--lora_block_ids requires --lora_target_modules, e.g. 'self_attn.q,self_attn.k,self_attn.v,self_attn.o'.")
            lora_target_modules = build_block_lora_target_regex(indices, per_block_targets)
            print(f"[B-LoRA] block_ids={block_ids}, stride={lora_block_stride} -> DiT indices {indices}; targets per block: {per_block_targets}")
            print(f"[B-LoRA] PEFT target_modules regex: {lora_target_modules}")

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        lora_block_ids=args.lora_block_ids,
        lora_block_stride=args.lora_block_stride,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    if args.lora_block_ids:
        model_logger = BlockGroupSplittingModelLogger(
            args.output_path,
            block_ids=parse_block_ids_cli(args.lora_block_ids),
            block_stride=args.lora_block_stride,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
    else:
        model_logger = ModelLogger(
            args.output_path,
            remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
