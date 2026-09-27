import os

import torch
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention
def _sdpa_drop_enable_gqa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _orig_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = _sdpa_drop_enable_gqa

from diffusers import FluxPipeline
from diffusers import StableDiffusion3Pipeline
import argparse
import random
import numpy as np
import yaml
from PIL import Image
from src.plug.preprocessed_dv.collect import PreprocessedDVCollector


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="device number to use")
    parser.add_argument("--config", type=str, help="experiment yaml file")
    parser.add_argument("--result_dir", type=str, help="output directory prefix")

    args = parser.parse_args()

    # set device
    device_number = args.device
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.config
    with open(exp_yaml) as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    model_type = exp_configs[0]["model_type"]

    # ========== Pre-load config files (later edits to files will not affect this run) ==========
    # Pre-load DVE config
    dve_yaml = exp_configs[0].get("dve_yaml")
    dve_config = None
    if dve_yaml and os.path.exists(dve_yaml):
        with open(dve_yaml) as file:
            dve_config = yaml.load(file, Loader=yaml.FullLoader)

    # Pre-load all dataset configs
    dataset_configs_cache = {}
    for exp_dict in exp_configs:
        dataset_yaml = exp_dict["dataset_yaml"]
        if dataset_yaml not in dataset_configs_cache:
            with open(dataset_yaml) as file:
                dataset_configs_cache[dataset_yaml] = yaml.load(file, Loader=yaml.FullLoader)

    basemodel = exp_configs[0].get("basemodel", "black-forest-labs/FLUX.1-dev")
    max_seq_len = exp_configs[0].get("max_sequence_length", 512)
    torch_dtype = getattr(torch, exp_configs[0].get("torch_dtype", "float16"))

    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained(basemodel, torch_dtype=torch_dtype)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained(basemodel, torch_dtype=torch_dtype)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    scheduler = pipe.scheduler
    vae_device = device
    text_encoder_device = device  # default to same as main device

    pipe = pipe.to(device)

    # ========== Initialize DVE concept erasure (using pre-loaded dve_config) ==========
    dve_manager = None
    if dve_config and dve_config.get("enabled"):
        from src.utils.dve import DVEManager
        dve_manager = DVEManager(dve_config, pipe, scheduler, max_sequence_length=max_seq_len, basemodel=basemodel, torch_dtype=torch_dtype)
        dve_manager.prepare()
        print(f"DVE enabled with {len(dve_manager.concepts)} concepts")

    if dve_manager is not None:
        from src.utils.flowedit import calc_v_flux, calculate_shift
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        from tqdm import tqdm
    else:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
        from tqdm import tqdm

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        guidance_scale = exp_dict["guidance_scale"]
        height = exp_dict.get("height", 1024)
        width = exp_dict.get("width", 1024)

        # Decide whether to use a global seed or per-prompt seeds
        global_seed = exp_dict.get("seed")  # returns None if no `seed` field

        # Global-seed mode: set the global RNG state
        if global_seed is not None:
            random.seed(global_seed)
            np.random.seed(global_seed)
            torch.manual_seed(global_seed)
            torch.cuda.manual_seed_all(global_seed)

        dataset_yaml = exp_dict["dataset_yaml"]
        dataset_configs = dataset_configs_cache[dataset_yaml]

        for data_dict in dataset_configs:
            prompts = data_dict["prompts"]
            seeds = data_dict.get("seeds")  # per-prompt seeds (if any)

            for prompt_num, prompt in enumerate(prompts):

                # Resolve the seed used for the current prompt
                if seeds is not None:
                    # per-prompt seed mode
                    seed = seeds[prompt_num]
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                else:
                    # global seed mode
                    seed = global_seed

                generator = torch.Generator(device=device).manual_seed(seed)

                # Skip already generated images (resume support)
                save_dir_check = f"{args.result_dir}/{exp_name}/{model_type}/prompt_{prompt_num}"
                output_file_check = f"{save_dir_check}/output_T_steps_{T_steps}_cfg_{guidance_scale}_seed{seed}.png"
                if os.path.exists(output_file_check):
                    print(f"[SKIP] prompt_{prompt_num} already exists")
                    continue

                with torch.autocast("cuda"), torch.inference_mode():
                    if dve_manager is None:
                        # Use the original pipe() when not enabling erasure
                        image = pipe(
                            prompt=prompt,
                            num_inference_steps=T_steps,
                            guidance_scale=guidance_scale,
                            height=height,
                            width=width,
                            generator=generator,
                            max_sequence_length=max_seq_len,
                        ).images[0]
                    else:
                        n_max = exp_dict.get("n_max", T_steps)
                        n_min = exp_dict.get("n_min", 0)
                        n_pre_raw = exp_dict.get("n_pre", None)
                        has_preprocessed = dve_manager is not None and any(
                            'preprocessed_provider' in c for c in dve_manager.concepts
                        )
                        if n_pre_raw is not None:
                            if not has_preprocessed:
                                raise ValueError("n_pre can only be set when using the preprocessed difference vector")
                            if not (n_min <= n_pre_raw <= n_max):
                                raise ValueError(f"n_pre({n_pre_raw}) must satisfy n_min({n_min}) <= n_pre <= n_max({n_max})")
                            # If n_pre < n_max, it means that some steps require real-time NFE, and prompt embeddings must be present
                            if n_pre_raw < n_max:
                                for c in dve_manager.concepts:
                                    if 'preprocessed_provider' in c and 'era_embeds' not in c:
                                        raise ValueError(
                                            f"when n_pre({n_pre_raw}) < n_max({n_max}), concept '{c['name']}' "
                                            f"must provide both erase_prompt and anchor_prompt to support real-time NFE steps")
                            n_pre = n_pre_raw
                        else:
                            n_pre = n_max if has_preprocessed else None
                        preprocess_collector = PreprocessedDVCollector(exp_dict, dve_manager)

                        if model_type == 'FLUX':
                            num_channels_latents = pipe.transformer.config.in_channels // 4

                            # prepare latents
                            latents_packed, latent_image_ids = pipe.prepare_latents(
                                batch_size=1,
                                num_channels_latents=num_channels_latents,
                                height=height,
                                width=width,
                                dtype=torch_dtype,
                                device=device,
                                generator=generator,
                                latents=None
                            )

                            # prepare timesteps
                            sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
                            image_seq_len = latents_packed.shape[1]
                            mu = calculate_shift(
                                image_seq_len,
                                scheduler.config.base_image_seq_len,
                                scheduler.config.max_image_seq_len,
                                scheduler.config.base_shift,
                                scheduler.config.max_shift,
                            )
                            timesteps, T_steps_actual = retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

                            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=None,
                                device=text_encoder_device
                            )
                            prompt_embeds = prompt_embeds.to(device)
                            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                            text_ids = text_ids.to(device)

                            # guidance
                            if pipe.transformer.config.guidance_embeds:
                                guidance = torch.tensor([guidance_scale], device=device).expand(latents_packed.shape[0])
                            else:
                                guidance = None

                            for i, t in tqdm(enumerate(timesteps)):
                                scheduler._init_step_index(t)
                                t_i = scheduler.sigmas[scheduler.step_index]
                                t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i + 1 < len(timesteps) else torch.zeros_like(t_i)

                                v = calc_v_flux(pipe, latents_packed, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t)

                                # Apply Erase Correction (Correction Range: n_min < T_steps - i <= n_max)
                                # That is, skip the first T_steps - n_max steps, and skip the last n_min steps
                                # n_pre partition: use real-time NFE for (n_pre, n_max], and use preprocessed vectors for (n_min, n_pre]
                                if dve_manager is not None and n_min < T_steps - i <= n_max:
                                    _use_realtime = n_pre is not None and T_steps - i > n_pre
                                    v = dve_manager.correct_velocity(v, latents_packed, t, latent_image_ids, step_idx=i, use_realtime=_use_realtime)

                                preprocess_collector.collect(i)

                                # Euler step in
                                latents_packed = latents_packed.to(torch.float32)
                                latents_packed = latents_packed + (t_im1 - t_i) * v
                                latents_packed = latents_packed.to(v.dtype)

                            latents = pipe._unpack_latents(latents_packed, height, width, pipe.vae_scale_factor)
                            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                            latents_on_vae = latents.to(vae_device)
                            image = pipe.vae.decode(latents_on_vae, return_dict=False)[0]
                            image = pipe.image_processor.postprocess(image)[0]
                        else:
                            # SD3 custom loop with optional DVE correction
                            do_cfg = guidance_scale > 1.0
                            num_channels_latents = pipe.transformer.config.in_channels
                            latents = pipe.prepare_latents(
                                batch_size=1,
                                num_channels_latents=num_channels_latents,
                                height=height,
                                width=width,
                                dtype=torch_dtype,
                                device=device,
                                generator=generator,
                                latents=None,
                            )

                            timesteps, T_steps_actual = retrieve_timesteps(scheduler, T_steps, device)

                            (
                                prompt_embeds,
                                negative_prompt_embeds,
                                pooled_prompt_embeds,
                                negative_pooled_prompt_embeds,
                            ) = pipe.encode_prompt(
                                prompt=prompt,
                                prompt_2=None,
                                prompt_3=None,
                                negative_prompt="",
                                negative_prompt_2=None,
                                negative_prompt_3=None,
                                do_classifier_free_guidance=do_cfg,
                                device=text_encoder_device,
                                max_sequence_length=max_seq_len,
                            )

                            prompt_embeds = prompt_embeds.to(device)
                            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                            if do_cfg:
                                negative_prompt_embeds = negative_prompt_embeds.to(device)
                                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)

                            for i, t in tqdm(enumerate(timesteps)):
                                t_i = t / 1000
                                t_im1 = (timesteps[i + 1] / 1000) if i + 1 < len(timesteps) else torch.zeros_like(t_i)

                                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                                timestep = t.expand(latent_model_input.shape[0])
                                if do_cfg:
                                    enc_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                                    pooled_states = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                                else:
                                    enc_states = prompt_embeds
                                    pooled_states = pooled_prompt_embeds

                                noise_pred = pipe.transformer(
                                    hidden_states=latent_model_input,
                                    timestep=timestep,
                                    encoder_hidden_states=enc_states,
                                    pooled_projections=pooled_states,
                                    joint_attention_kwargs=None,
                                    return_dict=False,
                                )[0]

                                if do_cfg:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    v = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                else:
                                    v = noise_pred

                                if dve_manager is not None and n_min < T_steps - i <= n_max:
                                    v = dve_manager.correct_velocity(v, latents, t, None, step_idx=i)

                                preprocess_collector.collect(i)

                                latents = scheduler.step(v, t, latents, return_dict=False)[0]

                            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                            latents_on_vae = latents.to(vae_device)
                            image = pipe.vae.decode(latents_on_vae, return_dict=False)[0]
                            image = pipe.image_processor.postprocess(image)[0]

                        preprocess_collector.save(f"{args.result_dir}/{exp_name}/{model_type}/preprocess_data", f"prompt_{prompt_num}")

                # make sure to create the directories before saving
                save_dir = f"{args.result_dir}/{exp_name}/{model_type}/prompt_{prompt_num}"
                os.makedirs(save_dir, exist_ok=True)

                image.save(f"{save_dir}/output_T_steps_{T_steps}_cfg_{guidance_scale}_seed{seed}.png")
                # also save prompt in txt file
                with open(f"{save_dir}/prompts.txt", "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Model type: {model_type}\n")

    print("Done")
