import os

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random
import numpy as np
import yaml
from src.utils.flowedit import FlowEditSD3, FlowEditFLUX



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="device number to use")
    parser.add_argument("--aux_device", type=int, default=None, help="aux device for VAE/TextEncoders (FLUX only); defaults to device if not provided")
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

    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")
    model_type = exp_configs[0]["model_type"] # currently only one model type per run

    if model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    elif model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")

    scheduler = pipe.scheduler
    pipe = pipe.to(device)
    vae_device = device
    text_encoder_device = device
    aux_device_num = args.aux_device if args.aux_device is not None else args.device
    aux_device = torch.device(f"cuda:{aux_device_num}" if torch.cuda.is_available() else "cpu")
    if model_type == 'FLUX' and aux_device != device:
        vae_device = aux_device
        text_encoder_device = aux_device
        pipe.vae.to(vae_device)
        torch.cuda.empty_cache()

    # ========== Load DVE Concept Erasure Configuration ==========
    dve_manager = None
    dve_yaml = exp_configs[0].get("dve_yaml")

    if dve_yaml and os.path.exists(dve_yaml):
        with open(dve_yaml) as file:
            dve_config = yaml.load(file, Loader=yaml.FullLoader)

        if dve_config and dve_config.get("enabled"):
            from src.utils.dve import DVEManager
            dve_manager = DVEManager(dve_config, pipe, scheduler)
            dve_manager.prepare()
            print(f"DVE enabled with {len(dve_manager.concepts)} concepts")

    # Migrate text encoder to the designated device (regardless of whether DVE is enabled)
    if torch.cuda.device_count() > 1 and model_type == 'FLUX' and text_encoder_device != device:
        pipe.text_encoder.to(text_encoder_device)
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(text_encoder_device)
        torch.cuda.empty_cache()

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml) as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        for data_dict in dataset_configs:

            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            negative_prompt =  "" # optionally add support for negative prompts (SD3)
            image_src_path = data_dict["input_img"]

            # load image
            image = Image.open(image_src_path)
            # crop image to have both dimensions divisibe by 16 - avoids issues with resizing
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            image_src = pipe.image_processor.preprocess(image)
            # cast image to half precision
            image_src = image_src.to(device).half()
            with torch.autocast("cuda"), torch.inference_mode():
                image_src_on_vae = image_src.to(vae_device)
                x0_src_denorm = pipe.vae.encode(image_src_on_vae).latent_dist.mode()
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            # send to cuda
            x0_src = x0_src.to(device)

            for tar_num, tar_prompt in enumerate(tar_prompts):

                if model_type == 'SD3':
                    x0_tar = FlowEditSD3(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            negative_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,)

                elif model_type == 'FLUX':
                    x0_tar = FlowEditFLUX(pipe,
                                                            scheduler,
                                                            x0_src,
                                                            src_prompt,
                                                            tar_prompt,
                                                            T_steps,
                                                            n_avg,
                                                            src_guidance_scale,
                                                            tar_guidance_scale,
                                                            n_min,
                                                            n_max,
                                                            dve_manager=dve_manager,
                                                            text_encoder_device=text_encoder_device)
                else:
                    raise NotImplementedError(f"Sampler type {model_type} not implemented")


                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    x0_tar_denorm_on_vae = x0_tar_denorm.to(vae_device)
                    image_tar = pipe.vae.decode(x0_tar_denorm_on_vae, return_dict=False)[0]
                image_tar = pipe.image_processor.postprocess(image_tar)

                src_prompt_txt = data_dict["input_img"].split("/")[-1].split(".")[0]

                tar_prompt_txt = str(tar_num)

                # make sure to create the directories before saving
                save_dir = f"{args.result_dir}/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)

                image_tar[0].save(f"{save_dir}/output_T_steps_{T_steps}_n_avg_{n_avg}_cfg_enc_{src_guidance_scale}_cfg_dec{tar_guidance_scale}_n_min_{n_min}_n_max_{n_max}_seed{seed}.png")
                # also save source and target prompt in txt file
                with open(f"{save_dir}/prompts.txt", "w") as f:
                    f.write(f"Source prompt: {src_prompt}\n")
                    f.write(f"Target prompt: {tar_prompt}\n")
                    f.write(f"Seed: {seed}\n")
                    f.write(f"Sampler type: {model_type}\n")





    print("Done")

    # %%
