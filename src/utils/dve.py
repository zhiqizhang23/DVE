
import torch


class DVEManager:

    def __init__(self, config, pipe, scheduler, max_sequence_length=512, basemodel="black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16):
        """Initialize the DVE manager from a parsed DVE config and a diffusers pipeline."""
        self.config = config
        self.pipe = pipe
        self.scheduler = scheduler
        self.concepts = []
        self.enabled = config.get('enabled', True)
        self.max_sequence_length = max_sequence_length
        self.basemodel = basemodel
        self.torch_dtype = torch_dtype
        self.erase_guidance_scale = config.get('erase_guidance_scale', 3.5)
        self.projection_enabled = config.get('projection', {}).get('enabled', False)
        self.threshold = config.get('projection', {}).get('threshold', 0.0)
        self.model_type = "SD3" if pipe.__class__.__name__ == "StableDiffusion3Pipeline" else "FLUX"

    def prepare(self):
        """Pre-encode prompt embeddings (and optionally load preprocessed ΔV files) for every concept; called once before sampling."""
        transformer_device = next(self.pipe.transformer.parameters()).device
        text_encoder_device = next(self.pipe.text_encoder.parameters()).device

        with torch.inference_mode():
            for concept_cfg in self.config.get('concepts', []):
                gamma_unified = concept_cfg.get('gamma')
                if gamma_unified is not None:
                    gamma_realtime = gamma_unified
                    gamma_preprocessed = gamma_unified
                else:
                    gamma_realtime = concept_cfg.get('gamma_realtime', 1.0)
                    gamma_preprocessed = concept_cfg.get('gamma_preprocessed', 1.0)
                concept = {
                    'name': concept_cfg['name'],
                    'gamma': gamma_realtime,
                    'gamma_realtime': gamma_realtime,
                    'gamma_preprocessed': gamma_preprocessed,
                }

                # Load preprocessed vectors (if any)
                preprocessed_file = concept_cfg.get('preprocessed_dv_file')
                if preprocessed_file:
                    from src.plug.preprocessed_dv.provide import PreprocessedDVProvider
                    concept['preprocessed_provider'] = PreprocessedDVProvider(
                        preprocessed_file, transformer_device, self.torch_dtype
                    )
                    print(f"  Loaded preprocessed vectors for '{concept['name']}' from {preprocessed_file}")

                if concept_cfg.get('erase_prompt') and concept_cfg.get('anchor_prompt'):
                    if self.model_type == "FLUX":
                        era_embeds, era_pooled, era_text_ids = self.pipe.encode_prompt(
                            prompt=concept_cfg['erase_prompt'],
                            prompt_2=None,
                            device=text_encoder_device
                        )
                        concept['era_embeds'] = era_embeds.to(transformer_device)
                        concept['era_pooled'] = era_pooled.to(transformer_device)
                        concept['era_text_ids'] = era_text_ids.to(transformer_device)
                        torch.cuda.empty_cache()

                        anc_embeds, anc_pooled, anc_text_ids = self.pipe.encode_prompt(
                            prompt=concept_cfg['anchor_prompt'],
                            prompt_2=None,
                            device=text_encoder_device
                        )
                        concept['anc_embeds'] = anc_embeds.to(transformer_device)
                        concept['anc_pooled'] = anc_pooled.to(transformer_device)
                        concept['anc_text_ids'] = anc_text_ids.to(transformer_device)
                        torch.cuda.empty_cache()
                    else:
                        (
                            era_embeds,
                            era_negative_embeds,
                            era_pooled,
                            era_negative_pooled,
                        ) = self.pipe.encode_prompt(
                            prompt=concept_cfg['erase_prompt'],
                            prompt_2=None,
                            prompt_3=None,
                            negative_prompt="",
                            negative_prompt_2=None,
                            negative_prompt_3=None,
                            do_classifier_free_guidance=True,
                            device=text_encoder_device,
                            max_sequence_length=self.max_sequence_length,
                        )
                        concept['era_embeds'] = era_embeds.to(transformer_device)
                        concept['era_negative_embeds'] = era_negative_embeds.to(transformer_device)
                        concept['era_pooled'] = era_pooled.to(transformer_device)
                        concept['era_negative_pooled'] = era_negative_pooled.to(transformer_device)
                        concept['era_text_ids'] = None
                        torch.cuda.empty_cache()

                        (
                            anc_embeds,
                            anc_negative_embeds,
                            anc_pooled,
                            anc_negative_pooled,
                        ) = self.pipe.encode_prompt(
                            prompt=concept_cfg['anchor_prompt'],
                            prompt_2=None,
                            prompt_3=None,
                            negative_prompt="",
                            negative_prompt_2=None,
                            negative_prompt_3=None,
                            do_classifier_free_guidance=True,
                            device=text_encoder_device,
                            max_sequence_length=self.max_sequence_length,
                        )
                        concept['anc_embeds'] = anc_embeds.to(transformer_device)
                        concept['anc_negative_embeds'] = anc_negative_embeds.to(transformer_device)
                        concept['anc_pooled'] = anc_pooled.to(transformer_device)
                        concept['anc_negative_pooled'] = anc_negative_pooled.to(transformer_device)
                        concept['anc_text_ids'] = None
                        torch.cuda.empty_cache()

                    print(f"  Prepared concept '{concept['name']}': "
                          f"erase='{concept_cfg['erase_prompt']}' -> anchor='{concept_cfg['anchor_prompt']}'")
                elif not preprocessed_file:
                    raise ValueError(f"Concept '{concept_cfg['name']}' must support preprocessed_dv_file or erase_prompt+anchor_prompt")

                self.concepts.append(concept)

        self.gamma = self.concepts[0]['gamma'] if self.concepts else 1.0

    def compute_delta_v(self, z_t, t, latent_image_ids, step_idx=None, use_realtime=False):
        """Compute the aggregated differential vector field ΔV at the current latent / timestep across all concepts."""
        if not self.enabled or not self.concepts:
            return None

        delta_v_total = torch.zeros_like(z_t)
        self.last_delta_v_per_concept = []  # Save delta_v_concept for the collector to retrieve

        # Use a separate guidance_scale (delayed initialization, used only when real-time computation is needed)
        guidance = None

        for concept in self.concepts:
            # Check if preprocessed vectors are used (when use_realtime=True, it forces real-time computation)
            if 'preprocessed_provider' in concept and not use_realtime:
                # Preprocessing Vector Pattern: Obtain from Provider
                delta_v_concept = concept['preprocessed_provider'].get_delta_v(step_idx, target_shape=z_t.shape)
                if delta_v_concept is None:
                    self.last_delta_v_per_concept.append(None)
                    continue
            else:
                # Real-time Computing Model
                if guidance is None and self.model_type == "FLUX" and "schnell" not in self.basemodel:
                    guidance = torch.tensor([self.erase_guidance_scale], device=z_t.device)
                    guidance = guidance.expand(z_t.shape[0])

                # v(z_t, t, c_era)
                v_era = self._calc_v(
                    z_t, t,
                    concept['era_embeds'],
                    concept['era_pooled'],
                    concept['era_text_ids'],
                    latent_image_ids,
                    guidance,
                    concept.get('era_negative_embeds'),
                    concept.get('era_negative_pooled'),
                )

                # v(z_t, t, c_anc)
                v_anc = self._calc_v(
                    z_t, t,
                    concept['anc_embeds'],
                    concept['anc_pooled'],
                    concept['anc_text_ids'],
                    latent_image_ids,
                    guidance,
                    concept.get('anc_negative_embeds'),
                    concept.get('anc_negative_pooled'),
                )

                # ΔV = v_era - v_anc
                delta_v_concept = v_era - v_anc

            # Save this concept's delta_v_concept for the collector to retrieve
            self.last_delta_v_per_concept.append(delta_v_concept)

            if self.projection_enabled:
                delta_v_total = delta_v_total + delta_v_concept
            else:
                _gamma = concept['gamma_realtime'] if use_realtime else concept['gamma_preprocessed']
                delta_v_total = delta_v_total + _gamma * delta_v_concept

        return delta_v_total

    def correct_velocity(self, v_user, z_t, t, latent_image_ids, step_idx=None, use_realtime=False):
        """Apply DVE correction to the user velocity: subtract ΔV directly, or apply per-concept projection-based selective correction."""
        delta_v = self.compute_delta_v(z_t, t, latent_image_ids, step_idx=step_idx, use_realtime=use_realtime)

        if delta_v is None:
            return v_user

        if self.projection_enabled:
            v_user_flat = v_user.view(-1)
            total_correction = torch.zeros_like(v_user_flat)

            for i, concept in enumerate(self.concepts):
                delta_v_concept = self.last_delta_v_per_concept[i]
                if delta_v_concept is None:
                    continue

                delta_v_flat = delta_v_concept.view(-1)
                delta_v_norm_vec = delta_v_flat / (torch.norm(delta_v_flat) + 1e-8)
                s = torch.sum(v_user_flat * delta_v_norm_vec)

                if s > self.threshold:
                    gamma = concept['gamma_realtime'] if use_realtime else concept['gamma_preprocessed']
                    correction = gamma * (s - self.threshold) * delta_v_norm_vec
                    total_correction = total_correction + correction

            if torch.norm(total_correction) > 0:
                return (v_user_flat - total_correction).view_as(v_user)
            else:
                return v_user
        else:
            return v_user - delta_v

    def _calc_v(self, latents, t, prompt_embeds, pooled_prompt_embeds,
                text_ids, latent_image_ids, guidance,
                negative_prompt_embeds=None, negative_pooled_prompt_embeds=None):
        """Run a single transformer forward pass to produce the velocity v(z_t, t, c); handles FLUX guidance and SD3 CFG."""
        with torch.no_grad():
            if self.model_type == "FLUX":
                timestep = (t / 1000).expand(latents.shape[0])
                v = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            else:
                do_cfg = self.erase_guidance_scale > 1.0
                if do_cfg:
                    latent_model_input = torch.cat([latents] * 2)
                    encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_projections = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    latent_model_input = latents
                    encoder_hidden_states = prompt_embeds
                    pooled_projections = pooled_prompt_embeds

                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    v = noise_pred_uncond + self.erase_guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    v = noise_pred

        return v
