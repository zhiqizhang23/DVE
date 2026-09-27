

import os
import numpy as np


class PreprocessedDVCollector:

    def __init__(self, exp_dict, dve_manager=None):
        config = exp_dict.get("preprocessed_dv_collect", False)
        if isinstance(config, dict):
            self.enabled = config.get("enabled", False)
        else:
            self.enabled = bool(config)

        self.dve_manager = dve_manager
        self.T_steps = exp_dict.get("T_steps", 28)

        self.delta_vs_per_concept = {}

    def collect(self, step_idx):
        if not self.enabled or self.dve_manager is None:
            return

        if not hasattr(self.dve_manager, 'last_delta_v_per_concept'):
            return

        per_concept = self.dve_manager.last_delta_v_per_concept
        concepts = self.dve_manager.concepts

        if not per_concept:
            for concept in concepts:
                name = concept['name']
                if name not in self.delta_vs_per_concept:
                    self.delta_vs_per_concept[name] = []
                self.delta_vs_per_concept[name].append((step_idx, None))
            return

        for i, concept in enumerate(concepts):
            name = concept['name']
            if name not in self.delta_vs_per_concept:
                self.delta_vs_per_concept[name] = []

            delta_v_concept = per_concept[i] if i < len(per_concept) else None

            if delta_v_concept is not None:
                dv = delta_v_concept.detach().cpu().float().numpy().flatten()
                self.delta_vs_per_concept[name].append((step_idx, dv))
            else:
                self.delta_vs_per_concept[name].append((step_idx, None))

    def save(self, save_dir, prompt_name):
        if not self.enabled or not self.delta_vs_per_concept:
            return

        os.makedirs(save_dir, exist_ok=True)

        for concept_name, delta_vs in self.delta_vs_per_concept.items():
            dim = None
            for step_idx, dv in delta_vs:
                if dv is not None:
                    dim = len(dv)
                    break
            if dim is None:
                continue

            delta_v_array = np.zeros((self.T_steps, dim), dtype=np.float32)
            has_delta_v = np.zeros(self.T_steps, dtype=bool)

            for step_idx, dv in delta_vs:
                if dv is not None and step_idx < self.T_steps:
                    delta_v_array[step_idx] = dv
                    has_delta_v[step_idx] = True

            filename = f"{prompt_name}_{concept_name}.npz"
            np.savez(
                f"{save_dir}/{filename}",
                delta_vs=delta_v_array,      # [T_steps, dim]
                has_delta_v=has_delta_v,     # [T_steps]
                T_steps=self.T_steps
            )
            print(f"  Saved: {save_dir}/{filename}")

        self.delta_vs_per_concept = {}
