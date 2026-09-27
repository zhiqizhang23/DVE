
import torch
import numpy as np


class PreprocessedDVProvider:

    def __init__(self, preprocessed_file, device, dtype=torch.float16):
        data = np.load(preprocessed_file)
        self.shape_type = str(data.get('shape_type', 'per_step'))
        self.T_steps = int(data['T_steps'])

        delta_vs_np = data['delta_vs']
        has_delta_v = data.get('has_delta_v', np.ones(self.T_steps, dtype=bool))

        if self.shape_type == 'per_step':
            self.delta_vs = []
            for t in range(self.T_steps):
                if has_delta_v[t]:
                    dv = torch.from_numpy(delta_vs_np[t]).to(device=device, dtype=dtype)
                    self.delta_vs.append(dv)
                else:
                    self.delta_vs.append(None)
        else:
            self.delta_v_shared = torch.from_numpy(delta_vs_np).to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        print(f"[PreprocessedDVProvider] Loaded {preprocessed_file}, "
              f"shape_type={self.shape_type}, T_steps={self.T_steps}")

    def get_delta_v(self, step_idx, target_shape=None):
        if self.shape_type == 'per_step':
            if step_idx is None or step_idx >= len(self.delta_vs):
                return None
            dv = self.delta_vs[step_idx]
            if dv is None:
                return None
        else:
            dv = self.delta_v_shared

        if target_shape is not None:
            dv = dv.view(target_shape)

        return dv
