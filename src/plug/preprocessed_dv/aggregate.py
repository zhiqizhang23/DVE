

import os
import numpy as np
import argparse


def load_all_files(input_dir):
    data_list = []
    filenames = sorted(os.listdir(input_dir))

    for idx, filename in enumerate(filenames):
        if not filename.endswith('.npz'):
            continue

        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath)

        print(f"[{idx}] Loading: {filename}")

        data_list.append({
            'filename': filename,
            'delta_vs': data['delta_vs'],       # [T_steps, dim]
            'has_delta_v': data['has_delta_v'], # [T_steps]
            'T_steps': int(data['T_steps'])
        })

    return data_list


def aggregate_mean_per_step(data_list):
    T_steps = data_list[0]['T_steps']
    dim = data_list[0]['delta_vs'].shape[1]

    for data in data_list:
        if data['T_steps'] != T_steps or data['delta_vs'].shape[1] != dim:
            raise ValueError(f"Shape mismatch: expected ({T_steps}, {dim}), "
                           f"got ({data['T_steps']}, {data['delta_vs'].shape[1]})")

    all_delta_vs = np.stack([d['delta_vs'] for d in data_list], axis=0)
    all_has_delta_v = np.stack([d['has_delta_v'] for d in data_list], axis=0)

    aggregated = np.zeros((T_steps, dim), dtype=np.float32)
    has_valid = np.zeros(T_steps, dtype=bool)

    for t in range(T_steps):
        valid_mask = all_has_delta_v[:, t]
        if valid_mask.any():
            aggregated[t] = all_delta_vs[valid_mask, t, :].mean(axis=0)
            has_valid[t] = True

    return aggregated, has_valid, T_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, default="mean_per_step",
                       choices=["mean_per_step"])
    args = parser.parse_args()

    data_list = load_all_files(args.input_dir)

    if not data_list:
        print("No .npz files found in input_dir")
        return

    print(f"\nAggregating {len(data_list)} files...")

    if args.method == "mean_per_step":
        aggregated, has_valid, T_steps = aggregate_mean_per_step(data_list)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/aggregated.npz"
    np.savez(
        output_file,
        delta_vs=aggregated,
        has_delta_v=has_valid,
        shape_type="per_step",
        T_steps=T_steps
    )
    print(f"Saved: {output_file}, shape: {aggregated.shape}")


if __name__ == "__main__":
    main()
