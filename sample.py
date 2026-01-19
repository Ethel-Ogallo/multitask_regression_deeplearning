
import os
import shutil
import numpy as np
import pandas as pd
import json

def load_data(root):
    aux_path = os.path.join(root, "aux_data.csv")
    return pd.read_csv(aux_path)

def sample_typhoons(df, total_typhoons):
    typhoon_info = df.groupby('id').agg({
        'year': 'min',
        'grade': 'max'
    }).reset_index().rename(columns={'year': 'first_year', 'grade': 'peak_grade'})
    all_years = sorted(typhoon_info['first_year'].unique())
    num_years = len(all_years)
    step = max(1, num_years // total_typhoons)
    target_year_indices = list(range(0, num_years, step))[:total_typhoons]
    target_years = [all_years[i] for i in target_year_indices]

    sampled_typhoons = []
    for year in target_years:
        candidates = typhoon_info[typhoon_info['first_year'] == year]
        if len(candidates) > 0:
            selected = candidates.iloc[0]
            sampled_typhoons.append(int(selected['id']))
    return sampled_typhoons

def sample_images(df, sampled_typhoons, images_per_typhoon):
    all_sampled_rows = []
    for typhoon_id in sampled_typhoons:
        typhoon_data = df[df['id'] == typhoon_id].copy()
        typhoon_data = typhoon_data.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
        n_images = len(typhoon_data)

        if n_images <= images_per_typhoon:
            sampled_rows = typhoon_data
        else:
            n_sample = images_per_typhoon
            peak_idx = typhoon_data['grade'].idxmax()
            early_idx = 0
            decay_idx = n_images - 1
            sampled_indices = {early_idx, peak_idx, decay_idx}

            remaining = n_sample - len(sampled_indices)
            if remaining > 0:
                available = list(set(range(n_images)) - sampled_indices)
                if len(available) > 0:
                    step_size = max(1, len(available) // remaining)
                    evenly_spaced = available[::step_size][:remaining]
                    sampled_indices.update(evenly_spaced)

                    if len(sampled_indices) < n_sample:
                        still_available = list(set(available) - set(evenly_spaced))
                        if still_available:
                            additional_needed = n_sample - len(sampled_indices)
                            additional = np.random.choice(
                                still_available,
                                size=min(additional_needed, len(still_available)),
                                replace=False
                            )
                            sampled_indices.update(additional)

            sampled_rows = typhoon_data.iloc[list(sampled_indices)]

        all_sampled_rows.append(sampled_rows)

    return pd.concat(all_sampled_rows, ignore_index=True)

def copy_images(df_sampled, root, output_dir):
    images_src_dir = os.path.join(root, "image")
    images_dst_dir = os.path.join(output_dir, "image")
    os.makedirs(images_dst_dir, exist_ok=True)

    copied = 0
    not_found = []
    for idx, row in df_sampled.iterrows():
        img_file = row['image_path']
        found = False
        for root_dir, dirs, files in os.walk(images_src_dir):
            if img_file in files:
                src = os.path.join(root_dir, img_file)
                rel_path = os.path.relpath(root_dir, images_src_dir)
                dst = os.path.join(images_dst_dir, rel_path, img_file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
                found = True
                break
        if not found:
            not_found.append(img_file)

    return copied, not_found

def save_sampled_data(df_sampled, output_dir):
    output_csv = os.path.join(output_dir, "aux_data.csv")
    df_sampled.to_csv(output_csv, index=False)

def copy_metadata(root, output_dir, sampled_typhoon_ids):
    metadata_src = os.path.join(root, "metadata")
    metadata_dst = os.path.join(output_dir, "metadata")
    if os.path.exists(metadata_src):
        if os.path.exists(metadata_dst):
            shutil.rmtree(metadata_dst)
        shutil.copytree(metadata_src, metadata_dst)

    metadata_json_src = os.path.join(root, "metadata.json")
    metadata_json_dst = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_json_src):
        with open(metadata_json_src, 'r') as f:
            metadata = json.load(f)
        sampled_ids_str = {str(tid) for tid in sampled_typhoon_ids}
        filtered_metadata = {key: value for key, value in metadata.items() if key in sampled_ids_str}
        with open(metadata_json_dst, 'w') as f:
            json.dump(filtered_metadata, f, indent=2)
