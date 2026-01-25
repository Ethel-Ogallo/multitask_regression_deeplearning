import os
import shutil
import numpy as np
import pandas as pd
import json


def load_data(root):
    '''Load auxiliary data CSV'''
    aux_path = os.path.join(root, "aux_data.csv")
    return pd.read_csv(aux_path)


def sample_typhoons(df, total_typhoons, seed=42):
    '''Sample typhoons uniformly across years'''
    np.random.seed(seed)
   
    # Group typhoons by year (first appearance)
    df_with_years = df.groupby('id')['year'].first().reset_index()
    df_with_years.columns = ['id', 'year']
   
    # Get unique years in sorted order
    years = sorted(df_with_years['year'].unique())
   
    # Calculate how many typhoons to sample per year (uniform)
    typhoons_per_year = total_typhoons / len(years)
   
    sampled_typhoons = []
    for year in years:
        year_typhoons = df_with_years[df_with_years['year'] == year]['id'].values
        n_to_sample = int(np.round(typhoons_per_year))
        n_to_sample = min(n_to_sample, len(year_typhoons))
       
        if n_to_sample > 0:
            sampled = np.random.choice(year_typhoons, size=n_to_sample, replace=False)
            sampled_typhoons.extend(sampled.tolist())
   
    # If we didn't get enough (due to rounding / small years), sample more randomly
    if len(sampled_typhoons) < total_typhoons:
        remaining = total_typhoons - len(sampled_typhoons)
        all_typhoons = df['id'].unique()
        available = list(set(all_typhoons) - set(sampled_typhoons))
        if available:
            additional = np.random.choice(available, size=min(remaining, len(available)), replace=False)
            sampled_typhoons.extend(additional.tolist())
   
    # If we somehow got too many, trim randomly
    if len(sampled_typhoons) > total_typhoons:
        sampled_typhoons = np.random.choice(sampled_typhoons, size=total_typhoons, replace=False).tolist()
   
    return sampled_typhoons


def sample_images(df, sampled_typhoons):
    '''Collect all images/rows for the selected typhoons'''
    all_sampled_rows = []
    for typhoon_id in sampled_typhoons:
        typhoon_data = df[df['id'] == typhoon_id].copy()
        typhoon_data = typhoon_data.sort_values(['year', 'month', 'day', 'hour']).reset_index(drop=True)
        all_sampled_rows.append(typhoon_data)
   
    return pd.concat(all_sampled_rows, ignore_index=True)


def filter_low_image_typhoons(df_sampled, min_images=50):
    '''Remove typhoons with too few images to reduce noise'''
    counts = df_sampled['id'].value_counts()
    valid_ids = counts[counts >= min_images].index
    filtered = df_sampled[df_sampled['id'].isin(valid_ids)].copy()
    return filtered


def copy_images(df_sampled, root, output_dir):
    '''Copy sampled images to output directory'''
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
    '''Save sampled auxiliary data to CSV'''
    output_csv = os.path.join(output_dir, "aux_data.csv")
    df_sampled.to_csv(output_csv, index=False)


def copy_metadata(root, output_dir, sampled_typhoon_ids):
    '''Copy and filter metadata for sampled typhoons'''
    # Copy metadata folder
    metadata_src = os.path.join(root, "metadata")
    metadata_dst = os.path.join(output_dir, "metadata")
    if os.path.exists(metadata_src):
        if os.path.exists(metadata_dst):
            shutil.rmtree(metadata_dst)
        shutil.copytree(metadata_src, metadata_dst)
    
    # Filter JSON metadata
    metadata_json_src = os.path.join(root, "metadata.json")
    metadata_json_dst = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_json_src):
        with open(metadata_json_src, 'r') as f:
            metadata = json.load(f)
        sampled_ids_str = {str(tid) for tid in sampled_typhoon_ids}
        filtered_metadata = {key: value for key, value in metadata.items() if key in sampled_ids_str}
        with open(metadata_json_dst, 'w') as f:
            json.dump(filtered_metadata, f, indent=2)


if __name__ == "__main__":
    # ────────────────────────────────────────────────
    #          Customize these values
    # ────────────────────────────────────────────────
    root = "/home/ogallo/DL4CV/DigitalTyphoon/WP"
    output_dir = "/home/ogallo/DL4CV/WP_sampled"        
    
    target_typhoons = 180                             # Increased for better diversity/results
    min_images_per_typhoon = 50                       # Keep for quality; set to 0 to disable filter
    random_seed = 42
    # ────────────────────────────────────────────────

    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = load_data(root)
    
    total_typhoons_available = df['id'].nunique()
    total_images_available = len(df)
    print(f"Full dataset: {total_typhoons_available} typhoons | {total_images_available:,} images")
    print(f"Average images per typhoon: {total_images_available / total_typhoons_available:.1f}")
    
    target_typhoons = min(target_typhoons, total_typhoons_available)
    print(f"\nSampling {target_typhoons} typhoons (year-stratified)...")
    
    sampled_typhoons = sample_typhoons(df, total_typhoons=target_typhoons, seed=random_seed)
    print(f"→ Sampled {len(sampled_typhoons)} unique typhoons")
    
    print("Collecting all images for selected typhoons...")
    df_sampled = sample_images(df, sampled_typhoons)
    
    print("Filtering typhoons with too few images...")
    df_sampled = filter_low_image_typhoons(df_sampled, min_images=min_images_per_typhoon)
    
    final_typhoons = df_sampled['id'].nunique()
    final_images = len(df_sampled)
    print(f"After filtering ≥ {min_images_per_typhoon} images:")
    print(f"→ Kept {final_typhoons} typhoons | {final_images:,} images")
    print(f"→ Average: {final_images / final_typhoons:.1f} images/typhoon")
    
    print("\nSaving auxiliary CSV...")
    save_sampled_data(df_sampled, output_dir)
    
    print("Copying images (this may take a while)...")
    copied, not_found = copy_images(df_sampled, root, output_dir)
    print(f"→ Copied {copied:,} images | Not found: {len(not_found)}")
    
    print("Handling metadata...")
    copy_metadata(root, output_dir, sampled_typhoons)
    
    print("\nDone! Subset created successfully.")
    print(f"Output directory: {output_dir}")