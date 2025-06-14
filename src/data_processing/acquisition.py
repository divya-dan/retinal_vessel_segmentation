#!/usr/bin/env python
import os
import sys

# Add project root to PYTHONPATH so modules in src/ can be imported
current_file = os.path.abspath(__file__)
# Go up three levels: src/data/acquisition.py -> src/data -> src -> project_root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, project_root)

from maples_dr import configure, export_train_set, export_test_set
from src.config import load_config

"""
Task 2: Data acquisition & organization
- Verify MESSIDOR images present
- Configure MAPLES-DR paths (labels + MESSIDOR images)
- Download labels/cache via maples_dr
- Export fundus images + vessel masks to disk for train/test
"""


def main():
    # Load YAML config
    cfg = load_config()
    data_root = cfg['data']['data_root']

    # Define key paths
    maples_dr_path = os.path.join(data_root, 'maples_dr')
    messidor_path = os.path.join(data_root, 'messidor')
    cache_path = os.path.join(data_root, 'cache')
    train_export_dir = os.path.join(data_root, 'train')
    test_export_dir = os.path.join(data_root, 'test')

    # Verify MESSIDOR data directory exists
    if not os.path.isdir(messidor_path):
        print(f"[acquisition] ERROR: MESSIDOR directory not found at '{messidor_path}'")
        print("Please download the 12 MESSIDOR zip archives (Base11.zip, Base12.zip, ..., Base34.zip) from https://www.adcis.net/en/third-party/messidor/ and place them all in this directory.")
        sys.exit(1)

    # Check for MESSIDOR zip files or image files
    import glob
    zip_files = glob.glob(os.path.join(messidor_path, 'Base*.zip'))
    img_files = glob.glob(os.path.join(messidor_path, '*.ppm')) + glob.glob(os.path.join(messidor_path, '*.jpg')) + glob.glob(os.path.join(messidor_path, '*.jpeg')) + glob.glob(os.path.join(messidor_path, '*.png')) + glob.glob(os.path.join(messidor_path, '*.tif'))
    if not zip_files and not img_files:
        print(f"[acquisition] ERROR: No MESSIDOR zip archives or image files found in '{messidor_path}'")
        print("Please ensure the MESSIDOR zip files are downloaded, or manually extract them into this directory.")
        sys.exit(1)

    # If zip files exist, assume user wants to use archives. Otherwise, images are already present.
    if zip_files:
        print(f"[acquisition] Found {len(zip_files)} MESSIDOR zip files. They will be processed by maples_dr.")
    else:
        print(f"[acquisition] Found {len(img_files)} MESSIDOR image files. Processing directly.")

    # Ensure export directories exist
    for d in (train_export_dir, test_export_dir):
        os.makedirs(d, exist_ok=True)

    # Configure MAPLES-DR dataset
    print(f"[acquisition] Configuring MAPLES-DR (labels -> {maples_dr_path}, messidor -> {messidor_path})")
    configure(
        maples_dr_path=maples_dr_path,
        messidor_path=messidor_path,
        cache=cache_path,
        resize=cfg['data'].get('resize', None),
        image_format='rgb',
        preprocessing=None,
        disable_check=False
    )

    # Export train & test subsets (fundus + vessels)
    fields = ['fundus', 'vessels']
    print(f"[acquisition] Exporting training data ({fields}) to {train_export_dir}")
    export_train_set(
        path=train_export_dir,
        fields=fields,
        fundus_as_jpg=False,
        n_workers=cfg.get('data', {}).get('n_workers', None)
    )

    print(f"[acquisition] Exporting testing data ({fields}) to {test_export_dir}")
    export_test_set(
        path=test_export_dir,
        fields=fields,
        fundus_as_jpg=False,
        n_workers=cfg.get('data', {}).get('n_workers', None)
    )

    print("[acquisition] Data acquisition complete.")


if __name__ == '__main__':
    main()
