#!/usr/bin/env python3
"""
Automated Kaggle setup script for MediTranslator v3_en2vi training
"""

import os
import json
import subprocess
from pathlib import Path

# Configuration
KAGGLE_USERNAME = "bunnoob2005"
PROJECT_NAME = "mediatranslator"
NOTEBOOK_TITLE = "MediTranslator v3 En→Vi Training"
DATASET_NAME = "mediatranslator-training-data"

def run_command(cmd, description=""):
    """Run shell command and return output"""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def create_kaggle_dataset():
    """Create/upload training data to Kaggle"""
    print("\n📦 Creating Kaggle dataset...")
    
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} not found!")
        return False
    
    # Check if dataset already exists
    cmd = f"kaggle datasets list --search {DATASET_NAME} 2>/dev/null | grep -q {DATASET_NAME}"
    result = run_command(cmd, "Checking if dataset exists")
    
    if result.returncode == 0:
        print(f"✅ Dataset {DATASET_NAME} already exists")
        return True
    
    # Create dataset metadata
    dataset_meta = {
        "title": "MediTranslator Training Data",
        "id": f"{KAGGLE_USERNAME}/{DATASET_NAME}",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["translation", "english", "vietnamese", "medical"],
        "collaborators": [],
        "data": []
    }
    
    # Create temporary dataset directory
    temp_dir = Path("/tmp/kaggle_dataset")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy data files
    print("Copying training files...")
    for file in ["train.en.txt", "train.vi.txt", "public_test.en.txt", "public_test.vi.txt"]:
        src = data_dir / file
        dst = temp_dir / file
        if src.exists():
            run_command(f"cp {src} {dst}", f"Copying {file}")
        else:
            print(f"⚠️  {file} not found")
    
    # Create dataset.json
    with open(temp_dir / "dataset-metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)
    
    # Upload dataset
    print(f"\n📤 Uploading dataset to Kaggle...")
    cmd = f"cd {temp_dir} && kaggle datasets create -p . --dir-mode zip -q"
    run_command(cmd, "Creating Kaggle dataset")
    
    print(f"✅ Dataset created: {KAGGLE_USERNAME}/{DATASET_NAME}")
    return True

def create_kaggle_notebook():
    """Create Kaggle notebook JSON"""
    print("\n📓 Creating Kaggle notebook...")
    
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MediTranslator v3 En→Vi Training\n",
                    "\n",
                    "This notebook trains the v3_en2vi model (278M parameters) for English→Vietnamese medical text translation."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Cell 1: Install Dependencies\n",
                    "!pip install -q torch torchaudio torchvision\n",
                    "!pip install -q wandb pyyaml tqdm"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Cell 2: Clone Repository\n",
                    "import os\n",
                    "if not os.path.exists('MediTranslator'):\n",
                    "    !git clone https://github.com/MothMalone/MediTranslator.git\n",
                    "%cd MediTranslator\n",
                    "print('Repository cloned successfully')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Cell 3: Setup Data Paths\n",
                    "import os\n",
                    "import shutil\n",
                    "\n",
                    "# Setup paths\n",
                    "DATA_INPUT = '/kaggle/input/mediatranslator-training-data'\n",
                    "DATA_OUTPUT = './data/raw'\n",
                    "os.makedirs(DATA_OUTPUT, exist_ok=True)\n",
                    "\n",
                    "# Copy training data\n",
                    "for file in ['train.en.txt', 'train.vi.txt', 'public_test.en.txt', 'public_test.vi.txt']:\n",
                    "    src = f'{DATA_INPUT}/{file}'\n",
                    "    dst = f'{DATA_OUTPUT}/{file}'\n",
                    "    if os.path.exists(src):\n",
                    "        shutil.copy(src, dst)\n",
                    "        print(f'✓ Copied {file}')\n",
                    "    else:\n",
                    "        print(f'⚠ {file} not found')\n",
                    "\n",
                    "print('Data setup complete!')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Cell 4: Login to W&B (Optional - for tracking)\n",
                    "# import wandb\n",
                    "# wandb.login()\n",
                    "print('Skip W&B login or uncomment to login')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "# Cell 5: Train Model with v3_en2vi Config\n",
                    "!python scripts/train.py --config experiments/v3_en2vi/config.yaml"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Training Complete!\n",
                    "\n",
                    "Checkpoints saved to: `experiments/v3_en2vi/checkpoints/`\n",
                    "\n",
                    "Logs available at: `experiments/v3_en2vi/logs/`"
                ]
            }
        ]
    }
    
    return notebook

def main():
    """Main execution"""
    print("\n🚀 MediTranslator - Kaggle Automation Setup")
    print("=" * 60)
    
    # Verify Kaggle CLI
    result = run_command("kaggle --version", "Verifying Kaggle CLI")
    if result.returncode != 0:
        print("❌ Kaggle CLI not installed. Please install it first:")
        print("   pip install kaggle")
        return False
    
    # Create dataset
    if not create_kaggle_dataset():
        print("⚠️  Dataset creation failed or skipped")
    
    # Create notebook JSON
    notebook = create_kaggle_notebook()
    notebook_path = Path("kaggle_notebook.ipynb")
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"✅ Notebook created: {notebook_path}")
    
    # Instructions
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print(f"\n1. Go to https://www.kaggle.com/code")
    print(f"2. Click '+ New Notebook'")
    print(f"3. Select GPU kernel (P100 recommended)")
    print(f"4. Add dataset as input: '{KAGGLE_USERNAME}/{DATASET_NAME}'")
    print(f"5. Copy notebook from: {notebook_path}")
    print(f"6. OR upload using CLI:")
    print(f"   kaggle kernels push -p . -n '{NOTEBOOK_TITLE}' -q public")
    
    print("\n✅ Setup complete! Ready to train on Kaggle")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
