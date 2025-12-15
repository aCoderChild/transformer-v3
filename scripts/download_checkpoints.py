"""
Download checkpoints from Kaggle notebook output during training.
Run this on your local machine to automatically sync checkpoints.

Usage:
    python scripts/download_checkpoints.py --notebook-name "your-notebook-name"
    
This script:
1. Monitors your Kaggle notebook output
2. Downloads new checkpoints as they're saved
3. Saves them to experiments/v3_en2vi/checkpoints/
"""
import os
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime


class KaggleCheckpointDownloader:
    def __init__(self, notebook_name, local_checkpoint_dir):
        """
        Initialize downloader.
        
        Args:
            notebook_name: Your Kaggle notebook name (from URL)
            local_checkpoint_dir: Where to save checkpoints locally
        """
        self.notebook_name = notebook_name
        self.checkpoint_dir = Path(local_checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.downloaded_files = set()
        self.log_file = self.checkpoint_dir.parent / "download.log"
    
    def log(self, message):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def list_kaggle_outputs(self):
        """List all files in Kaggle notebook output."""
        try:
            # Get notebook outputs
            result = subprocess.run(
                ["kaggle", "kernels", "output", "-k", self.notebook_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.log(f"Error listing Kaggle outputs: {result.stderr}")
                return []
            
            # Parse output files
            files = []
            for line in result.stdout.strip().split('\n'):
                if line.endswith('.pt'):
                    files.append(line)
            
            return files
        except Exception as e:
            self.log(f"Exception listing files: {e}")
            return []
    
    def download_checkpoint(self, remote_path):
        """Download checkpoint from Kaggle."""
        try:
            filename = os.path.basename(remote_path)
            local_path = self.checkpoint_dir / filename
            
            # Skip if already downloaded
            if filename in self.downloaded_files and local_path.exists():
                return False
            
            self.log(f"Downloading: {filename}")
            
            # Download using kaggle CLI
            result = subprocess.run(
                [
                    "kaggle", "kernels", "output",
                    "-k", self.notebook_name,
                    "-p", str(self.checkpoint_dir)
                ],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0 and local_path.exists():
                size_mb = local_path.stat().st_size / (1024**2)
                self.log(f"✓ Downloaded {filename} ({size_mb:.1f} MB)")
                self.downloaded_files.add(filename)
                return True
            else:
                self.log(f"✗ Failed to download {filename}")
                return False
        
        except subprocess.TimeoutExpired:
            self.log(f"✗ Download timeout for {filename}")
            return False
        except Exception as e:
            self.log(f"✗ Error downloading {filename}: {e}")
            return False
    
    def monitor(self, interval=300):
        """
        Monitor Kaggle output and download new checkpoints.
        
        Args:
            interval: Check interval in seconds (default: 5 minutes)
        """
        self.log("=" * 70)
        self.log("KAGGLE CHECKPOINT DOWNLOADER STARTED")
        self.log("=" * 70)
        self.log(f"Notebook: {self.notebook_name}")
        self.log(f"Save to: {self.checkpoint_dir}")
        self.log(f"Check interval: {interval}s")
        self.log("=" * 70)
        
        try:
            iteration = 0
            while True:
                iteration += 1
                self.log(f"\n[Iteration {iteration}] Checking for new checkpoints...")
                
                # List files on Kaggle
                kaggle_files = self.list_kaggle_outputs()
                
                if kaggle_files:
                    self.log(f"Found {len(kaggle_files)} checkpoint files on Kaggle")
                    
                    # Download new ones
                    new_count = 0
                    for remote_file in kaggle_files:
                        filename = os.path.basename(remote_file)
                        if filename not in self.downloaded_files:
                            if self.download_checkpoint(remote_file):
                                new_count += 1
                    
                    if new_count > 0:
                        self.log(f"Downloaded {new_count} new checkpoint(s)")
                    
                    # Summary
                    local_files = list(self.checkpoint_dir.glob("*.pt"))
                    total_size = sum(f.stat().st_size for f in local_files) / (1024**2)
                    self.log(f"Local checkpoints: {len(local_files)} files ({total_size:.1f} MB total)")
                else:
                    self.log("No checkpoints found yet. Training in progress...")
                
                # Wait for next check
                self.log(f"Next check in {interval}s...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            self.log("\nDownloader stopped by user")
        except Exception as e:
            self.log(f"Fatal error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download checkpoints from Kaggle notebook during training"
    )
    parser.add_argument(
        "--notebook-name",
        type=str,
        required=True,
        help="Kaggle notebook name (e.g., 'username/notebook-title')"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="experiments/v3_en2vi/checkpoints",
        help="Local directory to save checkpoints"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)"
    )
    
    args = parser.parse_args()
    
    # Create and run downloader
    downloader = KaggleCheckpointDownloader(
        args.notebook_name,
        args.checkpoint_dir
    )
    
    downloader.monitor(args.interval)


if __name__ == "__main__":
    main()
