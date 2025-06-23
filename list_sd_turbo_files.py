#!/usr/bin/env python3
"""List all files in the stabilityai/sd-turbo repository with their sizes."""

from huggingface_hub import HfApi


def list_repository_files(repo_id="stabilityai/sd-turbo"):
    """List all files in the repository with their paths and sizes."""
    api = HfApi()

    try:
        # Get repository info
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")

        print(f"Repository: {repo_id}")
        print(f"Total size: {repo_info.size_on_disk / (1024**3):.2f} GB")
        print("\n" + "=" * 80 + "\n")

        # Get all files
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")

        # Get detailed file info
        file_details = []
        for file_path in files:
            try:
                # Get file info from siblings
                for sibling in repo_info.siblings:
                    if sibling.rfilename == file_path:
                        file_details.append({
                            'path': file_path,
                            'size': sibling.size,
                            'lfs': hasattr(sibling, 'lfs') and sibling.lfs is not None
                        })
                        break
            except Exception as e:
                print(f"Error getting info for {file_path}: {e}")

        # Sort by size (largest first)
        file_details.sort(key=lambda x: x['size'], reverse=True)

        # Print all files
        print("All files in repository:")
        print("-" * 80)
        total_size = 0
        for file_info in file_details:
            size_mb = file_info['size'] / (1024**2)
            size_gb = file_info['size'] / (1024**3)
            lfs_marker = " (LFS)" if file_info['lfs'] else ""

            if size_gb >= 1:
                print(f"{file_info['path']:<60} {size_gb:>8.2f} GB{lfs_marker}")
            else:
                print(f"{file_info['path']:<60} {size_mb:>8.2f} MB{lfs_marker}")

            total_size += file_info['size']

        print("-" * 80)
        print(f"Total calculated size: {total_size / (1024**3):.2f} GB")

        # Filter and show .safetensors and .bin files
        print("\n" + "=" * 80 + "\n")
        print("Model files (.safetensors and .bin):")
        print("-" * 80)

        model_files = [f for f in file_details if f['path'].endswith(('.safetensors', '.bin'))]
        model_total = 0

        for file_info in model_files:
            size_mb = file_info['size'] / (1024**2)
            size_gb = file_info['size'] / (1024**3)
            lfs_marker = " (LFS)" if file_info['lfs'] else ""

            if size_gb >= 1:
                print(f"{file_info['path']:<60} {size_gb:>8.2f} GB{lfs_marker}")
            else:
                print(f"{file_info['path']:<60} {size_mb:>8.2f} MB{lfs_marker}")

            model_total += file_info['size']

        print("-" * 80)
        print(f"Total model files size: {model_total / (1024**3):.2f} GB")

        # Show file count by type
        print("\n" + "=" * 80 + "\n")
        print("File count by extension:")
        print("-" * 80)

        ext_count = {}
        ext_size = {}
        for file_info in file_details:
            ext = file_info['path'].split('.')[-1] if '.' in file_info['path'] else 'no_extension'
            ext_count[ext] = ext_count.get(ext, 0) + 1
            ext_size[ext] = ext_size.get(ext, 0) + file_info['size']

        for ext in sorted(ext_count.keys()):
            size_gb = ext_size[ext] / (1024**3)
            if size_gb >= 1:
                print(f".{ext:<20} {ext_count[ext]:>5} files   {size_gb:>8.2f} GB")
            else:
                size_mb = ext_size[ext] / (1024**2)
                print(f".{ext:<20} {ext_count[ext]:>5} files   {size_mb:>8.2f} MB")

    except Exception as e:
        print(f"Error accessing repository: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    list_repository_files()
