#!/usr/bin/env python3
"""Test HuggingFace API to debug siblings issue."""

from huggingface_hub import HfApi


def test_model_info(model_id):
    api = HfApi()

    print(f"Testing model: {model_id}")

    try:
        model_info = api.model_info(model_id)

        print(f"\nModel info type: {type(model_info)}")
        print(f"Has siblings: {hasattr(model_info, 'siblings')}")

        if hasattr(model_info, 'siblings'):
            print(f"Siblings type: {type(model_info.siblings)}")
            print(f"Number of siblings: {len(model_info.siblings) if model_info.siblings else 0}")

            if model_info.siblings:
                # Check first few siblings
                for i, sibling in enumerate(model_info.siblings[:5]):
                    print(f"\nSibling {i}:")
                    print(f"  Type: {type(sibling)}")
                    print(f"  Has rfilename: {hasattr(sibling, 'rfilename')}")
                    print(f"  Has size: {hasattr(sibling, 'size')}")

                    if hasattr(sibling, 'rfilename'):
                        print(f"  rfilename: {sibling.rfilename}")

                    if hasattr(sibling, 'size'):
                        print(f"  size type: {type(sibling.size)}")
                        print(f"  size value: {sibling.size}")

        # Also try list_repo_files
        print("\n\nTrying list_repo_files:")
        files = api.list_repo_files(model_id)
        print(f"Number of files: {len(files)}")
        print(f"First 5 files: {files[:5]}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model_info("google/gemma-3-4b-it")
