"""Push a kernel with machine_shape / docker_image pinning, which the CLI's
kernel-metadata.json parser does not expose.

Usage: python push_kernel.py <folder-with-kernel-metadata.json>
Extra metadata keys honoured: machine_shape, docker_image.
"""
import json
import os
import sys


def main(folder):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    meta_file = os.path.join(folder, "kernel-metadata.json")
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    code_file = os.path.join(folder, meta["code_file"])
    with open(code_file, encoding="utf-8") as f:
        text = f.read()

    api = KaggleApi()
    api.authenticate()
    owner = api.get_config_value("username")

    req = ApiSaveKernelRequest()
    slug = meta["id"].split("/")[-1]
    req.slug = slug
    req.new_title = meta.get("title", slug)
    req.text = text
    req.language = meta.get("language", "python")
    req.kernel_type = meta.get("kernel_type", "notebook")
    req.is_private = meta.get("is_private", True)
    req.enable_gpu = bool(meta.get("enable_gpu", False))
    req.enable_internet = bool(meta.get("enable_internet", False))
    req.dataset_data_sources = meta.get("dataset_sources", [])
    req.competition_data_sources = meta.get("competition_sources", [])
    req.model_data_sources = meta.get("model_sources", [])
    if meta.get("machine_shape"):
        req.machine_shape = meta["machine_shape"]
    if meta.get("docker_image"):
        req.docker_image = meta["docker_image"]
        req.docker_image_pinning_type = "pinned"

    with api.build_kaggle_client() as kaggle:
        try:
            resp = kaggle.kernels.kernels_api_client.save_kernel(req)
        except Exception as e:
            body = getattr(getattr(e, "response", None), "text", "")
            print("PUSH FAILED:", type(e).__name__, str(e)[:300], body[:500])
            raise
    print("pushed:", getattr(resp, "url", None) or "(no url)", "|",
          getattr(resp, "message", "") or "(no msg)", "| version:",
          getattr(resp, "version_number", None))


if __name__ == "__main__":
    main(sys.argv[1])
