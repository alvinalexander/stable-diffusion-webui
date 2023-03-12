import os
import pdb
import uuid
import logging
import requests
import shutil


logger = logging.get_logger(__name__)

def download_from_url(url):
    r = requests.get(url, stream=True)
    file_path = url.split('/')[-1]
    if r.ok:
        logger.info("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        logger.exception("Download failed: status code {}\n{}".format(r.status_code, r.text))
        return None
    return os.path.abspath(file_path)


def download_models():
    """
    Download all models and files necessary for controlnet through Unprompted
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if not current_dir.split("/")[-1] == "stable-diffusion-webui":
        logger.exception("Please run this method from the root of $REPO")
        return

    try:
        # Download SD 1.5, which controlnet is compatible with
        sd15_base_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
        sd15_ckpt = download_from_url(sd15_base_url)
        target_dir = os.path.join(current_dir, "models/Stable-diffusion")
        shutil.move(sd15_ckpt, target_dir)
        logger.info("Downloaded {sd15_ckpt.split('/')[-1]} and moved it to {target_dir}")

        # To be placed in models/Stable-diffusion + rename to .ckpt
        sd15_openpose_url = "https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_openpose.pth"
        sd15_openpose_pth = download_from_url(sd15_openpose_url)
        target_file = os.path.join(current_dir, "models/Stable-diffusion/control_sd15_openpose.ckpt")
        shutil.rename(sd15_openpose_pth, target_file)
        logger.info(f"Downloaded + renamed file: {target_file}")

        # To be placed in models/Stable-diffusion + rename to control_sd15_openpose.yaml
        control_sd15_openpose_yaml_url = "https://github.com/lllyasviel/ControlNet/blob/main/models/cldm_v15.yaml"
        control_sd15_openpose_yaml = download_from_url(control_sd15_openpose_yaml_url)
        target_file = os.path.join(current_dir, "models/Stable-diffusion/control_sd15_openpose.yaml")
        shutil.rename(control_sd15_openpose_yaml, target_file)
        logger.info(f"Downloaded + renamed file: {target_file}")

        # To be placed in extensions/unprompted/lib_unprompted/stable_diffusion/controlnet/annotator/ckpts
        annotators_bodypose_pth = download_from_url(annotators_bodypose_url)

        annotators_bodypose_url = "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/body_pose_model.pth"
        target_dir = os.path.join(current_dir, "extensions/unprompted/lib_unprompted/stable_diffusion/controlnet/annotator/ckpts")
        shutil.move(
                annotators_bodypose_pth,
                target_dir)
        logger.info(f"Annotator file {annotators_bodypose_pth.split('/')[-1]} placed in {target_dir}"

        annotators_handpose_url = "https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/hand_pose_model.pth"
        annotators_handpose_pth = download_from_url(annotators_handpose_url)
        shutil.move(
                annotators_handpose_pth,
                target_dir)
        logger.info(f"Annotator file {annotators_handpose_pth.split('/')[-1]} placed in {target_dir}"
    except Exception as e:
        logger.exception(e)

if __name__ == "__main__":
    download_models()
