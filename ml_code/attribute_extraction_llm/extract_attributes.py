import os
import json
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image
import pandas as pd
import yaml
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config("config.yaml")

DATA_DIR = Path(config["data_dir"])
RESIZED_DIR = Path(config["resized_dir"])
OUTPUT_PATH = Path(config["output_path"])
PROMPT_PATH = Path(config["prompt_path"])
IMG_SIZE = tuple(config["image_size"])

def get_openai_client() -> OpenAI:
    """
    Initialize the OpenAI client using an API key from the environment.

    Returns:
        OpenAI: Authenticated OpenAI client.

    Raises:
        RuntimeError: If the API key is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    return OpenAI(api_key=api_key)

def resize_images(source_dir: Path, dest_dir: Path,
                  size: tuple[int, int]) -> list[Path]:
    """
    Resize all images in a directory to a fixed size, centered on a white canvas.

    Args:
        source_dir (Path): Directory containing original images.
        dest_dir (Path): Output directory for resized images.
        size (tuple[int, int]): Target width and height.

    Returns:
        list[Path]: Paths to resized images.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    resized_paths = []

    for img_name in os.listdir(source_dir):
        try:
            src_path = source_dir / img_name
            dest_path = dest_dir / img_name

            img = Image.open(src_path).convert("RGB")
            img.thumbnail(size, Image.LANCZOS)
            new_img = Image.new("RGB", size, (255, 255, 255))
            offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
            new_img.paste(img, offset)
            new_img.save(dest_path, "JPEG", quality=85)
            resized_paths.append(dest_path)
            logger.info(f"Resized {img_name}")

        except Exception as e:
            logger.error(f"Could not process {img_name}: {e}")

    return resized_paths

def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image file as a base64 string.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64-encoded image.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_attributes(client: OpenAI, image_paths: list[Path]) -> list[dict]:
    """
    Send images to OpenAI GPT-4o with a prompt to extract structured clothing attributes.

    Args:
        client (OpenAI): Initialized OpenAI client.
        image_paths (list[Path]): List of image file paths to process.

    Returns:
        list[dict]: List of dictionaries containing extracted attributes and image IDs.
    """
    attributes = []

    PROMPT_PATH = Path(__file__).parent / "prompt.txt"
    with open(PROMPT_PATH, "r") as f:
        prompt_text = f.read()

    for path in image_paths:
        try:
            base64_img = encode_image_to_base64(path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }]
            )
            content = response.choices[0].message.content.strip("```json\n").strip("```")
            data = json.loads(content)
            data["id"] = path.stem
            attributes.append(data)
            print(f"[Processed] {path.name}: {data}")
        except Exception as e:
            print(f"[Error] {path.name}: {e}")

    return attributes

def save_attributes(attributes: list[dict], output_path: Path):
    """
    Save the extracted attribute list to a JSON file.

    Args:
        attributes (list[dict]): List of attribute dictionaries to save.
        output_path (Path): File path to write the output JSON.
    """
    with open(output_path, "w") as f:
        json.dump(attributes, f, indent=2)
    print(f"[Saved] Attributes to {output_path}")

if __name__ == "__main__":
    client = get_openai_client()
    resized_images = resize_images(DATA_DIR, RESIZED_DIR, IMG_SIZE)
    attributes = extract_attributes(client, resized_images)
    save_attributes(attributes, OUTPUT_PATH)
