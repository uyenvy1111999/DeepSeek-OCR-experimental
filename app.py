import gradio as gr
import torch
import requests
from transformers import AutoModel, AutoTokenizer
import spaces
from typing import Iterable
import os
import tempfile
from PIL import Image, ImageDraw
import re
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes
from docling_core.types.doc import DoclingDocument, DocTagsDocument

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)


colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

print("Determining device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

print("Loading model and tokenizer...")
model_name = "prithivMLmods/DeepSeek-OCR-Latest-BF16.I64" # - (https://huggingface.co/deepseek-ai/DeepSeek-OCR)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name,
    #_attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
).to(device).eval() # Move to device and set to eval mode

if device.type == 'cuda':
    model = model.to(torch.bfloat16)

print("✅ Model loaded successfully to device and in eval mode.")

def find_result_image(path):
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                print(f"Error opening result image {filename}: {e}")
    return None

@spaces.GPU
def process_ocr_task(image, model_size, task_type, ref_text):
    """
    Processes an image with DeepSeek-OCR. The model is already on the correct device.
    """
    if image is None:
        return "Please upload an image first.", None

    print("✅ Model is already on the designated device.")

    with tempfile.TemporaryDirectory() as output_path:
        # Build the prompt
        if task_type == "Free OCR":
            prompt = "<image>\nFree OCR."
        elif task_type == "Convert to Markdown":
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        elif task_type == "Parse Figure":
            prompt = "<image>\nParse the figure."
        elif task_type == "Locate Object by Reference":
            if not ref_text or ref_text.strip() == "":
                raise gr.Error("For the 'Locate' task, you must provide the reference text to find!")
            prompt = f"<image>\nLocate <|ref|>{ref_text.strip()}<|/ref|> in the image."
        else:
            prompt = "<image>\nFree OCR."

        temp_image_path = os.path.join(output_path, "temp_image.png")
        image.save(temp_image_path)

        size_configs = {
            "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
            "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
            "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        }
        config = size_configs.get(model_size, size_configs["Gundam (Recommended)"])

        print(f"🏃 Running inference with prompt: {prompt}")
        text_result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=temp_image_path,
            output_path=output_path,
            base_size=config["base_size"],
            image_size=config["image_size"],
            crop_mode=config["crop_mode"],
            save_results=True,
            test_compress=True,
            eval_mode=True,
        )

        print(f"====\n📄 Text Result: {text_result}\n====")

        result_image_pil = None
        pattern = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
        matches = list(pattern.finditer(text_result))

        if matches:
            print(f"✅ Found {len(matches)} bounding box(es). Drawing on the original image.")
            image_with_bboxes = image.copy()
            draw = ImageDraw.Draw(image_with_bboxes)
            w, h = image.size

            for match in matches:
                coords_norm = [int(c) for c in match.groups()]
                x1_norm, y1_norm, x2_norm, y2_norm = coords_norm

                x1 = int(x1_norm / 1000 * w)
                y1 = int(y1_norm / 1000 * h)
                x2 = int(x2_norm / 1000 * w)
                y2 = int(y2_norm / 1000 * h)

                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            result_image_pil = image_with_bboxes
        else:
            print("⚠️ No bounding box coordinates found in text result. Falling back to search for a result image file.")
            result_image_pil = find_result_image(output_path)

        return text_result, result_image_pil

# url = "https://huggingface.co/spaces/prithivMLmods/Multimodal-OCR3/resolve/main/examples/3.jpg?download=true"
# example_image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    gr.Markdown("# **DeepSeek OCR [exp]**", elem_id="main-title")

    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])
            model_size = gr.Dropdown(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], value="Large", label="Resolution Size")
            task_type = gr.Dropdown(choices=["Free OCR", "Convert to Markdown", "Parse Figure", "Locate Object by Reference"], value="Convert to Markdown", label="Task Type")
            ref_text_input = gr.Textbox(label="Reference Text (for Locate task)", placeholder="e.g., the teacher, 20-10, a red car...", visible=False)
            submit_btn = gr.Button("Process Image", variant="primary")

            examples = gr.Examples(
                examples=["examples/1.jpg", "examples/2.jpg", "examples/3.jpg"],
                                inputs=image_input, label="Examples"
                            )

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Output (OCR)", lines=8, show_copy_button=True)
            output_image = gr.Image(label="Layout Detection (If Any)", type="pil")
            
            with gr.Accordion("Note", open=False):
                gr.Markdown("Inference using Huggingface transformers on NVIDIA GPUs. This app is running with transformers version 4.57.1 and torch version 2.6.0.")
                
    def toggle_ref_text_visibility(task):
        return gr.Textbox(visible=True) if task == "Locate Object by Reference" else gr.Textbox(visible=False)

    task_type.change(fn=toggle_ref_text_visibility, inputs=task_type, outputs=ref_text_input)
    submit_btn.click(fn=process_ocr_task, inputs=[image_input, model_size, task_type, ref_text_input], outputs=[output_text, output_image])

if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True, mcp_server=True, ssr_mode=False)