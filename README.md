# **DeepSeek-OCR-experimental** 

A Gradio-powered web interface for performing advanced OCR tasks using the DeepSeek-OCR model. This experimental app leverages Hugging Face Transformers to process images for text extraction, document conversion, figure parsing, and object localization. Optimized for NVIDIA GPUs with support for various resolution sizes.

| **Resource** | **Link** | **Description** |
|---------------|-----------|----------------|
| **Updated Model** | [DeepSeek-OCR-Latest-BF16.I64](https://huggingface.co/prithivMLmods/DeepSeek-OCR-Latest-BF16.I64) | Latest optimized OCR model supporting BF16 & I64 precision types. |
| **Demo Space** | [DeepSeek-OCR-Experimental](https://huggingface.co/spaces/prithivMLmods/DeepSeek-OCR-experimental) | Interactive demo for real-time OCR inference and testing. |

# **About the Model**

![1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/W5ZoaaWAEQ2NtQENuG0p2.png)

# **DeepSeek-OCR-Latest-BF16**

> **DeepSeek-OCR-Latest-BF16.I64** is an optimized and updated version of the original [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR). It is an open-source vision-language OCR model designed to extract text from images and scanned documents—including both digital and handwritten content—and can output results as plain text or Markdown. This model leverages a powerful multimodal backbone (**3B VLM**) to improve reading comprehension and layout understanding for both typed and cursive handwriting. It also excels at preserving document structures such as **headings, tables, and lists** in its outputs.

The **BF16 variant** has been updated and tested with the following environment:

```
transformers: 4.57.1
torch: 2.6.0+cu124 (or) the latest version (i.e., torch 2.9.0)
cuda: 12.4
device: NVIDIA H200 MIG 3g.71gb
```

This version allows flexible configuration of attention implementations—such as `flash_attention` or `sdpa`—for performance optimization or standardization. Users can also **opt out** of specific attention implementations if desired.

## Quick Start with Transformers 🤗

#### Install the required packages

```
gradio
torch
transformers==4.57.1 
einops
addict 
easydict
```

### Run Demo

```py
import gradio as gr
import torch
import requests
from transformers import AutoModel, AutoTokenizer
from typing import Iterable
import os
import tempfile
from PIL import Image, ImageDraw
import re
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model_name = "prithivMLmods/DeepSeek-OCR-Latest-BF16.I64"
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

with gr.Blocks(css=css) as demo:
    gr.Markdown("# **DeepSeek OCR [exp]**", elem_id="main-title")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard"])
            model_size = gr.Dropdown(choices=["Tiny", "Small", "Base", "Large", "Gundam (Recommended)"], value="Large", label="Resolution Size")
            task_type = gr.Dropdown(choices=["Free OCR", "Convert to Markdown", "Parse Figure", "Locate Object by Reference"], value="Convert to Markdown", label="Task Type")
            ref_text_input = gr.Textbox(label="Reference Text (for Locate task)", placeholder="e.g., the teacher, 20-10, a red car...", visible=False)
            submit_btn = gr.Button("Process Image", variant="primary")

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
```

## Model and Resource Links

| Resource Type | Description | Link |
|----------------|--------------|------|
| Original Model Card | Official DeepSeek-OCR release by deepseek-ai | [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) |
| Test Model (StrangerZone HF) | Community test deployment (experimental) | [strangervisionhf/deepseek-ocr-latest-transformers](https://huggingface.co/strangervisionhf/deepseek-ocr-latest-transformers) |
| Standard Model Card | Optimized version supporting Transformers v4.57.1 (BF16 precision) | [DeepSeek-OCR-Latest-BF16.I64](https://huggingface.co/prithivMLmods/DeepSeek-OCR-Latest-BF16.I64) |
| Research Paper | DeepSeek-OCR: Contexts Optical Compression | [arXiv:2510.18234](https://huggingface.co/papers/2510.18234) |
| Demo Space | Interactive demo hosted on Hugging Face Spaces | [DeepSeek-OCR Experimental Demo](https://huggingface.co/spaces/prithivMLmods/DeepSeek-OCR-experimental) | 
