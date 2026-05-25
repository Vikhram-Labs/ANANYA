"""Post-training quantization for CPU / edge deployment."""

from __future__ import annotations

from pathlib import Path


def export_gguf(
    merged_model_dir: str,
    output_gguf: str,
    quantization: str = "q4_k_m",
) -> None:
    """Export via llama.cpp convert script — run in Colab shell."""
    import subprocess

    script = Path("scripts/export/convert_to_gguf.sh")
    if not script.exists():
        raise FileNotFoundError(
            "GGUF conversion requires llama.cpp. Clone: "
            "git clone https://github.com/ggerganov/llama.cpp && "
            "python convert_hf_to_gguf.py <model_dir> --outfile <out.gguf>"
        )
    subprocess.run(
        ["bash", str(script), merged_model_dir, output_gguf, quantization],
        check=True,
    )


def load_4bit_for_inference(model_name: str, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path or model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer
