"""Load Krikri-8B or Mistral-7B (4-bit NF4) with trained LoRA adapters.

Supports both chat formats:
  - Krikri/Llama 3.1: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...
  - Mistral (baseline): <s>[INST]...[/INST]

Heavy imports (torch/transformers/peft) happen inside functions so the rest
of the package stays importable on machines without a GPU stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def gpu_info() -> dict[str, Any]:
    """GPU name/VRAM with the total_memory/total_mem fallback."""
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    return {"available": True, "name": props.name, "vram_gb": round(total / 1024**3, 1)}


def load_model_and_tokenizer(
    base_model: str = "ilsp/Llama-Krikri-8B-Instruct",
    adapters_path: str | Path | None = None,
    four_bit: bool = True,
):
    """Return (model, tokenizer); optionally with LoRA adapters attached.

    Args:
        base_model:    HF id — default is Krikri-8B; use Mistral for baseline.
        adapters_path: directory with adapter_model.safetensors.
        four_bit:      NF4 quantization — same setting used at training time.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if four_bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config, **model_kwargs
    )

    # Load tokenizer from adapter dir if available (has fine-tuned chat template)
    tokenizer_source = str(adapters_path) if adapters_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if adapters_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapters_path))

    model.eval()
    return model, tokenizer


def detect_chat_format(base_model: str) -> str:
    """Detect chat format from model name."""
    lower = base_model.lower()
    if "krikri" in lower or "llama" in lower:
        return "llama3"
    if "mistral" in lower:
        return "mistral"
    return "llama3"  # default


def format_prompt(
    tokenizer,
    message: str,
    system_prompt: str = "",
    rag_context: str = "",
    chat_format: str = "llama3",
) -> str:
    """Build a prompt in the correct chat template format.

    For Krikri/Llama3: uses tokenizer.apply_chat_template()
    For Mistral: uses <s>[INST]...[/INST] format
    """
    if chat_format == "llama3":
        # Build system content with RAG context if available
        system_content = system_prompt
        if rag_context:
            system_content += (
                f"\n\nΣχετικό context από προηγούμενες συνομιλίες:\n{rag_context}"
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": message},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Mistral format (baseline comparison)
        instruction = message
        if rag_context:
            instruction = f"Context: {rag_context}\n\n{message}"
        if system_prompt:
            instruction = f"{system_prompt}\n\n{instruction}"
        return f"<s>[INST] {instruction} [/INST]"


def generate_reply(
    model, tokenizer, message: str,
    system_prompt: str = "",
    context: str = "",
    chat_format: str = "llama3",
    max_new_tokens: int = 150,
    temperature: float = 0.5,
    top_p: float = 0.85,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
) -> str:
    """Single-turn generation with full parameter control."""
    import torch

    prompt = format_prompt(tokenizer, message, system_prompt, context, chat_format)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.strip()
