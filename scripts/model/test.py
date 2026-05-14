"""
Gemma 3 Diagnostic Script — Identify why outputs are empty
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "google/gemma-3-12b-it"

print("=" * 70)
print("GEMMA 3 DIAGNOSTIC")
print("=" * 70)

# 1. Load processor
print("\n[1] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")
print(f"    Processor type: {type(processor)}")
print(f"    Tokenizer type: {type(processor.tokenizer)}")
print(f"    Image processor type: {type(processor.image_processor)}")
print(f"    Pad token ID: {processor.tokenizer.pad_token_id}")
print(f"    EOS token ID: {processor.tokenizer.eos_token_id}")

# 2. Check chat template
print("\n[2] Checking chat template...")
print(f"    Has chat_template: {hasattr(processor.tokenizer, 'chat_template')}")
if hasattr(processor.tokenizer, 'chat_template'):
    template = processor.tokenizer.chat_template
    print(f"    Template length: {len(template) if template else 'None'}")
    print(f"    Template preview: {template[:200] if template else 'None'}...")

# 3. Create dummy image and test message
print("\n[3] Creating test inputs...")
dummy_image = Image.new("RGB", (224, 224), color="blue")
test_question = "What color is this image? Answer in one word."

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": test_question},
        ],
    }
]

# 4. Test apply_chat_template
print("\n[4] Testing apply_chat_template...")
try:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(f"    ✅ apply_chat_template succeeded")
    print(f"    Keys returned: {list(inputs.keys())}")
    print(f"    input_ids shape: {inputs['input_ids'].shape}")
    print(f"    input_ids dtype: {inputs['input_ids'].dtype}")
    
    if "pixel_values" in inputs:
        print(f"    pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"    pixel_values dtype: {inputs['pixel_values'].dtype}")
    else:
        print(f"    ⚠️ NO pixel_values in inputs!")
    
    if "token_type_ids" in inputs:
        print(f"    token_type_ids shape: {inputs['token_type_ids'].shape}")
        unique_ttids = torch.unique(inputs['token_type_ids'])
        print(f"    token_type_ids unique values: {unique_ttids.tolist()}")
    else:
        print(f"    ⚠️ NO token_type_ids in inputs!")
    
    if "attention_mask" in inputs:
        print(f"    attention_mask shape: {inputs['attention_mask'].shape}")
    
    # Decode input to verify format
    decoded_input = processor.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"\n    Decoded input (first 500 chars):")
    print(f"    {decoded_input[:500]}")
    
except Exception as e:
    print(f"    ❌ apply_chat_template FAILED: {e}")
    import traceback
    traceback.print_exc()

# 5. Load model and test generation
print("\n[5] Loading model (4-bit)...")
try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
    )
    model.eval()
    print(f"    ✅ Model loaded successfully")
    print(f"    Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"    ❌ Model loading FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test generation
print("\n[6] Testing generation...")
try:
    inputs_on_device = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Convert float tensors to bfloat16
    for key in inputs_on_device:
        if inputs_on_device[key].dtype == torch.float32:
            inputs_on_device[key] = inputs_on_device[key].to(dtype=torch.bfloat16)
    
    input_len = inputs_on_device["input_ids"].shape[-1]
    print(f"    Input length: {input_len}")
    
    with torch.inference_mode():
        output = model.generate(
            **inputs_on_device,
            max_new_tokens=50,
            do_sample=False,
        )
    
    print(f"    Output shape: {output.shape}")
    print(f"    Output length: {output.shape[-1]}")
    print(f"    New tokens generated: {output.shape[-1] - input_len}")
    
    # Decode full output
    full_output = processor.decode(output[0], skip_special_tokens=False)
    print(f"\n    Full output (with special tokens):")
    print(f"    {full_output[-500:]}")
    
    # Decode only generated part
    generated_tokens = output[0][input_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=True)
    print(f"\n    Generated response only:")
    print(f"    '{response}'")
    
    if response.strip() == "":
        print(f"\n    ⚠️ EMPTY RESPONSE - This is the bug!")
        print(f"    Generated token IDs: {generated_tokens.tolist()}")
        print(f"    Decoded with special tokens: '{processor.decode(generated_tokens, skip_special_tokens=False)}'")
    else:
        print(f"\n    ✅ Generation working correctly!")

except Exception as e:
    print(f"    ❌ Generation FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)