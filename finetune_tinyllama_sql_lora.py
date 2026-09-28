"""
LoRA fine-tuning of TinyLlama-1.1B for Text-to-SQL.

Pipeline:
    1. Check the GPU
    2. Load the base model in FP16
    3. Run a few probe prompts on the BASE model (before fine-tuning)
    4. Load and format a small slice of the sql-create-context dataset
    5. Attach LoRA adapters to the attention layers
    6. Train with TRL's SFTTrainer
    7. Re-run the same probes on the FINE-TUNED model and compare
    8. Save only the LoRA adapter (a few MB)

Plain LoRA is used here (no QLoRA / no 4-bit quantization).
Tested target: Colab free tier, Tesla T4 (16 GB VRAM).

Usage:
    pip install -r requirements.txt
    python finetune_tinyllama_sql_lora.py
"""

import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # base model (~2.2 GB in FP16)
DATASET_ID = "b-mc2/sql-create-context"           # (question, context, answer) triples
NUM_SAMPLES = 3000                                # small slice so training is quick
MAX_SEQ_LENGTH = 512                              # truncate longer examples
OUTPUT_DIR = "./tinyllama-sql-lora"               # checkpoints + logs
ADAPTER_DIR = "./tinyllama-sql-lora-adapter"      # final adapter
SEED = 42

# Same system prompt is used for training and inference so the model
# sees a consistent format.
SYSTEM_PROMPT = (
    "You are a SQL assistant. Given a table schema and a question, "
    "reply with ONLY the SQL query, nothing else."
)

# Fixed prompts we run before AND after training to compare behaviour.
PROBES = [
    {
        "schema": "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);",
        "question": "List the names of employees in the Engineering department earning more than 100000.",
    },
    {
        "schema": "CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT, order_date DATE);",
        "question": "What is the total order amount per customer in 2024?",
    },
    {
        "schema": "CREATE TABLE movies (title TEXT, year INT, rating FLOAT, genre TEXT);",
        "question": "Show the top 5 highest rated horror movies released after 2015.",
    },
]


# --------------------------------------------------------------------------
# 1. Environment check
# --------------------------------------------------------------------------
def print_environment_info():
    """Print Python / PyTorch versions and GPU details."""
    print("Python     :", sys.version.split()[0])
    print("PyTorch    :", torch.__version__)
    print("CUDA avail :", torch.cuda.is_available())

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU name   :", torch.cuda.get_device_name(0))
        print(f"GPU memory : {props.total_memory / 1024**3:.2f} GB")  # bytes -> GB
    else:
        print("WARNING: no GPU found, training will be extremely slow.")


# --------------------------------------------------------------------------
# 2. Load model + tokenizer
# --------------------------------------------------------------------------
def load_model_and_tokenizer():
    """Load TinyLlama in FP16 along with its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Llama tokenizers have no pad token by default; reuse EOS for padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # right-padding is the safe choice for training

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,   # half precision: faster and lighter than FP32 on a T4
        device_map="auto",     # let accelerate place the model on the GPU
    )

    model.config.use_cache = False       # KV cache conflicts with gradient checkpointing
    model.config.pretraining_tp = 1      # disable tensor-parallel slicing of linear layers

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.1f} M")
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")

    return model, tokenizer


# --------------------------------------------------------------------------
# 3. Prompt building + generation helpers
# --------------------------------------------------------------------------
def build_prompt(tokenizer, schema, question):
    """Build an inference prompt using TinyLlama's chat template."""
    user_message = f"Schema:\n{schema}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    # add_generation_prompt=True appends the assistant tag so the model
    # knows it should start answering
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()  # no gradients needed for inference
def generate(model, tokenizer, prompt, max_new_tokens=120):
    """Generate an answer for a prompt and return only the new text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                       # greedy decoding -> reproducible output
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # The output contains the prompt + the answer, so cut off the prompt part
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_probes(model, tokenizer):
    """Run every probe prompt and return the list of answers."""
    answers = []
    for i, probe in enumerate(PROBES, start=1):
        prompt = build_prompt(tokenizer, probe["schema"], probe["question"])
        answer = generate(model, tokenizer, prompt)
        answers.append(answer)
        print(f"\n--- Probe {i} ---")
        print("Q:", probe["question"])
        print("A:", answer)
    return answers


# --------------------------------------------------------------------------
# 4. Dataset preparation
# --------------------------------------------------------------------------
def prepare_datasets(tokenizer):
    """Load the dataset, format it as chat text and tokenize it."""
    raw = load_dataset(DATASET_ID, split="train")
    print("Full dataset size:", len(raw))
    print("Example row      :", raw[0])

    # Shuffle and keep a small subset so the demo trains in a few minutes
    raw = raw.shuffle(seed=SEED).select(range(NUM_SAMPLES))

    # 95% train / 5% eval
    split = raw.train_test_split(test_size=0.05, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]
    print("Train:", len(train_ds), "| Eval:", len(eval_ds))

    def format_example(row):
        """Turn one dataset row into a full chat-formatted training string."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}"},
            {"role": "assistant", "content": row["answer"]},   # the target SQL
        ]
        # No generation prompt here, since the assistant answer is already included
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    # Replace the original columns with a single "text" column
    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_example, remove_columns=eval_ds.column_names)

    print("\n--- Formatted training example ---\n")
    print(train_ds[0]["text"][:800])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,   # the trainer pads dynamically per batch
        )

    # Pre-tokenize so SFTTrainer doesn't need to handle the text field itself
    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    return train_tok, eval_tok


# --------------------------------------------------------------------------
# 5. LoRA setup
# --------------------------------------------------------------------------
def attach_lora(model):
    """Freeze the base model and add trainable LoRA adapters."""
    # Recompute activations in the backward pass to save VRAM
    model.gradient_checkpointing_enable()
    # Needed because the base weights are frozen, otherwise gradients don't flow
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,                 # rank of the low-rank matrices
        lora_alpha=32,        # scaling factor (alpha / r = 2.0)
        lora_dropout=0.05,    # small dropout on the adapter path to reduce overfitting
        bias="none",          # don't train bias terms
        task_type="CAUSAL_LM",
        # Attention projection layers of the Llama architecture
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # expect well under 1% of all weights
    return model


# --------------------------------------------------------------------------
# 6. Training
# --------------------------------------------------------------------------
def train(model, tokenizer, train_tok, eval_tok):
    """Fine-tune the LoRA adapters with SFTTrainer."""
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,    # effective batch size = 2 * 8 = 16
        gradient_checkpointing=True,
        learning_rate=2e-4,               # LoRA usually tolerates a higher LR than full fine-tuning
        lr_scheduler_type="cosine",
        warmup_steps=10,                  # ramp the LR up slowly for stability
        optim="adamw_torch",
        fp16=True,                        # T4 supports FP16 but not BF16
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to="none",                 # no WandB / TensorBoard
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
    )
    trainer.train()

    if torch.cuda.is_available():
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    return trainer


# --------------------------------------------------------------------------
# 7. Compare before / after
# --------------------------------------------------------------------------
def compare_outputs(model, tokenizer, base_outputs):
    """Print base-model vs fine-tuned answers side by side."""
    model.config.use_cache = True   # turn the KV cache back on for fast inference
    model.eval()

    print("=" * 70)
    print("FINE-TUNED MODEL (after LoRA)")
    print("=" * 70)

    for i, probe in enumerate(PROBES):
        prompt = build_prompt(tokenizer, probe["schema"], probe["question"])
        tuned_answer = generate(model, tokenizer, prompt)

        print(f"\n--- Probe {i + 1} ---")
        print("Q:     ", probe["question"])
        print("BEFORE:", base_outputs[i])
        print("=" * 70)
        print("AFTER :", tuned_answer)
        print("=" * 70)


# --------------------------------------------------------------------------
# 8. Save adapter
# --------------------------------------------------------------------------
def save_adapter(model, tokenizer):
    """Save only the LoRA adapter (not the full base model) plus the tokenizer."""
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Report how big the saved adapter is
    total_bytes = sum(
        os.path.getsize(os.path.join(ADAPTER_DIR, f))
        for f in os.listdir(ADAPTER_DIR)
        if os.path.isfile(os.path.join(ADAPTER_DIR, f))
    )
    print(f"Adapter saved to {ADAPTER_DIR} ({total_bytes / 1024**2:.2f} MB)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print_environment_info()

    model, tokenizer = load_model_and_tokenizer()

    # Baseline answers BEFORE any fine-tuning
    print("=" * 70)
    print("BASE MODEL (before fine-tuning)")
    print("=" * 70)
    base_outputs = run_probes(model, tokenizer)

    train_tok, eval_tok = prepare_datasets(tokenizer)

    model = attach_lora(model)
    train(model, tokenizer, train_tok, eval_tok)

    compare_outputs(model, tokenizer, base_outputs)
    save_adapter(model, tokenizer)


def load_finetuned_model():
    """
    Example: load the saved adapter later for inference.

        model, tokenizer = load_finetuned_model()
    """
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)   # attach the adapter on top
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    return model, tokenizer


if __name__ == "__main__":
    main()
