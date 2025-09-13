import json
import os
import torch
import gc
import random

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


class Step0Trainer:
    """Step 0 of Multimodal CoT:
    Extract spatial relations and reference landmarks from instructions.
    """

    def __init__(self, model_name="yentinglin/Taiwan-LLM-13B-v2.0-Chat"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def create_training_text(self, example):
        user_input = example["input"]
        assistant_output = example["output"]
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_output}<|eot_id|><|end_of_text|>"
        )

    def setup_model_and_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            use_rslora=True
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    def prepare_dataset(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} samples")

        random.seed(42)
        random.shuffle(raw_data)
        training_texts = [self.create_training_text(ex) for ex in raw_data]

        total = len(training_texts)
        n_train = int(total * 0.7)
        n_val = int(total * 0.15)

        train_dataset = Dataset.from_dict({"text": training_texts[:n_train]})
        val_dataset = Dataset.from_dict({"text": training_texts[n_train:n_train + n_val]})
        test_set = training_texts[n_train + n_val:]

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_set)}")
        return train_dataset, val_dataset, test_set

    def tokenize_function(self, examples):
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self, json_path="step0_train.json", output_dir="./models/llama-step0", epochs=8):
        self.clear_gpu_memory()
        self.setup_model_and_tokenizer()

        train_dataset, val_dataset, test_set = self.prepare_dataset(json_path)

        print("Tokenizing...")
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",

            logging_steps=5,
            eval_steps=20,
            save_steps=40,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            bf16=True,
            gradient_checkpointing=True,
            report_to=[],
            dataloader_drop_last=True
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        print("=== Training Step 0 ===")
        trainer.train()

        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self.model.save_pretrained(os.path.join(final_dir, "peft_model"))

        with open(os.path.join(final_dir, "test_set.json"), "w", encoding="utf-8") as f:
            json.dump(test_set, f, ensure_ascii=False, indent=2)

        print(f"Model saved to: {final_dir}")
        return trainer, final_dir


def main():
    trainer = Step0Trainer()
    trainer.train(
        json_path="step0_train.json",
        output_dir="./models/llama-step0",
        epochs=8
    )


if __name__ == "__main__":
    main()
