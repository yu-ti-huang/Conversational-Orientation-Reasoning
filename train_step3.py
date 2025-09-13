import os
import json
import torch
import pandas as pd
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Step3Trainer:
    def __init__(self, model_name="./models/llama-step2/final_model"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Loading from Step 2 model: {self.model_name}")

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()

    def create_training_text(self, example):
        user_input = example["input"]
        assistant_output = example["output"]
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_output}<|eot_id|><|end_of_text|>"

    def setup_model_and_tokenizer(self):
        print("Loading tokenizer from Step 2 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Step 2 fine-tuned model with 4-bit quantization...")
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
        print(f"Step 3 Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

    def load_split_datasets(self, train_path, val_path, test_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print(f"Loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        train_data = [{"input": row['multimodal_input_clean'], "output": row['training_target']} for _, row in train_df.iterrows()]
        val_data = [{"input": row['multimodal_input_clean'], "output": row['training_target']} for _, row in val_df.iterrows()]
        test_data = [{"input": row['multimodal_input_clean'], "output": row['training_target']} for _, row in test_df.iterrows()]

        return train_data, val_data, test_data

    def prepare_dataset(self, train_path, val_path, test_path):
        raw_train_data, raw_val_data, raw_test_data = self.load_split_datasets(train_path, val_path, test_path)
        train_texts = [self.create_training_text(ex) for ex in raw_train_data]
        val_texts = [self.create_training_text(ex) for ex in raw_val_data]
        test_texts = [self.create_training_text(ex) for ex in raw_test_data]

        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})

        print(f"Final datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_texts)}")
        return train_dataset, val_dataset, test_texts, raw_test_data

    def tokenize_function(self, examples):
        result = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=768,
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self, train_path="spatial_train.csv", val_path="spatial_val.csv", test_path="spatial_test.csv",
              output_dir="./models/llama-step3", epochs=5):
        print("=== Starting Step 3 Complete Reasoning Training ===")
        self.clear_gpu_memory()
        self.setup_model_and_tokenizer()
        train_dataset, val_dataset, test_set, raw_test_data = self.prepare_dataset(train_path, val_path, test_path)

        print("Tokenizing...")
        train_dataset = train_dataset.map(self.tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True, remove_columns=val_dataset.column_names)

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
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
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

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        print("Starting Step 3 training...")
        trainer.train()

        final_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self.model.save_pretrained(os.path.join(final_dir, "peft_model"))

        test_data_clean = []
        import re
        for training_format, raw_data in zip(test_set, raw_test_data):
            user_match = re.search(r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", training_format, re.DOTALL)
            assistant_match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", training_format, re.DOTALL)
            if user_match and assistant_match:
                test_data_clean.append({
                    "input": user_match.group(1).strip(),
                    "expected_output": assistant_match.group(1).strip(),
                    "original_data": raw_data
                })

        with open(os.path.join(final_dir, "test_set.json"), "w", encoding="utf-8") as f:
            json.dump(test_data_clean, f, ensure_ascii=False, indent=2)

        print(f"Step 3 model saved to: {final_dir}")
        return trainer, final_dir


def main():
    trainer = Step3Trainer()
    trainer.train(
        train_path="spatial_train.csv",
        val_path="spatial_val.csv",
        test_path="spatial_test.csv",
        output_dir="./models/llama-step3",
        epochs=5
    )


if __name__ == "__main__":
    main()
