import json
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class B4Evaluator:
    """B4 Evaluator: Direct classification from fine-tuned model"""

    def __init__(self, model_dir="./models/llama-b4/final_model"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.eval()

    def load_test_set(self, path="data/spatial_test.csv"):
        return pd.read_csv(path)

    def evaluate(self, test_csv="data/spatial_test.csv", save_path="b4_eval_results.json"):
        df = self.load_test_set(test_csv)
        correct, total = 0, 0
        results = []

        for _, row in df.iterrows():
            user_input = row["multimodal_input_clean"]
            expected = {"North": "北方", "South": "南方", "East": "東方", "West": "西方"}[row["target_direction"]]

            prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=5)

            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            predicted = None
            for d in ["東方", "西方", "南方", "北方"]:
                if d in generated:
                    predicted = d
                    break

            correct_flag = (predicted == expected)
            correct += int(correct_flag)
            total += 1

            results.append({
                "input": user_input,
                "expected": expected,
                "predicted": predicted,
                "raw_output": generated,
                "correct": correct_flag
            })

        accuracy = correct / total
        print(f"B4 Evaluation Accuracy: {accuracy:.3f} ({correct}/{total})")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {save_path}")


def main():
    evaluator = B4Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
