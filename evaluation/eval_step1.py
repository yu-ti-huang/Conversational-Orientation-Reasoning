import json
import re
import torch
from typing import Tuple, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class Step1Evaluator:
    """Step 1 Evaluator: Vector Calculation and Direction Classification"""

    def __init__(self, model_path="./models/llama-step1/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Step 1 Evaluator - Model path: {self.model_path}")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        print("Loading Step 1 model for evaluation...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        print("Step 1 model loaded successfully!")

    def load_test_set(self):
        test_set_path = f"{self.model_path}/test_set.json"
        try:
            with open(test_set_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Test set not found at {test_set_path}. Please run training first.")

    def parse_test_sample(self, test_sample):
        text = test_sample
        user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        user_match = re.search(user_pattern, text, re.DOTALL)
        user_input = user_match.group(1).strip() if user_match else ""
        user_input = re.sub(r'\n請嚴格按照格式輸出：向量:\(x, y\) 方向:東方/西方/南方/北方$', '', user_input)
        assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        assistant_match = re.search(assistant_pattern, text, re.DOTALL)
        expected_output = assistant_match.group(1).strip() if assistant_match else ""
        return user_input, expected_output

    def parse_vector_direction(self, response: str) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        vector_pattern = r'向量[:：]\s*\((-?\d+),\s*(-?\d+)\)'
        vector_match = re.search(vector_pattern, response)
        direction_pattern = r'方向[:：]\s*(東方|西方|南方|北方)'
        direction_match = re.search(direction_pattern, response)
        vector = (int(vector_match.group(1)), int(vector_match.group(2))) if vector_match else None
        direction = direction_match.group(1) if direction_match else None
        return vector, direction

    def generate_response(self, user_input: str) -> str:
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"{user_input}\n請嚴格按照格式輸出：向量:(x, y) 方向:東方/西方/南方/北方"
                  f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=320)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        del outputs, inputs
        self.clear_memory()
        return generated_text

    def run_evaluation(self) -> Dict[str, Any]:
        print("="*60)
        print("Step 1 Evaluation: Vector Calculation and Direction Classification")
        print("="*60)

        self.load_model()
        test_data = self.load_test_set()

        results = []
        correct_vector = correct_direction = correct_both = format_errors = 0

        for i, test_sample in enumerate(test_data):
            print(f"\nSample {i+1}/{len(test_data)}")
            user_input, expected_output = self.parse_test_sample(test_sample)
            expected_vector, expected_direction = self.parse_vector_direction(expected_output)

            try:
                predicted_output = self.generate_response(user_input)
                pred_vector, pred_direction = self.parse_vector_direction(predicted_output)

                if pred_vector is None or pred_direction is None:
                    format_errors += 1

                vector_correct = pred_vector == expected_vector
                direction_correct = pred_direction == expected_direction
                both_correct = vector_correct and direction_correct

                if vector_correct: correct_vector += 1
                if direction_correct: correct_direction += 1
                if both_correct: correct_both += 1

                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'predicted': predicted_output,
                    'expected_vector': expected_vector,
                    'expected_direction': expected_direction,
                    'predicted_vector': pred_vector,
                    'predicted_direction': pred_direction,
                    'vector_correct': vector_correct,
                    'direction_correct': direction_correct,
                    'both_correct': both_correct
                })

            except Exception as e:
                format_errors += 1
                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'error': str(e),
                    'vector_correct': False,
                    'direction_correct': False,
                    'both_correct': False
                })

        total_samples = len(test_data)
        vector_accuracy = correct_vector / total_samples
        direction_accuracy = correct_direction / total_samples
        overall_accuracy = correct_both / total_samples
        format_error_rate = format_errors / total_samples

        evaluation_results = {
            'step': 'Step 1 - Vector Calculation and Direction Classification',
            'model_path': self.model_path,
            'metrics': {
                'total_samples': total_samples,
                'vector_accuracy': vector_accuracy,
                'direction_accuracy': direction_accuracy,
                'overall_accuracy': overall_accuracy,
                'format_error_rate': format_error_rate,
                'correct_vector': correct_vector,
                'correct_direction': correct_direction,
                'correct_both': correct_both,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        with open('step1_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: step1_evaluation_results.json")
        return evaluation_results

def main():
    evaluator = Step1Evaluator()
    results = evaluator.run_evaluation()
    print("\n" + "="*60)
    print("FINAL RESULTS FOR PAPER:")
    print(f"Step 1 Vector Accuracy: {results['metrics']['vector_accuracy']:.3f}")
    print(f"Step 1 Direction Accuracy: {results['metrics']['direction_accuracy']:.3f}")
    print(f"Step 1 Overall Accuracy: {results['metrics']['overall_accuracy']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
