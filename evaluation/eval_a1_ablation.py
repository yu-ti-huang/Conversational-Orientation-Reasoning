import json
import re
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os
import pandas as pd


class A1AblationEvaluator:
    """A1 Ablation Study: Clean Text Only (No ASR Processing)"""

    def __init__(self, model_path="./models/llama-step3/final_model", csv_path="./spatial_test.csv"):
        self.model_path = model_path
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
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

    def load_test_set(self):
        df = pd.read_csv(self.csv_path)
        return df.to_dict('records')

    def parse_test_sample_a1(self, test_sample):
        if 'multimodal_input_clean' in test_sample:
            clean_input = test_sample.get('multimodal_input_clean', '')
            expected_output = test_sample.get('training_target', '')
            return clean_input, expected_output
        return test_sample.get('input', ''), test_sample.get('expected_output', '')

    def extract_expected_orientation(self, test_sample):
        if 'target_direction' in test_sample:
            mapping = {'North': '北', 'South': '南', 'East': '東', 'West': '西'}
            return mapping.get(test_sample.get('target_direction', ''), None)
        return None

    def extract_predicted_orientation(self, response: str) -> Optional[str]:
        response = response.strip()
        if response in ['東', '西', '南', '北']:
            return response
        patterns = [
            r'使用者面朝([東西南北]方?)',
            r'面朝([東西南北]方?)',
            r'朝向([東西南北]方?)',
            r'方向.*?([東西南北]方?)',
            r'([東西南北])方',
            r'答案.*?([東西南北]方?)',
            r'結論.*?([東西南北]方?)',
            r'^([東西南北])',
            r'([東西南北])$'
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                direction = match.group(1)
                return direction.replace('方', '') if direction else None
        return None

    def generate_response_a1(self, user_input: str) -> str:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}\n\n請只回答使用者面朝的方向，只需要一個字：北、南、東、西。<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n答案："
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
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
        self.load_model()
        test_data = self.load_test_set()

        results = []
        correct_orientations = 0
        format_errors = 0

        for test_sample in test_data:
            user_input, expected_output = self.parse_test_sample_a1(test_sample)
            expected_orientation = self.extract_expected_orientation(test_sample)

            try:
                predicted_output = self.generate_response_a1(user_input)
                predicted_orientation = self.extract_predicted_orientation(predicted_output)

                has_format_error = predicted_orientation is None
                if has_format_error:
                    format_errors += 1

                orientation_correct = (
                    expected_orientation is not None and
                    predicted_orientation is not None and
                    predicted_orientation == expected_orientation
                )
                if orientation_correct:
                    correct_orientations += 1

                results.append({
                    'input': user_input,
                    'expected_output': expected_output,
                    'predicted_output': predicted_output,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'orientation_correct': orientation_correct,
                    'has_format_error': has_format_error
                })
            except Exception as e:
                format_errors += 1
                results.append({
                    'input': user_input,
                    'expected_output': expected_output,
                    'error': str(e),
                    'orientation_correct': False,
                    'has_format_error': True
                })

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples

        evaluation_results = {
            'method': 'A1 - Clean Text Only + Direct Prediction',
            'model_path': self.model_path,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        with open('a1_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        return evaluation_results


def main():
    evaluator = A1AblationEvaluator(csv_path="./spatial_test.csv")
    results = evaluator.run_evaluation()
    print("\n" + "="*60)
    print("FINAL A1 RESULTS FOR PAPER:")
    print(f"A1 Orientation Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"A1 Format Error Rate: {results['metrics']['format_error_rate']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
