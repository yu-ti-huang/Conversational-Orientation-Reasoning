import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class B4Evaluator:
    """B4 Evaluator: Fine-tuned Model Without Chain-of-Thought"""

    def __init__(self, model_path="./models/llama-b4/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"B4 Evaluator - Model path: {self.model_path}")

        # Memory optimization
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        """Load the trained B4 model"""
        print("Loading B4 fine-tuned model for evaluation...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
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
        print("B4 model loaded successfully!")

    def load_test_data_from_csv(self, test_csv="step3_test.csv"):
        """Load test data directly from CSV file"""
        print(f"Loading test data from: {test_csv}")
        test_df = pd.read_csv(test_csv)

        test_data = []
        for _, row in test_df.iterrows():
            user_input = row['multimodal_input_clean']

            # Expected output mapping
            direction_map = {
                'North': '北方',
                'South': '南方',
                'East': '東方',
                'West': '西方'
            }
            expected_output = direction_map.get(row['target_direction'], row['target_direction'])

            test_data.append({
                'input': user_input,
                'expected_output': expected_output,
                'original_data': row.to_dict()
            })

        print(f"Loaded {len(test_data)} test samples")
        return test_data

    def extract_orientation(self, response: str) -> Optional[str]:
        """Extract orientation from model response"""
        response = response.strip()
        orientation_patterns = [
            r'([東西南北])方',
            r'面朝([東西南北])',
            r'朝向([東西南北])',
            r'方向.*?([東西南北])',
            r'^([東西南北])$'
        ]
        for pattern in orientation_patterns:
            match = re.search(pattern, response)
            if match:
                direction = match.group(1)
                direction_map = {
                    '東': 'East',
                    '西': 'West',
                    '南': 'South',
                    '北': 'North'
                }
                return direction_map.get(direction, direction)
        return None

    def extract_expected_orientation(self, expected_output: str) -> Optional[str]:
        """Extract expected orientation from expected output"""
        direction_map = {
            '東方': 'East',
            '西方': 'West',
            '南方': 'South',
            '北方': 'North',
            '東': 'East',
            '西': 'West',
            '南': 'South',
            '北': 'North'
        }
        for chinese, english in direction_map.items():
            if chinese in expected_output:
                return english
        return None

    def generate_response(self, user_input: str) -> str:
        """Generate model response for given input (ChatML format)"""
        # ChatML prompt (和訓練一致)
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_input}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=300
        )
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

    def run_evaluation(self, test_csv="step3_test.csv") -> Dict[str, Any]:
        """Run complete B4 evaluation"""
        print("="*60)
        print("B4 Evaluation: Fine-tuned Model Without Chain-of-Thought")
        print("="*60)

        self.load_model()
        test_data = self.load_test_data_from_csv(test_csv)

        results, correct_orientations, format_errors = [], 0, 0

        for i, test_sample in enumerate(test_data):
            user_input = test_sample['input']
            expected_output = test_sample['expected_output']
            sample_id = test_sample['original_data'].get('sample_id', i)
            target_direction = test_sample['original_data'].get('target_direction', 'Unknown')

            expected_orientation = self.extract_expected_orientation(expected_output)

            try:
                predicted_output = self.generate_response(user_input)
                predicted_orientation = self.extract_orientation(predicted_output)

                if not predicted_orientation:
                    format_errors += 1

                orientation_correct = predicted_orientation == expected_orientation
                if orientation_correct:
                    correct_orientations += 1

                print(f"Sample {i+1}/{len(test_data)} | "
                      f"Expected: {expected_orientation}, Predicted: {predicted_orientation}, "
                      f"Correct: {orientation_correct}")

                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected': expected_output,
                    'predicted': predicted_output,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'orientation_correct': orientation_correct,
                    'target_direction': target_direction
                })

            except Exception as e:
                format_errors += 1
                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected': expected_output,
                    'error': str(e),
                    'orientation_correct': False
                })

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples

        print("\n" + "="*60)
        print("B4 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Orientation accuracy: {orientation_accuracy:.3f}")
        print(f"Format error rate: {format_error_rate:.3f}")

        evaluation_results = {
            'method': 'B4 - Fine-tuned Without Chain-of-Thought',
            'description': 'Direct input-to-orientation mapping without reasoning steps',
            'model_path': self.model_path,
            'test_data_source': step3_test.csv,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        with open('b4_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        return evaluation_results

def main():
    evaluator = B4Evaluator()
    results = evaluator.run_evaluation("step3_test.csv")

    print("\n" + "="*60)
    print("FINAL B4 RESULTS FOR PAPER:")
    print(f"B4 Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"B4 Format Error Rate: {results['metrics']['format_error_rate']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
