import json
import re
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class B1BaselineEvaluator:
    """B1 Baseline Evaluator: Zero-shot Pretrained Model"""

    def __init__(self, csv_path="./step3_test.csv"):
        self.csv_path = csv_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"B1 Baseline Evaluator - CSV: {self.csv_path}")
        print(f"Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def setup_data_files(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Missing test CSV at {self.csv_path}. Please place step3_test.csv and retry."
            )

    def load_base_model_only(self):
        print("Loading base Taiwan LLM model (zero-shot baseline)...")
        base_model_path = "yentinglin/Taiwan-LLM-13B-v2.0-Chat"

        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                use_fast=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer loaded successfully!")

            print("Setting up quantization config...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            print("Base Taiwan LLM model loaded successfully (zero-shot)!")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        except torch.cuda.OutOfMemoryError:
            print("Out of GPU memory. Tips: restart runtime, use a larger GPU, or close other processes.")
            raise
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

    def load_test_set(self):
        import pandas as pd

        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} test samples from {self.csv_path}")
            print(f"CSV columns: {df.columns.tolist()}")

            test_data = []
            for _, row in df.iterrows():
                if 'multimodal_input_clean' in df.columns and 'training_target' in df.columns and 'target_direction' in df.columns:
                    test_data.append({
                        'input': row['multimodal_input_clean'],
                        'expected_output': row['training_target'],
                        'target_direction': row['target_direction'],
                        'sample_id': row.get('sample_id', ''),
                        'original_text': row.get('original_text', ''),
                        'asr_text': row.get('asr_text', ''),
                        'has_asr_error': row.get('has_asr_error', False)
                    })
                else:
                    missing = [c for c in ['multimodal_input_clean', 'training_target', 'target_direction'] if c not in df.columns]
                    raise ValueError(f"Missing required columns: {missing}")

            print(f"Successfully converted {len(test_data)} samples")
            print(f"Sample data structure: {list(test_data[0].keys())}")
            return test_data, {'test_size': len(test_data)}

        except FileNotFoundError:
            raise FileNotFoundError(f"Test set not found at {self.csv_path}.")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

    def parse_test_sample(self, test_sample):
        if isinstance(test_sample, dict) and 'input' in test_sample:
            return test_sample['input'], test_sample['expected_output']
        else:
            text = test_sample
            user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
            user_match = re.search(user_pattern, text, re.DOTALL)
            user_input = user_match.group(1).strip() if user_match else ""

            assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
            assistant_match = re.search(assistant_pattern, text, re.DOTALL)
            expected_output = assistant_match.group(1).strip() if assistant_match else ""
            return user_input, expected_output

    def parse_test_sample_b1(self, test_sample):
        user_input, expected_output = self.parse_test_sample(test_sample)
        return user_input, expected_output

    def extract_expected_orientation(self, expected_output: str, target_direction: str = None) -> Optional[str]:
        if target_direction:
            mapping = {'North': '北', 'South': '南', 'East': '東', 'West': '西'}
            return mapping.get(target_direction, target_direction)

        patterns = [r'使用者面朝([東西南北])', r'面朝([東西南北])', r'朝向([東西南北])', r'結論.*?([東西南北])']
        for p in patterns:
            m = re.search(p, expected_output)
            if m:
                return m.group(1)
        return None

    def extract_predicted_orientation(self, response: str) -> Optional[str]:
        response = re.sub(r'<\|[^|]+\|>', '', response.strip())
        response = re.sub(r'<[^>]+>', '', response)

        for d in ['東', '西', '南', '北']:
            if d in response:
                return d

        eng2chi = {'North': '北', 'South': '南', 'East': '東', 'West': '西',
                   'north': '北', 'south': '南', 'east': '東', 'west': '西'}
        for eng, chi in eng2chi.items():
            if eng in response:
                return chi

        patterns = [r'([東西南北])', r'面朝([東西南北])', r'朝向([東西南北])', r'方向.*?([東西南北])', r'([東西南北])方']
        for p in patterns:
            m = re.search(p, response)
            if m:
                return m.group(1)
        return None

    def generate_response_b1(self, user_input: str) -> str:
        prompt = f"Question: {user_input}\n使用者面朝哪個方向？請回答東西南北其中一個。\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            text = text.replace("ASSISTANT:", "").replace("Assistant:", "").strip()
            del outputs, inputs
            self.clear_memory()
            return text
        except Exception as e:
            print(f"Generation error: {e}")
            self.clear_memory()
            return ""

    def analyze_problematic_cases(self, results):
        problematic = []
        for i, r in enumerate(results):
            if not r.get('orientation_correct', False):
                problematic.append({
                    'sample_index': i,
                    'input': r['input'],
                    'predicted_output': r['predicted_output'],
                    'expected_orientation': r['expected_orientation'],
                    'predicted_orientation': r['predicted_orientation'],
                    'error_type': 'orientation_incorrect'
                })
        return problematic

    def run_b1_evaluation(self) -> Dict[str, Any]:
        print("="*60)
        print("B1 Baseline Evaluation: Zero-shot")
        print("Description: Taiwan LLM without fine-tuning; directly predicts from raw input")
        print("="*60)

        self.setup_data_files()
        self.load_base_model_only()
        test_data, split_info = self.load_test_set()

        results = []
        correct_orientations = 0
        format_errors = 0

        print(f"\nEvaluating all {len(test_data)} test samples...")

        for i, test_sample in enumerate(test_data):
            print(f"\nSample {i+1}/{len(test_data)}")
            user_input, expected_output = self.parse_test_sample_b1(test_sample)
            target_direction = test_sample.get('target_direction', '')
            sample_id = test_sample.get('sample_id', i)

            expected_orientation = self.extract_expected_orientation(expected_output, target_direction)

            print(f"Sample ID: {sample_id}")
            print(f"Input: {user_input[:100]}...")
            print(f"Target Direction: {target_direction} (Expected: {expected_orientation})")

            try:
                predicted_output = self.generate_response_b1(user_input)
                print(f"Predicted: {predicted_output[:150]}...")
                predicted_orientation = self.extract_predicted_orientation(predicted_output)

                print(f"Expected orientation: {expected_orientation}")
                print(f"Predicted orientation: {predicted_orientation}")

                if not predicted_orientation:
                    format_errors += 1
                    print("Format error: cannot extract final orientation")

                orientation_correct = predicted_orientation == expected_orientation
                if orientation_correct:
                    correct_orientations += 1

                print(f"Orientation correct: {'Yes' if orientation_correct else 'No'}")

                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected_output': expected_output,
                    'predicted_output': predicted_output,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'orientation_correct': orientation_correct,
                    'target_direction': target_direction,
                    'has_asr_error': test_sample.get('has_asr_error', False)
                })

            except Exception as e:
                print(f"Error: {e}")
                format_errors += 1
                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected_output': expected_output,
                    'error': str(e),
                    'orientation_correct': False,
                    'target_direction': target_direction,
                    'has_asr_error': test_sample.get('has_asr_error', False)
                })

        problematic_cases = self.analyze_problematic_cases(results)

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples

        print("\n" + "="*60)
        print("B1 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Orientation accuracy: {orientation_accuracy:.3f} ({correct_orientations}/{total_samples})")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")

        print(f"\nError case analysis ({len(problematic_cases)} cases):")
        for case in problematic_cases[:5]:
            print(f"\nSample {case['sample_index']}:")
            print(f"  Input: {case['input'][:100]}...")
            print(f"  Expected: {case['expected_orientation']}")
            print(f"  Predicted: {case['predicted_orientation']}")
            print(f"  Output: {case['predicted_output'][:100]}...")
            print("-" * 50)

        evaluation_results = {
            'method': 'B1 - Zero-shot Baseline',
            'description': 'Taiwan LLM without fine-tuning; directly predicts from raw input',
            'base_model': 'yentinglin/Taiwan-LLM-13B-v2.0-Chat',
            'split_info': split_info,
            'problematic_cases': problematic_cases,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        os.makedirs("./results", exist_ok=True)
        out_path = "./results/b1_evaluation_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {out_path}")

        return evaluation_results

def main():
    evaluator = B1BaselineEvaluator(csv_path="./step3_test.csv")
    results = evaluator.run_b1_evaluation()
    print("\n" + "="*60)
    print("FINAL B1 RESULTS:")
    print(f"B1 Zero-shot Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"B1 Format Error Rate: {results['metrics']['format_error_rate']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
