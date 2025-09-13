import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class B2BaselineEvaluator:
    """B2 Baseline Evaluator: Few-shot Prompting (No CoT)"""

    def __init__(self, test_set_path="./data"):
        self.test_set_path = test_set_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"B2 Baseline Evaluator - Test set path: {self.test_set_path}")
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

    def load_base_model_only(self):
        """Load base Taiwan LLM without LoRA weights"""
        print("Loading base Taiwan LLM model (few-shot baseline)...")
        base_model_path = "yentinglin/Taiwan-LLM-13B-v2.0-Chat"

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print("Base Taiwan LLM model loaded successfully (few-shot)!")

    def load_test_set(self):
        test_set_path = f"{self.test_set_path}/spatial_test.csv"
        df = pd.read_csv(test_set_path)
        print(f"Loaded {len(df)} test samples from {test_set_path}")
        print(f"CSV columns: {df.columns.tolist()}")

        test_data = []
        for _, row in df.iterrows():
            test_data.append({
                'input': row['multimodal_input_clean'],
                'expected_output': row['training_target'],
                'target_direction': row['target_direction'],
                'sample_id': row.get('sample_id', ''),
                'original_text': row.get('original_text', ''),
                'asr_text': row.get('asr_text', ''),
                'has_asr_error': row.get('has_asr_error', False)
            })
        return test_data, {'test_size': len(test_data)}

    def parse_test_sample_b2(self, test_sample):
        return test_sample['input'], test_sample['expected_output']

    def extract_expected_orientation(self, expected_output: str, target_direction: str = None) -> Optional[str]:
        if target_direction:
            mapping = {'North': '北', 'South': '南', 'East': '東', 'West': '西'}
            return mapping.get(target_direction, target_direction)
        for pattern in [r'使用者面朝([東西南北])', r'面朝([東西南北])', r'朝向([東西南北])', r'結論.*?([東西南北])']:
            match = re.search(pattern, expected_output)
            if match:
                return match.group(1)
        return None

    def extract_predicted_orientation(self, response: str) -> Optional[str]:
        response = response.strip()
        response = re.sub(r'<\|[^|]+\|>', '', response)
        response = re.sub(r'<[^>]+>', '', response)

        for d in ['東', '西', '南', '北']:
            if d in response:
                return d

        mapping = {'North': '北','South': '南','East': '東','West': '西',
                   'north': '北','south': '南','east': '東','west': '西'}
        for eng, chi in mapping.items():
            if eng in response:
                return chi

        for pattern in [r'使用者面朝([東西南北])', r'面朝([東西南北])', r'朝向([東西南北])',
                        r'方向.*?([東西南北])', r'([東西南北])方', r'([東西南北])']:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None

    def create_few_shot_examples(self):
        return [
            {"input": "Audio: 我現在在健身房，前面是藥局 | Coordinates: 健身房(4,6), 藥局(4,7)", "output": "北"},
            {"input": "Audio: 我現在在公園，後面是自來水園區 | Coordinates: 公園(0,0), 自來水園區(0,1)", "output": "南"},
            {"input": "Audio: 我現在在基金會，右邊是高中 | Coordinates: 基金會(0,7), 高中(0,6)", "output": "東"},
            {"input": "Audio: 我現在在合作社，左邊是電影院 | Coordinates: 合作社(8,8), 電影院(8,7)", "output": "西"}
        ]

    def generate_response_b2(self, user_input: str) -> str:
        examples = self.create_few_shot_examples()
        few_shot_prompt = "根據語音描述和座標資訊，判斷使用者面朝的方向。\n\n"
        for e in examples:
            few_shot_prompt += f"範例：{e['input']}\n答案：{e['output']}\n\n"
        few_shot_prompt += f"問題：{user_input}\n答案："

        inputs = self.tokenizer(
            few_shot_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=600
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        self.clear_memory()
        return text.strip()

    def run_b2_evaluation(self) -> Dict[str, Any]:
        print("="*60)
        print("B2 Baseline Evaluation: Few-shot Prompting (No CoT)")
        print("="*60)

        self.load_base_model_only()
        test_data, split_info = self.load_test_set()

        results, correct, errors = [], 0, 0

        for i, sample in enumerate(test_data):
            user_input, expected_output = self.parse_test_sample_b2(sample)
            target = sample['target_direction']
            expected = self.extract_expected_orientation(expected_output, target)

            predicted_output = self.generate_response_b2(user_input)
            predicted = self.extract_predicted_orientation(predicted_output)

            if not predicted:
                errors += 1
            if predicted == expected:
                correct += 1

            results.append({
                'sample_id': sample['sample_id'],
                'input': user_input,
                'expected_output': expected_output,
                'predicted_output': predicted_output,
                'expected_orientation': expected,
                'predicted_orientation': predicted,
                'orientation_correct': predicted == expected,
                'target_direction': target,
                'has_asr_error': sample.get('has_asr_error', False)
            })

        total = len(test_data)
        acc = correct / total
        err_rate = errors / total

        print("\n" + "="*60)
        print("B2 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total}")
        print(f"Orientation accuracy: {acc:.3f} ({correct}/{total})")
        print(f"Format error rate: {err_rate:.3f} ({errors}/{total})")

        evaluation_results = {
            'method': 'B2 - Few-shot Prompting (No CoT)',
            'description': 'Few-shot prompt without reasoning steps using base Taiwan LLM',
            'metrics': {
                'total_samples': total,
                'orientation_accuracy': acc,
                'format_error_rate': err_rate,
                'correct_orientations': correct,
                'format_errors': errors
            },
            'detailed_results': results
        }

        os.makedirs("./results", exist_ok=True)
        with open('./results/b2_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print("Results saved to ./results/b2_evaluation_results.json")

        return evaluation_results

def main():
    evaluator = B2BaselineEvaluator()
    results = evaluator.run_b2_evaluation()
    print("\n" + "="*60)
    print("FINAL B2 RESULTS FOR PAPER:")
    print(f"B2 Few-shot Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"B2 Format Error Rate: {results['metrics']['format_error_rate']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
