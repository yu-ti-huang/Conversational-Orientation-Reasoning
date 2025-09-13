import json
import re
import os
import gc
import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class B3BaselineEvaluator:
    """B3 Baseline Evaluator: Few-shot Prompting + CoT"""

    def __init__(self, test_set_path="./data"):
        self.test_set_path = test_set_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()
        print(f"B3 Baseline Evaluator - Test set path: {self.test_set_path}")
        print(f"Device: {self.device}")

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_base_model_only(self):
        print("Loading base Taiwan LLM model (few-shot CoT baseline)...")
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
        print("Base Taiwan LLM model loaded successfully (few-shot CoT)!")

    def load_test_set(self):
        csv_path = os.path.join(self.test_set_path, "step3_test.csv")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} test samples from {csv_path}")

        test_data = []
        for _, row in df.iterrows():
            if all(k in df.columns for k in ['multimodal_input_clean','training_target','target_direction']):
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

    def parse_test_sample(self, test_sample):
        if isinstance(test_sample, dict) and 'input' in test_sample:
            return test_sample['input'], test_sample['expected_output']
        text = test_sample
        user_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        user_match = re.search(user_pattern, text, re.DOTALL)
        assistant_match = re.search(assistant_pattern, text, re.DOTALL)
        user_input = user_match.group(1).strip() if user_match else ""
        expected_output = assistant_match.group(1).strip() if assistant_match else ""
        return user_input, expected_output

    def create_few_shot_cot_examples(self):
        return [
            {
                "input": "Audio: 我現在在健身房，前面是藥局 | Coordinates: 健身房(4,6), 藥局(4,7)",
                "output": """第一步：提取空間關係
關係：空間關係=前面，參考地標=藥局

第二步：計算絕對方向
參考地標=藥局，從健身房到藥局方向向量：(4,7)-(4,6)=(0,1)，方向：北方

第三步：推理朝向
藥局在北方且在使用者的前面，空間映射：面朝北方時，前面=北方
結論：使用者面朝北"""
            },
            {
                "input": "Audio: 我現在在公園，後面是自來水園區 | Coordinates: 公園(0,0), 自來水園區(0,1)",
                "output": """第一步：提取空間關係
關係：空間關係=後面，參考地標=自來水園區

第二步：計算絕對方向
參考地標=自來水園區，從公園到自來水園區方向向量：(0,1)-(0,0)=(0,1)，方向：北方

第三步：推理朝向
自來水園區在北方且在使用者的後面，空間映射：面朝南方時，後面=北方
結論：使用者面朝南"""
            },
            {
                "input": "Audio: 我現在在基金會，右邊是高中 | Coordinates: 基金會(0,7), 高中(0,6)",
                "output": """第一步：提取空間關係
關係：空間關係=右邊，參考地標=高中

第二步：計算絕對方向
參考地標=高中，從基金會到高中方向向量：(0,6)-(0,7)=(0,-1)，方向：南方

第三步：推理朝向
高中在南方且在使用者的右邊，空間映射：面朝東方時，右邊=南方
結論：使用者面朝東"""
            },
            {
                "input": "Audio: 我現在在合作社，左邊是電影院 | Coordinates: 合作社(8,8), 電影院(8,7)",
                "output": """第一步：提取空間關係
關係：空間關係=左邊，參考地標=電影院

第二步：計算絕對方向
參考地標=電影院，從合作社到電影院方向向量：(8,7)-(8,8)=(0,-1)，方向：南方

第三步：推理朝向
電影院在南方且在使用者的左邊，空間映射：面朝西方時，左邊=南方
結論：使用者面朝西"""
            }
        ]

    def parse_test_sample_b3(self, test_sample):
        return self.parse_test_sample(test_sample)

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
        for pattern in [r'使用者面朝([東西南北])', r'面朝([東西南北])', r'朝向([東西南北])', r'方向.*?([東西南北])', r'([東西南北])方']:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None

    def analyze_cot_structure(self, response: str) -> Dict[str, Any]:
        analysis = {'has_step1': False, 'has_step2': False, 'has_step3': False, 'has_conclusion': False, 'reasoning_quality_score': 0.0}
        if re.search(r'第一步', response):
            analysis['has_step1'] = True
        if re.search(r'第二步', response):
            analysis['has_step2'] = True
        if re.search(r'第三步', response):
            analysis['has_step3'] = True
        if re.search(r'結論', response):
            analysis['has_conclusion'] = True
        score = 0.0
        if analysis['has_step1']: score += 0.25
        if analysis['has_step2']: score += 0.25
        if analysis['has_step3']: score += 0.25
        if analysis['has_conclusion']: score += 0.25
        analysis['reasoning_quality_score'] = score
        return analysis

    def generate_response_b3(self, user_input: str) -> str:
        examples = self.create_few_shot_cot_examples()
        few_shot_prompt = "請根據語音描述和座標資訊，使用三步驟推理判斷使用者面朝的方向。\n\n"
        for i, ex in enumerate(examples):
            few_shot_prompt += f"範例 {i+1}：\n輸入：{ex['input']}\n輸出：{ex['output']}\n\n"
        few_shot_prompt += f"現在請使用相同的三步驟推理方式：\n輸入：{user_input}\n輸出："

        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{few_shot_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1200
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        del outputs, inputs
        self.clear_memory()
        return generated_text

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
                    'cot_analysis': r.get('cot_analysis', {}),
                    'error_type': 'orientation_incorrect'
                })
        return problematic

    def run_b3_evaluation(self) -> Dict[str, Any]:
        print("="*60)
        print("B3 Baseline Evaluation: Few-shot Prompting + CoT")
        print("="*60)

        self.load_base_model_only()
        test_data, split_info = self.load_test_set()

        results = []
        correct_orientations = 0
        format_errors = 0
        reasoning_quality_scores = []

        for i, test_sample in enumerate(test_data):
            user_input, expected_output = self.parse_test_sample_b3(test_sample)
            target_direction = test_sample.get('target_direction', '')
            sample_id = test_sample.get('sample_id', i)
            expected_orientation = self.extract_expected_orientation(expected_output, target_direction)

            predicted_output = self.generate_response_b3(user_input)
            cot_analysis = self.analyze_cot_structure(predicted_output)
            reasoning_quality_scores.append(cot_analysis['reasoning_quality_score'])
            predicted_orientation = self.extract_predicted_orientation(predicted_output)

            if not predicted_orientation:
                format_errors += 1

            orientation_correct = (predicted_orientation == expected_orientation)
            if orientation_correct:
                correct_orientations += 1

            results.append({
                'sample_id': sample_id,
                'input': user_input,
                'expected_output': expected_output,
                'predicted_output': predicted_output,
                'expected_orientation': expected_orientation,
                'predicted_orientation': predicted_orientation,
                'orientation_correct': orientation_correct,
                'target_direction': target_direction,
                'has_asr_error': test_sample.get('has_asr_error', False),
                'cot_analysis': cot_analysis
            })

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples
        avg_reasoning_quality = sum(reasoning_quality_scores) / len(reasoning_quality_scores) if reasoning_quality_scores else 0.0
        problematic_cases = self.analyze_problematic_cases(results)

        print("\n" + "="*60)
        print("B3 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Orientation accuracy: {orientation_accuracy:.3f} ({correct_orientations}/{total_samples})")
        print(f"Average reasoning quality: {avg_reasoning_quality:.3f}")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")

        evaluation_results = {
            'method': 'B3 - Few-shot Prompting + CoT',
            'description': 'Few-shot CoT prompting using base Taiwan LLM',
            'base_model': 'yentinglin/Taiwan-LLM-13B-v2.0-Chat',
            'split_info': split_info,
            'problematic_cases': problematic_cases,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'average_reasoning_quality': avg_reasoning_quality,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        os.makedirs("./results", exist_ok=True)
        out_path = './results/b3_evaluation_results.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {out_path}")

        return evaluation_results

def main():
    evaluator = B3BaselineEvaluator()
    results = evaluator.run_b3_evaluation()
    print("\n" + "="*60)
    print("FINAL B3 RESULTS:")
    print(f"B3 Few-shot CoT Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"B3 Reasoning Quality: {results['metrics']['average_reasoning_quality']:.3f}")
    print(f"B3 Format Error Rate: {results['metrics']['format_error_rate']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
