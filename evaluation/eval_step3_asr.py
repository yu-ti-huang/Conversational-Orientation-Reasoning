import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class Step3Evaluator:
    def __init__(self, model_path="./models/llama-step3/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Step 3 Evaluator - Model path: {self.model_path}")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        print("Loading Step 3 model for evaluation...")
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
        print("Step 3 model loaded successfully!")

    def load_test_data_from_csv(self, test_csv="step3_test.csv"):
        print(f"Loading test data from: {test_csv}")
        test_df = pd.read_csv(test_csv)
        test_data = []
        for _, row in test_df.iterrows():
            user_input = row['multimodal_input_asr']
            direction_map = {
                'North': '北方',
                'South': '南方',
                'East': '東方',
                'West': '西方'
            }
            expected_direction = direction_map.get(row['target_direction'], row['target_direction'])
            expected_output = f"Final Answer: {expected_direction}"
            test_data.append({
                'input': user_input,
                'expected_output': expected_output,
                'target_direction': row['target_direction'],
                'original_data': row.to_dict()
            })
        print(f"Loaded {len(test_data)} test samples")
        return test_data

    def parse_step3_response(self, response: str) -> Optional[str]:
        response = response.strip()
        orientation_patterns = [
            r'Final Answer[:：]\s*(North|South|East|West)',
            r'結論[:：]\s*使用者面朝([東西南北])方?',
            r'結論[:：].*?使用者.*?面朝([東西南北])方?',
            r'User faces\s*(North|South|East|West)',
            r'使用者面朝([東西南北])方?',
            r'面朝([東西南北])方?',
            r'朝向([東西南北])方?',
            r'(North|South|East|West)\s*$',
            r'([東西南北])方?\s*$'
        ]
        for pattern in orientation_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                direction = match.group(1)
                if direction in ['東', '西', '南', '北']:
                    direction_map = {'東': 'East', '西': 'West', '南': 'South', '北': 'North'}
                    return direction_map.get(direction, direction)
                return direction
        return None

    def extract_expected_orientation(self, expected_output: str) -> Optional[str]:
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
        for direction in ['North', 'South', 'East', 'West']:
            if direction in expected_output:
                return direction
        return None

    def extract_reasoning_components(self, response: str) -> Dict[str, Any]:
        components = {
            'step1_extraction': '',
            'step2_mapping': '',
            'step3_inference': '',
            'final_answer': '',
            'has_step1': False,
            'has_step2': False,
            'has_step3': False,
            'has_final_answer': False,
            'complete_reasoning': False
        }
        step_patterns = [
            (r'Step 1[:：]\s*(.*?)(?=Step 2|Step 3|Final Answer|$)', 'step1_extraction'),
            (r'Step 2[:：]\s*(.*?)(?=Step 3|Final Answer|$)', 'step2_mapping'),
            (r'Step 3[:：]\s*(.*?)(?=Final Answer|$)', 'step3_inference'),
            (r'第一步[:：]\s*(.*?)(?=第二步|第三步|Final Answer|結論|$)', 'step1_extraction'),
            (r'第二步[:：]\s*(.*?)(?=第三步|Final Answer|結論|$)', 'step2_mapping'),
            (r'第三步[:：]\s*(.*?)(?=Final Answer|結論|$)', 'step3_inference')
        ]
        for pattern, step_name in step_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                step_content = match.group(1).strip()
                components[f'has_{step_name.split("_")[0]}'] = True
                components[step_name] = step_content
        final_answer_patterns = [
            r'Final Answer[:：]\s*(.*?)(?=\n|$)',
            r'結論[:：]\s*(.*?)(?=\n|$)',
            r'Answer[:：]\s*(.*?)(?=\n|$)'
        ]
        for pattern in final_answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
            if match:
                components['has_final_answer'] = True
                components['final_answer'] = match.group(1).strip()
                break
        components['complete_reasoning'] = all([
            components['has_step1'],
            components['has_step2'],
            components['has_step3'],
            components['has_final_answer']
        ])
        return components

    def generate_response(self, user_input: str) -> str:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
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

    def run_evaluation(self, test_csv="spatial_test.csv") -> Dict[str, Any]:
        print("="*60)
        print("Step 3 Evaluation: End-to-end Chain-of-Thought Spatial Reasoning")
        print("Using spatial_test.csv for evaluation")
        print("="*60)

        self.load_model()
        test_data = self.load_test_data_from_csv(test_csv)

        results = []
        correct_orientations = 0
        format_errors = 0
        reasoning_quality_scores = []
        complete_reasoning_count = 0
        step_completion_counts = {'step1': 0,'step2': 0,'step3': 0,'final_answer': 0}

        print(f"\nEvaluating {len(test_data)} test samples...")

        for i, test_sample in enumerate(test_data):
            user_input = test_sample['input']
            expected_output = test_sample['expected_output']
            sample_id = test_sample['original_data'].get('sample_id', i)
            target_direction = test_sample['target_direction']
            expected_orientation = self.extract_expected_orientation(expected_output) or target_direction
            try:
                predicted_output = self.generate_response(user_input)
                predicted_orientation = self.parse_step3_response(predicted_output)
                if predicted_orientation is None:
                    format_errors += 1
                    print(f"Format error: Cannot parse orientation")
                orientation_correct = predicted_orientation == expected_orientation
                if orientation_correct:
                    correct_orientations += 1
                predicted_components = self.extract_reasoning_components(predicted_output)
                if predicted_components['has_step1']:
                    step_completion_counts['step1'] += 1
                if predicted_components['has_step2']:
                    step_completion_counts['step2'] += 1
                if predicted_components['has_step3']:
                    step_completion_counts['step3'] += 1
                if predicted_components['has_final_answer']:
                    step_completion_counts['final_answer'] += 1
                if predicted_components['complete_reasoning']:
                    complete_reasoning_count += 1
                reasoning_score = 0.0
                if predicted_components['has_step1']:
                    reasoning_score += 0.25
                if predicted_components['has_step2']:
                    reasoning_score += 0.25
                if predicted_components['has_step3']:
                    reasoning_score += 0.25
                if predicted_components['has_final_answer']:
                    reasoning_score += 0.25
                reasoning_quality_scores.append(reasoning_score)

                print(f"Sample {i+1}/{len(test_data)}")
                print(f"Sample ID: {sample_id}")
                print(f"Target Direction: {target_direction} (Expected: {expected_orientation})")
                print(f"Predicted orientation: {predicted_orientation}")
                print(f"Orientation correct: {'Yes' if orientation_correct else 'No'}")
                print(f"Reasoning score: {reasoning_score:.2f}\n")

                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected': expected_output,
                    'predicted': predicted_output,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'orientation_correct': orientation_correct,
                    'reasoning_score': reasoning_score,
                    'complete_reasoning': predicted_components['complete_reasoning'],
                    'target_direction': target_direction,
                    'predicted_components': predicted_components,
                    'text_variation_type': test_sample['original_data'].get('text_variation_type', 'unknown')
                })
            except Exception as e:
                print(f"Error at sample {i+1}: {e}")
                format_errors += 1
                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected': expected_output,
                    'error': str(e),
                    'orientation_correct': False,
                    'reasoning_score': 0.0,
                    'complete_reasoning': False
                })

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples
        avg_reasoning_score = sum(reasoning_quality_scores) / len(reasoning_quality_scores) if reasoning_quality_scores else 0.0
        complete_reasoning_rate = complete_reasoning_count / total_samples
        step_completion_rates = {step: count / total_samples for step, count in step_completion_counts.items()}

        print("\n" + "="*60)
        print("Step 3 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Orientation accuracy: {orientation_accuracy:.3f} ({correct_orientations}/{total_samples})")
        print(f"Average reasoning quality: {avg_reasoning_score:.3f}")
        print(f"Complete reasoning rate: {complete_reasoning_rate:.3f} ({complete_reasoning_count}/{total_samples})")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")
        print(f"Step 1 completion: {step_completion_rates['step1']:.3f}")
        print(f"Step 2 completion: {step_completion_rates['step2']:.3f}")
        print(f"Step 3 completion: {step_completion_rates['step3']:.3f}")
        print(f"Final Answer completion: {step_completion_rates['final_answer']:.3f}")

        evaluation_results = {
            'method': 'Step 3 - End-to-end Chain-of-Thought Spatial Reasoning',
            'description': 'Complete multimodal CoT reasoning with structured 3-step inference',
            'model_path': self.model_path,
            'test_data_source': test_csv,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'average_reasoning_quality': avg_reasoning_score,
                'complete_reasoning_rate': complete_reasoning_rate,
                'format_error_rate': format_error_rate,
                'step_completion_rates': step_completion_rates,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors,
                'complete_reasoning_count': complete_reasoning_count
            },
            'detailed_results': results
        }

        with open('step3_asr_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: step3_asr_evaluation_results.json")
        return evaluation_results

def main():
    evaluator = Step3Evaluator()
    results = evaluator.run_evaluation("step3_test.csv")
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Step 3 Orientation Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"Step 3 Reasoning Quality: {results['metrics']['average_reasoning_quality']:.3f}")
    print(f"Step 3 Complete Chain Rate: {results['metrics']['complete_reasoning_rate']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
