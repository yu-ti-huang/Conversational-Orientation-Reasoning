import json
import re
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class Step2Evaluator:
    """Step 2 Evaluator: Spatial Mapping Inference"""

    def __init__(self, model_path="./models/llama-step2/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Step 2 Evaluator - Model path: {self.model_path}")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def load_model(self):
        print("Loading Step 2 model for evaluation...")
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
        print("Step 2 model loaded successfully!")

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
        assistant_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>"
        assistant_match = re.search(assistant_pattern, text, re.DOTALL)
        expected_output = assistant_match.group(1).strip() if assistant_match else ""
        return user_input, expected_output

    def parse_spatial_mapping_response(self, response: str) -> Optional[str]:
        patterns = [
            r'結論：使用者面朝([東西南北])',
            r'結論：.*?面朝([東西南北])',
            r'使用者面朝([東西南北])',
            r'面朝([東西南北])'
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None

    def extract_reasoning_components(self, response: str) -> Dict[str, str]:
        components = {'analysis': '', 'spatial_mapping': '', 'conclusion': ''}
        analysis_match = re.search(r'分析：(.*?)(?=\n|空間映射|結論|$)', response, re.DOTALL)
        if analysis_match:
            components['analysis'] = analysis_match.group(1).strip()
        mapping_match = re.search(r'空間映射：(.*?)(?=\n|結論|$)', response, re.DOTALL)
        if mapping_match:
            components['spatial_mapping'] = mapping_match.group(1).strip()
        conclusion_match = re.search(r'結論：(.*?)(?=\n|$)', response, re.DOTALL)
        if conclusion_match:
            components['conclusion'] = conclusion_match.group(1).strip()
        return components

    def generate_response(self, user_input: str) -> str:
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
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
        print("Step 2 Evaluation: Spatial Mapping Inference")
        print("="*60)

        self.load_model()
        test_data = self.load_test_set()

        results = []
        correct_orientations = format_errors = 0
        reasoning_quality_scores = []

        for test_sample in test_data:
            user_input, expected_output = self.parse_test_sample(test_sample)
            expected_orientation = self.parse_spatial_mapping_response(expected_output)
            try:
                predicted_output = self.generate_response(user_input)
                predicted_orientation = self.parse_spatial_mapping_response(predicted_output)

                if predicted_orientation is None:
                    format_errors += 1

                orientation_correct = predicted_orientation == expected_orientation
                if orientation_correct:
                    correct_orientations += 1

                predicted_components = self.extract_reasoning_components(predicted_output)
                expected_components = self.extract_reasoning_components(expected_output)

                reasoning_score = 0.0
                if predicted_components['analysis']: reasoning_score += 0.3
                if predicted_components['spatial_mapping']: reasoning_score += 0.3
                if predicted_components['conclusion']: reasoning_score += 0.4

                reasoning_quality_scores.append(reasoning_score)

                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'predicted': predicted_output,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'orientation_correct': orientation_correct,
                    'reasoning_score': reasoning_score,
                    'predicted_components': predicted_components,
                    'expected_components': expected_components
                })

            except Exception as e:
                format_errors += 1
                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'error': str(e),
                    'orientation_correct': False,
                    'reasoning_score': 0.0
                })

        total_samples = len(test_data)
        orientation_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples
        avg_reasoning_score = sum(reasoning_quality_scores) / len(reasoning_quality_scores) if reasoning_quality_scores else 0.0

        evaluation_results = {
            'step': 'Step 2 - Spatial Mapping Inference',
            'model_path': self.model_path,
            'metrics': {
                'total_samples': total_samples,
                'orientation_accuracy': orientation_accuracy,
                'average_reasoning_quality': avg_reasoning_score,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        with open('step2_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: step2_evaluation_results.json")
        return evaluation_results

def main():
    evaluator = Step2Evaluator()
    results = evaluator.run_evaluation()
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Step 2 Orientation Accuracy: {results['metrics']['orientation_accuracy']:.3f}")
    print(f"Step 2 Reasoning Quality: {results['metrics']['average_reasoning_quality']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
