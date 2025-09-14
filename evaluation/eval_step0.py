import json
import re
import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def clean_model_output(text: str) -> str:
    special_tokens = ['<|eot_id|>', '<|end_of_text|>', '<|begin_of_text|>',
                      '<|start_header_id|>', '<|end_header_id|>']
    for token in special_tokens:
        text = text.replace(token, '')
    text = text.strip()
    if text and ord(text[-1]) > 127:
        text = text.rstrip()
        if text and text[-1] in ['�', '�']:
            text = text[:-1]
    return text.strip()

def extract_spatial_relations(text: str) -> List[Tuple[str, str]]:
    relations: List[Tuple[str, str]] = []
    pattern1 = r"空間關係\s*[=：:]\s*([^，,\n]+)[，,]\s*參考地標\s*[=：:]\s*([^，,\n\r]+)"
    relations.extend(re.findall(pattern1, text))
    pattern2 = r"關係\d+[：:]\s*空間關係\s*[=：:]\s*([^，,\n]+)[，,]\s*參考地標\s*[=：:]\s*([^，,\n\r]+)"
    relations.extend(re.findall(pattern2, text))
    pattern3 = r"關係[：:]\s*空間關係\s*[=：:]\s*([^，,\n]+)[，,]\s*參考地標\s*[=：:]\s*([^，,\n\r]+)"
    relations.extend(re.findall(pattern3, text))
    relations = list(set([(spatial.strip(), landmark.strip()) for spatial, landmark in relations]))
    return relations

class Step0Evaluator:
    """Complete Step 0 Model Evaluator using saved test set"""

    def __init__(self, model_path="./models/llama-step0/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Step 0 Evaluator - Model path: {self.model_path}")

    def load_model(self):
        print("Loading Step 0 model for evaluation...")
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
        print("Step 0 model loaded successfully!")

    def load_test_set(self):
        test_set_path = f"{self.model_path}/test_set.json"
        try:
            with open(test_set_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            print(f"Loaded {len(test_data)} test samples from {test_set_path}")
            return test_data
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

    def generate_response(self, user_input: str) -> str:
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return clean_model_output(generated_text)

    def run_evaluation(self) -> Dict[str, Any]:
        print("="*60)
        print("Step 0 Evaluation: Spatial Relation Extraction")
        print("="*60)

        self.load_model()
        test_data = self.load_test_set()

        results = []
        exact_matches = 0
        partial_matches = 0
        format_errors = 0

        print(f"\nEvaluating {len(test_data)} test samples...")

        for i, test_sample in enumerate(test_data):
            print(f"\nSample {i+1}/{len(test_data)}")
            user_input, expected_output = self.parse_test_sample(test_sample)
            print(f"Input: {user_input[:100]}")
            print(f"Expected: {expected_output[:100]}")

            try:
                predicted_output = self.generate_response(user_input)
                print(f"Predicted: {predicted_output[:100]}")

                expected_relations = extract_spatial_relations(expected_output)
                predicted_relations = extract_spatial_relations(predicted_output)

                print(f"Expected relations: {expected_relations}")
                print(f"Predicted relations: {predicted_relations}")

                expected_set = set(expected_relations)
                predicted_set = set(predicted_relations)

                is_exact_match = expected_set == predicted_set
                has_partial_match = len(expected_set & predicted_set) > 0

                if not predicted_relations and predicted_output.strip():
                    format_errors += 1

                if is_exact_match:
                    exact_matches += 1
                if has_partial_match:
                    partial_matches += 1

                print(f"Exact match: {'PASS' if is_exact_match else 'FAIL'}")
                print(f"Partial match: {'PASS' if has_partial_match else 'FAIL'}")

                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'predicted': predicted_output,
                    'expected_relations': expected_relations,
                    'predicted_relations': predicted_relations,
                    'exact_match': is_exact_match,
                    'partial_match': has_partial_match
                })

            except Exception as e:
                print(f"Error: {e}")
                format_errors += 1
                results.append({
                    'input': user_input,
                    'expected': expected_output,
                    'error': str(e),
                    'exact_match': False,
                    'partial_match': False
                })

        total_samples = len(test_data)
        exact_match_accuracy = exact_matches / total_samples
        partial_match_accuracy = partial_matches / total_samples
        format_error_rate = format_errors / total_samples

        print("\n" + "="*60)
        print("Step 0 Evaluation Results")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Exact match accuracy: {exact_match_accuracy:.3f} ({exact_matches}/{total_samples})")
        print(f"Partial match accuracy: {partial_match_accuracy:.3f} ({partial_matches}/{total_samples})")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")

        evaluation_results: Dict[str, Any] = {
            'step': 'Step 0 - Spatial Relation Extraction',
            'model_path': self.model_path,
            'metrics': {
                'total_samples': total_samples,
                'exact_match_accuracy': exact_match_accuracy,
                'partial_match_accuracy': partial_match_accuracy,
                'format_error_rate': format_error_rate,
                'exact_matches': exact_matches,
                'partial_matches': partial_matches,
                'format_errors': format_errors
            },
            'detailed_results': results
        }

        with open('step0_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: step0_evaluation_results.json")
        return evaluation_results

def main():
    evaluator = Step0Evaluator()
    results = evaluator.run_evaluation()
    print("\n" + "="*60)
    print("FINAL RESULTS:")
    print(f"Step 0 Exact Match Accuracy: {results['metrics']['exact_match_accuracy']:.3f}")
    print(f"Step 0 Partial Match Accuracy: {results['metrics']['partial_match_accuracy']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
