import json
import re
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList
import gc
import os

class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids_list):
        super().__init__()
        self.stop_ids_list = stop_ids_list
    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_ids_list:
            need = stop_ids.shape[1]
            if input_ids.shape[1] >= need and torch.equal(input_ids[0, -need:], stop_ids[0]):
                return True
        return False

class R3EvaluatorReferentialAmbiguity:
    def __init__(self, model_path="./models/llama-step3/final_model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"R3 Evaluator - Model path: {self.model_path}")
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.clear_memory()

    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def convert_for_json(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_for_json(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj

    def load_model(self):
        print("Loading Step 3 model for R3 referential ambiguity evaluation...")
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
        print("Step 3 model loaded successfully!")

    def load_robustness_dataset(self, excel_path="r3_robustness.xlsx"):
        print(f"Loading robustness test dataset from: {excel_path}")
        df = pd.read_excel(excel_path) if excel_path.endswith('.xlsx') else pd.read_csv(excel_path)
        print(f"Robustness dataset loaded: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        if 'variation_subtype' in df.columns:
            variation_types = df['variation_subtype'].value_counts()
            print("Variation subtypes:")
            for var_type, count in variation_types.items():
                print(f"  {var_type}: {count} samples")
        return df

    def create_multimodal_input(self, row):
        audio_part = row['original_text']
        coordinates_part = row['landmarks']
        return f"Audio: {audio_part} | Coordinates: {coordinates_part}"

    def extract_orientation(self, response: str) -> Optional[str]:
        response = response.strip()
        patterns = [
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
        for p in patterns:
            m = re.search(p, response, re.IGNORECASE)
            if m:
                d = m.group(1)
                if d in ['東', '西', '南', '北']:
                    zh2en = {'東': 'East', '西': 'West', '南': 'South', '北': 'North'}
                    return zh2en.get(d, d)
                return d
        return None

    def generate_response(self, user_input: str) -> str:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        stop_phrases = ["\nFinal Answer:", "\n結論：", "\n結論:"]
        stop_ids_list = [self.tokenizer(s, return_tensors="pt").input_ids.to(device) for s in stop_phrases]
        stopping = StoppingCriteriaList([StopOnSequences(stop_ids_list)])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping
            )
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        del outputs, inputs
        self.clear_memory()
        return generated_text

    def analyze_ambiguity_robustness(self, results, test_df) -> Dict[str, Any]:
        analysis = {'overall_robustness': 0.0, 'subtype_performance': {}, 'ambiguity_impact': {}, 'variation_details_performance': {}}
        correct = sum(1 for r in results if r.get('orientation_correct', False))
        total = len(results)
        analysis['overall_robustness'] = correct / total if total > 0 else 0.0

        if 'variation_subtype' in test_df.columns:
            for subtype in test_df['variation_subtype'].unique():
                if pd.isna(subtype):
                    continue
                subs = [r for i, r in enumerate(results) if test_df.iloc[i]['variation_subtype'] == subtype]
                if subs:
                    acc = sum(1 for r in subs if r.get('orientation_correct', False)) / len(subs)
                    analysis['subtype_performance'][subtype] = {'accuracy': acc, 'count': len(subs)}

        if 'variation_details' in test_df.columns:
            for detail in test_df['variation_details'].unique():
                if pd.isna(detail):
                    continue
                dets = [r for i, r in enumerate(results) if test_df.iloc[i]['variation_details'] == detail]
                if dets and len(dets) >= 2:
                    acc = sum(1 for r in dets if r.get('orientation_correct', False)) / len(dets)
                    analysis['variation_details_performance'][detail] = {'accuracy': acc, 'count': len(dets)}
        return analysis

    def error_pattern_analysis(self, results, test_df) -> Dict[str, Any]:
        out = {'errors_by_ambiguity_type': {}, 'errors_by_variation_details': {}, 'all_error_cases': []}
        for i, r in enumerate(results):
            if not r.get('orientation_correct', False):
                out['all_error_cases'].append({
                    'sample_id': r.get('sample_id', i),
                    'input': r.get('input', ''),
                    'expected_orientation': r.get('expected_orientation', ''),
                    'predicted_orientation': r.get('predicted_orientation', ''),
                    'predicted_output': r.get('predicted_output', ''),
                    'variation_subtype': r.get('variation_subtype', ''),
                    'variation_details': r.get('variation_details', ''),
                    'spatial_relation': r.get('spatial_relation', ''),
                    'error_type': 'format_error' if r.get('predicted_orientation') is None else 'misclassification'
                })
                amb = r.get('variation_subtype', 'unknown')
                out['errors_by_ambiguity_type'][amb] = out['errors_by_ambiguity_type'].get(amb, 0) + 1
                det = r.get('variation_details', 'unknown')
                out['errors_by_variation_details'][det] = out['errors_by_variation_details'].get(det, 0) + 1
        return out

    def compare_with_clear_references(self, ambiguity_accuracy) -> Dict[str, Any]:
        comp = {'ambiguity_tolerance': 0.0, 'performance_drop': 0.0, 'robustness_assessment': ''}
        try:
            with open('step3_evaluation_results.json', 'r', encoding='utf-8') as f:
                clear_results = json.load(f)
                clear_acc = clear_results['metrics']['orientation_accuracy']
        except FileNotFoundError:
            print("Warning: Step3 evaluation results not found for comparison")
            comp['robustness_assessment'] = 'Cannot compare - Step3 results unavailable'
            return comp

        comp['ambiguity_tolerance'] = ambiguity_accuracy / clear_acc if clear_acc > 0 else 0.0
        comp['performance_drop'] = clear_acc - ambiguity_accuracy
        comp['clear_reference_accuracy'] = clear_acc
        comp['ambiguous_reference_accuracy'] = ambiguity_accuracy

        if comp['ambiguity_tolerance'] >= 0.85:
            comp['robustness_assessment'] = 'Excellent robustness to referential ambiguity'
        elif comp['ambiguity_tolerance'] >= 0.75:
            comp['robustness_assessment'] = 'Good robustness to referential ambiguity'
        elif comp['ambiguity_tolerance'] >= 0.65:
            comp['robustness_assessment'] = 'Acceptable robustness with room for improvement'
        else:
            comp['robustness_assessment'] = 'Poor robustness - model struggles with ambiguous references'
        return comp

    def run_r3_evaluation(self, robustness_path="r3_robustness.xlsx") -> Dict[str, Any]:
        print("="*60)
        print("R3 Evaluation: Referential Ambiguity Set - Robustness Analysis")
        print("Using r3_robustness.xlsx for evaluation")
        print("="*60)

        self.load_model()
        test_df = self.load_robustness_dataset(robustness_path)

        results = []
        correct_orientations = 0
        format_errors = 0

        print(f"\nEvaluating {len(test_df)} referential ambiguity test samples...")

        for i, (_, row) in enumerate(test_df.iterrows()):
            user_input = self.create_multimodal_input(row)
            expected_orientation = row['target_direction']
            zh2en = {'東':'East','西':'West','南':'South','北':'North','東方':'East','西方':'West','南方':'South','北方':'North'}
            expected_orientation = zh2en.get(expected_orientation, expected_orientation)
            sample_id = row.get('sample_id', i)
            variation_subtype = row.get('variation_subtype', 'unknown')
            variation_details = row.get('variation_details', 'N/A')

            try:
                predicted_output = self.generate_response(user_input)
                predicted_orientation = self.extract_orientation(predicted_output)

                if predicted_orientation is None:
                    format_errors += 1

                orientation_correct = (predicted_orientation == expected_orientation)
                if orientation_correct:
                    correct_orientations += 1

                print(f"Sample {i+1}/{len(test_df)}")
                print(f"Sample ID: {sample_id}")
                print(f"Input: {user_input[:80]}...")
                print(f"Ambiguity type: {variation_subtype}")
                print(f"Variation: {variation_details}")
                print(f"Expected: {expected_orientation}")
                print(f"Predicted: {predicted_output[:100]}...")
                print(f"Predicted orientation: {predicted_orientation}")
                print(f"Orientation correct: {'Yes' if orientation_correct else 'No'}\n")

                results.append({
                    'sample_id': self.convert_for_json(sample_id),
                    'input': user_input,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'predicted_output': predicted_output,
                    'orientation_correct': orientation_correct,
                    'variation_subtype': variation_subtype,
                    'variation_details': variation_details,
                    'spatial_relation': row.get('spatial_relations', ''),
                    'landmark_count': self.convert_for_json(row.get('landmark_count', 0))
                })

            except Exception as e:
                print(f"Error at sample {i+1}: {e}")
                format_errors += 1
                results.append({
                    'sample_id': self.convert_for_json(sample_id),
                    'input': user_input,
                    'expected_orientation': expected_orientation,
                    'error': str(e),
                    'orientation_correct': False,
                    'variation_subtype': variation_subtype,
                    'variation_details': variation_details
                })

        total_samples = len(test_df)
        ambiguity_accuracy = correct_orientations / total_samples if total_samples else 0.0
        format_error_rate = format_errors / total_samples if total_samples else 0.0

        robustness_analysis = self.analyze_ambiguity_robustness(results, test_df)
        error_patterns = self.error_pattern_analysis(results, test_df)
        clear_comparison = self.compare_with_clear_references(ambiguity_accuracy)

        print("\n" + "="*60)
        print("R3 Evaluation Results - Referential Ambiguity Robustness")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Ambiguity robustness accuracy: {ambiguity_accuracy:.3f} ({correct_orientations}/{total_samples})")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")

        if 'clear_reference_accuracy' in clear_comparison:
            print("\nRobustness Comparison:")
            print(f"Clear reference accuracy: {clear_comparison['clear_reference_accuracy']:.3f}")
            print(f"Ambiguous reference accuracy: {clear_comparison['ambiguous_reference_accuracy']:.3f}")
            print(f"Ambiguity tolerance: {clear_comparison['ambiguity_tolerance']:.3f}")
            print(f"Performance drop: {clear_comparison['performance_drop']:.3f}")

        if robustness_analysis['subtype_performance']:
            print("\nPerformance by Ambiguity Type:")
            for subtype, stats in robustness_analysis['subtype_performance'].items():
                print(f"{subtype}: {stats['accuracy']:.3f} ({stats['count']} samples)")

        if robustness_analysis['variation_details_performance']:
            print("\nVariation Details Impact:")
            for variation, stats in robustness_analysis['variation_details_performance'].items():
                print(f"{variation}: {stats['accuracy']:.3f} ({stats['count']} samples)")

        print(f"\nError Summary: {len(error_patterns['all_error_cases'])} cases")

        evaluation_results = {
            'method': 'R3 - Referential Ambiguity Set',
            'description': 'Robustness to semantic underspecification, incomplete utterances, and referential ambiguity using r3_robustness.xlsx',
            'model_path': self.model_path,
            'test_data_source': robustness_path,
            'metrics': {
                'total_samples': total_samples,
                'ambiguity_robustness_accuracy': ambiguity_accuracy,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors,
                'robustness_analysis': self.convert_for_json(robustness_analysis),
                'clear_comparison': self.convert_for_json(clear_comparison)
            },
            'error_patterns': self.convert_for_json(error_patterns),
            'detailed_results': self.convert_for_json(results)
        }

        with open('r3_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print("\nResults saved to: r3_evaluation_results.json")
        return evaluation_results

def main():
    evaluator = R3EvaluatorReferentialAmbiguity()
    results = evaluator.run_r3_evaluation("r3_robustness.xlsx")
    print("\n" + "="*60)
    print("FINAL R3 RESULTS FOR PAPER:")
    print(f"R3 Ambiguity Robustness: {results['metrics']['ambiguity_robustness_accuracy']:.3f}")
    if 'ambiguity_tolerance' in results['metrics']['clear_comparison']:
        print(f"R3 Ambiguity Tolerance: {results['metrics']['clear_comparison']['ambiguity_tolerance']:.3f}")
        print(f"R3 Performance Drop: {results['metrics']['clear_comparison']['performance_drop']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
