import json
import re
import torch
import pandas as pd
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

class R2EvaluatorCrossDomain:
    """R2 Evaluator: Cross-domain Test Set - Generalization Analysis"""

    def __init__(self, model_path="./models/llama-step3/final_model", data_path="./TaipeiStation_Generalization.xlsx"):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"R2 Evaluator - Model path: {self.model_path}")
        print(f"R2 Evaluator - Cross-domain data: {self.data_path}")

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
        """Load the trained Step 3 model"""
        print("Loading Step 3 model for R2 cross-domain evaluation...")

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
        print("Step 3 model loaded successfully!")

    def load_cross_domain_dataset(self):
        """Load cross-domain test dataset from Excel or CSV file"""
        print(f"Loading cross-domain test dataset from: {self.data_path}")

        try:
            # Load Excel or CSV file
            if self.data_path.endswith('.xlsx'):
                try:
                    df = pd.read_excel(self.data_path)
                except ImportError:
                    raise ImportError("openpyxl is required to read Excel files. Install it with: pip install openpyxl")
            else:
                df = pd.read_csv(self.data_path)

            print(f"Cross-domain dataset loaded: {len(df)} samples")
            print(f"Domain: Taipei Station (different from training domain: Gongguan)")
            print(f"Columns: {list(df.columns)}")

            return df

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def extract_expected_orientation(self, test_sample):
        """Extract expected orientation from CSV test sample"""
        # Handle both dict and pandas Series
        if hasattr(test_sample, 'get'):
            target_direction = test_sample.get('target_direction', '')
        else:
            target_direction = getattr(test_sample, 'target_direction', '')

        # Convert English to Chinese
        direction_map = {
            'North': '北', 'South': '南',
            'East': '東', 'West': '西'
        }
        return direction_map.get(target_direction, None)

    def extract_predicted_orientation(self, response: str) -> Optional[str]:
        """Extract orientation from model response with Final Answer pattern"""
        # Clean response first
        response = response.strip()

        # Primary pattern: Final Answer
        final_answer_pattern = r'Final Answer[:：]\s*([北南東西])'
        match = re.search(final_answer_pattern, response)
        if match:
            return match.group(1)

        # Fallback patterns
        orientation_patterns = [
            r'結論[:：]\s*使用者面朝([東西南北])方?',
            r'使用者面朝([東西南北])方?',
            r'面朝([東西南北])方?',
            r'朝向([東西南北])方?',
            r'([東西南北])方?\s*$'
        ]

        for pattern in orientation_patterns:
            match = re.search(pattern, response)
            if match:
                direction = match.group(1)
                return direction.replace('方', '') if direction else None

        return None

    def generate_response(self, user_input: str) -> str:
        """Generate model response for given input"""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        # Move to correct device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Extract only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Clean memory
        del outputs, inputs
        self.clear_memory()

        return generated_text

    def analyze_cross_domain_performance(self, results, test_df) -> Dict[str, Any]:
        """Analyze cross-domain generalization performance"""
        analysis = {
            'domain_transfer_accuracy': 0.0,
            'asr_error_impact': {},
            'spatial_relation_performance': {}
        }

        # Calculate domain transfer accuracy
        correct = sum(1 for r in results if r.get('orientation_correct', False))
        total = len(results)
        analysis['domain_transfer_accuracy'] = correct / total if total > 0 else 0.0

        # ASR error impact analysis
        if 'has_asr_error' in test_df.columns:
            asr_error_results = [r for i, r in enumerate(results) if test_df.iloc[i].get('has_asr_error', False)]
            asr_clean_results = [r for i, r in enumerate(results) if not test_df.iloc[i].get('has_asr_error', False)]

            if asr_error_results:
                asr_error_acc = sum(1 for r in asr_error_results if r.get('orientation_correct', False)) / len(asr_error_results)
                analysis['asr_error_impact']['error_samples_accuracy'] = asr_error_acc
                analysis['asr_error_impact']['error_samples_count'] = len(asr_error_results)

            if asr_clean_results:
                asr_clean_acc = sum(1 for r in asr_clean_results if r.get('orientation_correct', False)) / len(asr_clean_results)
                analysis['asr_error_impact']['clean_samples_accuracy'] = asr_clean_acc
                analysis['asr_error_impact']['clean_samples_count'] = len(asr_clean_results)

        # Spatial relation performance
        if 'spatial_relations' in test_df.columns:
            spatial_relations = test_df['spatial_relations'].unique()
            for relation in spatial_relations:
                relation_results = [r for i, r in enumerate(results) if test_df.iloc[i]['spatial_relations'] == relation]
                if relation_results:
                    relation_acc = sum(1 for r in relation_results if r.get('orientation_correct', False)) / len(relation_results)
                    analysis['spatial_relation_performance'][relation] = {
                        'accuracy': relation_acc,
                        'count': len(relation_results)
                    }

        return analysis

    def compare_with_training_domain(self, cross_domain_accuracy) -> Dict[str, Any]:
        """Compare cross-domain performance with training domain performance"""
        comparison = {
            'generalization_ratio': 0.0,
            'performance_drop': 0.0,
            'generalization_assessment': ''
        }

        # Try to load training domain performance from step3 results
        training_accuracy = None
        result_files = ['step3_evaluation_results.json', 'a4_evaluation_results.json', 'evaluation_results.json']

        for filename in result_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    training_results = json.load(f)
                    training_accuracy = training_results['metrics']['orientation_accuracy']
                    print(f"Found training domain accuracy in {filename}: {training_accuracy:.3f}")
                    break
            except FileNotFoundError:
                continue

        if training_accuracy is None:
            print("Warning: Training domain results not found for comparison")
            comparison['generalization_assessment'] = 'Cannot compare - training results unavailable'
            return comparison

        comparison['generalization_ratio'] = cross_domain_accuracy / training_accuracy if training_accuracy > 0 else 0.0
        comparison['performance_drop'] = training_accuracy - cross_domain_accuracy
        comparison['training_domain_accuracy'] = training_accuracy
        comparison['cross_domain_accuracy'] = cross_domain_accuracy

        # Assessment
        if comparison['generalization_ratio'] >= 0.90:
            comparison['generalization_assessment'] = 'Excellent generalization'
        elif comparison['generalization_ratio'] >= 0.80:
            comparison['generalization_assessment'] = 'Good generalization'
        elif comparison['generalization_ratio'] >= 0.70:
            comparison['generalization_assessment'] = 'Acceptable generalization'
        else:
            comparison['generalization_assessment'] = 'Poor generalization'

        return comparison

    def run_r2_evaluation(self) -> Dict[str, Any]:
        """Run R2 evaluation: Cross-domain Test Set"""
        print("="*60)
        print("R2 Evaluation: Cross-domain Test Set - Generalization Analysis")
        print("Training Domain: Gongguan MRT Area")
        print("Test Domain: Taipei Station Area")
        print("="*60)

        # Load model and cross-domain test set
        self.load_model()
        test_df = self.load_cross_domain_dataset()

        results = []
        correct_orientations = 0
        format_errors = 0

        print(f"\nEvaluating {len(test_df)} cross-domain test samples...")

        for i, (_, row) in enumerate(test_df.iterrows()):
            print(f"\nSample {i+1}/{len(test_df)}")

            # Get input and expected output
            user_input = row.get('multimodal_input_asr', row.get('multimodal_input_clean', ''))
            expected_orientation = self.extract_expected_orientation(row)
            sample_id = row.get('sample_id', i)

            print(f"Sample ID: {sample_id}")
            print(f"Input: {user_input[:80]}...")
            print(f"Expected: {expected_orientation}")

            try:
                # Generate model response
                predicted_output = self.generate_response(user_input)
                print(f"Predicted: {predicted_output[:150]}...")

                # Extract predicted orientation
                predicted_orientation = self.extract_predicted_orientation(predicted_output)
                print(f"Predicted orientation: {predicted_orientation}")

                # Check format errors
                if predicted_orientation is None:
                    format_errors += 1
                    print("Format error: Cannot extract orientation")

                # Evaluate accuracy
                orientation_correct = (
                    expected_orientation is not None and
                    predicted_orientation is not None and
                    predicted_orientation == expected_orientation
                )

                if orientation_correct:
                    correct_orientations += 1

                print(f"ASR Error: {'Yes' if row.get('has_asr_error', False) else 'No'}")
                print(f"Orientation correct: {'Yes' if orientation_correct else 'No'}")

                # Debug info for first few samples
                if i < 3:
                    print(f"\n=== Debug Sample {i+1} ===")
                    print(f"Full Generated Output:\n{predicted_output}")
                    print("="*50)

                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected_orientation': expected_orientation,
                    'predicted_orientation': predicted_orientation,
                    'predicted_output': predicted_output,
                    'orientation_correct': orientation_correct,
                    'has_asr_error': row.get('has_asr_error', False),
                    'asr_similarity': row.get('asr_similarity', 1.0),
                    'spatial_relation': row.get('spatial_relations', ''),
                    'landmark_count': row.get('landmark_count', 0)
                })

            except Exception as e:
                print(f"Error at sample {i+1}: {e}")
                format_errors += 1
                results.append({
                    'sample_id': sample_id,
                    'input': user_input,
                    'expected_orientation': expected_orientation,
                    'error': str(e),
                    'orientation_correct': False,
                    'has_asr_error': row.get('has_asr_error', False),
                    'spatial_relation': row.get('spatial_relations', '')
                })

        # Basic metrics
        total_samples = len(test_df)
        cross_domain_accuracy = correct_orientations / total_samples
        format_error_rate = format_errors / total_samples

        # Cross-domain analysis
        cross_domain_analysis = self.analyze_cross_domain_performance(results, test_df)

        # Compare with training domain
        domain_comparison = self.compare_with_training_domain(cross_domain_accuracy)

        print("\n" + "="*60)
        print("R2 Evaluation Results - Cross-domain Generalization")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Cross-domain accuracy: {cross_domain_accuracy:.3f} ({correct_orientations}/{total_samples})")
        print(f"Format error rate: {format_error_rate:.3f} ({format_errors}/{total_samples})")

        print(f"\nDomain Transfer Analysis:")
        if 'training_domain_accuracy' in domain_comparison:
            print(f"Training domain accuracy: {domain_comparison['training_domain_accuracy']:.3f}")
            print(f"Cross-domain accuracy: {domain_comparison['cross_domain_accuracy']:.3f}")
            print(f"Generalization ratio: {domain_comparison['generalization_ratio']:.3f}")
            print(f"Performance drop: {domain_comparison['performance_drop']:.3f}")
            print(f"Assessment: {domain_comparison['generalization_assessment']}")

        print(f"\nASR Error Impact in Cross-domain:")
        asr_impact = cross_domain_analysis['asr_error_impact']
        if 'error_samples_accuracy' in asr_impact:
            print(f"ASR error samples accuracy: {asr_impact['error_samples_accuracy']:.3f} ({asr_impact['error_samples_count']} samples)")
        if 'clean_samples_accuracy' in asr_impact:
            print(f"Clean samples accuracy: {asr_impact['clean_samples_accuracy']:.3f} ({asr_impact['clean_samples_count']} samples)")

        print(f"\nSpatial Relation Performance:")
        for relation, stats in cross_domain_analysis['spatial_relation_performance'].items():
            print(f"{relation}: {stats['accuracy']:.3f} ({stats['count']} samples)")

        # Save results
        evaluation_results = {
            'method': 'R2 - Cross-domain Test Set',
            'description': 'Generalization from Gongguan to Taipei Station area',
            'model_path': self.model_path,
            'training_domain': 'Gongguan MRT Area',
            'test_domain': 'Taipei Station Area',
            'test_data_source': self.data_path,
            'metrics': {
                'total_samples': total_samples,
                'cross_domain_accuracy': cross_domain_accuracy,
                'format_error_rate': format_error_rate,
                'correct_orientations': correct_orientations,
                'format_errors': format_errors,
                'cross_domain_analysis': cross_domain_analysis,
                'domain_comparison': domain_comparison
            },
            'detailed_results': results
        }

        # Save to file
        with open('r2_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: r2_evaluation_results.json")

        return evaluation_results

def main():
    """Run R2 evaluation"""
    evaluator = R2EvaluatorCrossDomain(data_path="./TaipeiStation_Generalization.xlsx")
    results = evaluator.run_r2_evaluation()

    print("\n" + "="*60)
    print("FINAL R2 RESULTS FOR PAPER:")
    print(f"R2 Cross-domain Accuracy: {results['metrics']['cross_domain_accuracy']:.3f}")
    if 'generalization_ratio' in results['metrics']['domain_comparison']:
        print(f"R2 Generalization Ratio: {results['metrics']['domain_comparison']['generalization_ratio']:.3f}")
        print(f"R2 Performance Drop: {results['metrics']['domain_comparison']['performance_drop']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
