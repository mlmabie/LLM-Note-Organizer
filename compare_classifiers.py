#!/usr/bin/env python3
"""
Comparison utility for note classification methods.

This script analyzes the results from the human-in-the-loop note processing
and compares the performance of different classification methods.
"""

import os
import json
import argparse
import yaml
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Set

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib and scikit-learn not found. Install with 'pip install matplotlib scikit-learn' for visualization.")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Install with 'pip install rich' for better display.")

def load_processed_notes(file_path: str) -> List[Dict[str, Any]]:
    """Load processed notes from JSON file."""
    try:
        with open(file_path, 'r') as f:
            notes = json.load(f)
        return notes
    except Exception as e:
        print(f"Error loading processed notes: {e}")
        return []

def calculate_accuracy(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics for different classification methods."""
    if not notes:
        return {"error": "No notes provided"}
    
    # Count successful classifications
    methods = ["keyword", "llm", "embedding"]
    
    # Initialize counters
    total_notes = 0
    method_correct = {method: 0 for method in methods}
    method_partial = {method: 0 for method in methods}
    method_missing = {method: 0 for method in methods}
    method_extra = {method: 0 for method in methods}
    
    # Category-level metrics
    category_metrics = defaultdict(lambda: {
        method: {"correct": 0, "incorrect": 0, "missed": 0} 
        for method in methods
    })
    
    # Confidence thresholds for testing
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {
        method: {threshold: {"correct": 0, "incorrect": 0, "total": 0} 
                for threshold in thresholds}
        for method in methods
    }
    
    # Process each note
    for note in notes:
        if note.get("status") != "approved" or "approved_categories" not in note:
            continue
        
        total_notes += 1
        approved_categories = set(note["approved_categories"])
        
        for method in methods:
            if method not in note["categories"]:
                continue
                
            # Get predictions for this method
            predictions = note["categories"][method]
            
            # Test different thresholds
            for threshold in thresholds:
                predicted_categories = {category for category, score in predictions if score >= threshold}
                correct = len(predicted_categories.intersection(approved_categories))
                incorrect = len(predicted_categories - approved_categories)
                
                threshold_results[method][threshold]["correct"] += correct
                threshold_results[method][threshold]["incorrect"] += incorrect
                threshold_results[method][threshold]["total"] += len(predicted_categories)
            
            # Use 0.7 as baseline for other metrics
            predicted_categories = {category for category, score in predictions if score >= 0.7}
            
            # Calculate metrics
            correct = predicted_categories.intersection(approved_categories)
            missed = approved_categories - predicted_categories
            extra = predicted_categories - approved_categories
            
            # Update counters
            if predicted_categories == approved_categories:
                method_correct[method] += 1
            elif len(correct) > 0:
                method_partial[method] += 1
            
            method_missing[method] += len(missed)
            method_extra[method] += len(extra)
            
            # Update category-level metrics
            for category in approved_categories:
                if category in predicted_categories:
                    category_metrics[category][method]["correct"] += 1
                else:
                    category_metrics[category][method]["missed"] += 1
            
            for category in predicted_categories - approved_categories:
                category_metrics[category][method]["incorrect"] += 1
    
    # Calculate accuracy percentages
    accuracy = {
        "total_notes": total_notes,
        "exact_match": {method: method_correct[method] / total_notes if total_notes else 0 
                        for method in methods},
        "partial_match": {method: method_partial[method] / total_notes if total_notes else 0 
                         for method in methods},
        "missing_categories": {method: method_missing[method] / (total_notes if total_notes else 1) 
                              for method in methods},
        "extra_categories": {method: method_extra[method] / (total_notes if total_notes else 1) 
                           for method in methods},
        "category_metrics": dict(category_metrics),
        "threshold_results": threshold_results
    }
    
    return accuracy

def calculate_ensemble_performance(notes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate performance metrics for ensemble methods."""
    if not notes:
        return {"error": "No notes provided"}
    
    # Count successful classifications
    methods = ["keyword", "llm", "embedding"]
    ensemble_methods = [
        "majority_vote",
        "weighted_vote",
        "any_method"
    ]
    
    # Initialize counters
    total_notes = 0
    ensemble_correct = {method: 0 for method in ensemble_methods}
    ensemble_partial = {method: 0 for method in ensemble_methods}
    ensemble_missing = {method: 0 for method in ensemble_methods}
    ensemble_extra = {method: 0 for method in ensemble_methods}
    
    # Process each note
    for note in notes:
        if note.get("status") != "approved" or "approved_categories" not in note:
            continue
        
        total_notes += 1
        approved_categories = set(note["approved_categories"])
        
        # Collect predictions from all methods
        method_predictions = {}
        for method in methods:
            if method in note["categories"]:
                # Use 0.7 as threshold
                method_predictions[method] = {
                    category for category, score in note["categories"][method] if score >= 0.7
                }
            else:
                method_predictions[method] = set()
        
        # Majority vote ensemble
        category_votes = defaultdict(int)
        for method, predictions in method_predictions.items():
            for category in predictions:
                category_votes[category] += 1
        
        majority_predictions = {category for category, votes in category_votes.items() if votes >= 2}
        
        # Weighted vote ensemble (weight LLM higher)
        weighted_votes = defaultdict(float)
        weights = {"keyword": 0.8, "llm": 1.5, "embedding": 1.0}
        for method, predictions in method_predictions.items():
            for category in predictions:
                weighted_votes[category] += weights[method]
        
        weighted_predictions = {category for category, score in weighted_votes.items() if score >= 1.5}
        
        # Any method ensemble (union of all methods)
        any_predictions = set()
        for predictions in method_predictions.values():
            any_predictions.update(predictions)
        
        # Calculate metrics for each ensemble
        ensembles = {
            "majority_vote": majority_predictions,
            "weighted_vote": weighted_predictions,
            "any_method": any_predictions
        }
        
        for ensemble_name, predictions in ensembles.items():
            # Calculate metrics
            correct = predictions.intersection(approved_categories)
            missed = approved_categories - predictions
            extra = predictions - approved_categories
            
            # Update counters
            if predictions == approved_categories:
                ensemble_correct[ensemble_name] += 1
            elif len(correct) > 0:
                ensemble_partial[ensemble_name] += 1
            
            ensemble_missing[ensemble_name] += len(missed)
            ensemble_extra[ensemble_name] += len(extra)
    
    # Calculate accuracy percentages
    ensemble_accuracy = {
        "total_notes": total_notes,
        "exact_match": {method: ensemble_correct[method] / total_notes if total_notes else 0 
                       for method in ensemble_methods},
        "partial_match": {method: ensemble_partial[method] / total_notes if total_notes else 0 
                         for method in ensemble_methods},
        "missing_categories": {method: ensemble_missing[method] / (total_notes if total_notes else 1) 
                              for method in ensemble_methods},
        "extra_categories": {method: ensemble_extra[method] / (total_notes if total_notes else 1) 
                           for method in ensemble_methods}
    }
    
    return ensemble_accuracy

def display_results(accuracy: Dict[str, Any], ensemble_accuracy: Dict[str, Any]) -> None:
    """Display classification accuracy results."""
    if RICH_AVAILABLE:
        # Display method accuracy
        console.print("\n[bold cyan]Classification Method Accuracy[/bold cyan]")
        
        table = Table(title=f"Results for {accuracy['total_notes']} notes")
        table.add_column("Method", style="cyan")
        table.add_column("Exact Match", style="green")
        table.add_column("Partial Match", style="yellow")
        table.add_column("Missing", style="red")
        table.add_column("Extra", style="magenta")
        
        methods = ["keyword", "llm", "embedding"]
        for method in methods:
            exact = f"{accuracy['exact_match'][method]:.1%}"
            partial = f"{accuracy['partial_match'][method]:.1%}"
            missing = f"{accuracy['missing_categories'][method]:.1f} per note"
            extra = f"{accuracy['extra_categories'][method]:.1f} per note"
            
            table.add_row(method, exact, partial, missing, extra)
        
        console.print(table)
        
        # Display ensemble accuracy
        console.print("\n[bold cyan]Ensemble Method Accuracy[/bold cyan]")
        
        table = Table(title="Ensemble Results")
        table.add_column("Method", style="cyan")
        table.add_column("Exact Match", style="green")
        table.add_column("Partial Match", style="yellow")
        table.add_column("Missing", style="red")
        table.add_column("Extra", style="magenta")
        
        ensemble_methods = ["majority_vote", "weighted_vote", "any_method"]
        for method in ensemble_methods:
            exact = f"{ensemble_accuracy['exact_match'][method]:.1%}"
            partial = f"{ensemble_accuracy['partial_match'][method]:.1%}"
            missing = f"{ensemble_accuracy['missing_categories'][method]:.1f} per note"
            extra = f"{ensemble_accuracy['extra_categories'][method]:.1f} per note"
            
            table.add_row(method, exact, partial, missing, extra)
        
        console.print(table)
        
        # Display threshold analysis
        console.print("\n[bold cyan]Threshold Analysis[/bold cyan]")
        
        table = Table(title="Precision at Different Thresholds")
        table.add_column("Method", style="cyan")
        table.add_column("Threshold", style="yellow")
        table.add_column("Precision", style="green")
        table.add_column("Categories", style="blue")
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for method in methods:
            for threshold in thresholds:
                results = accuracy["threshold_results"][method][threshold]
                total = results["correct"] + results["incorrect"]
                precision = results["correct"] / total if total else 0
                
                table.add_row(
                    method,
                    f"{threshold:.1f}",
                    f"{precision:.1%}",
                    f"{results['total'] / accuracy['total_notes']:.1f} per note"
                )
        
        console.print(table)
        
    else:
        # Fallback to standard print
        print("\nClassification Method Accuracy")
        print(f"Results for {accuracy['total_notes']} notes")
        print("-" * 80)
        print(f"{'Method':<10} {'Exact Match':<15} {'Partial Match':<15} {'Missing':<15} {'Extra':<15}")
        print("-" * 80)
        
        methods = ["keyword", "llm", "embedding"]
        for method in methods:
            exact = f"{accuracy['exact_match'][method]:.1%}"
            partial = f"{accuracy['partial_match'][method]:.1%}"
            missing = f"{accuracy['missing_categories'][method]:.1f} per note"
            extra = f"{accuracy['extra_categories'][method]:.1f} per note"
            
            print(f"{method:<10} {exact:<15} {partial:<15} {missing:<15} {extra:<15}")
        
        print("\nEnsemble Method Accuracy")
        print("-" * 80)
        print(f"{'Method':<15} {'Exact Match':<15} {'Partial Match':<15} {'Missing':<15} {'Extra':<15}")
        print("-" * 80)
        
        ensemble_methods = ["majority_vote", "weighted_vote", "any_method"]
        for method in ensemble_methods:
            exact = f"{ensemble_accuracy['exact_match'][method]:.1%}"
            partial = f"{ensemble_accuracy['partial_match'][method]:.1%}"
            missing = f"{ensemble_accuracy['missing_categories'][method]:.1f} per note"
            extra = f"{ensemble_accuracy['extra_categories'][method]:.1f} per note"
            
            print(f"{method:<15} {exact:<15} {partial:<15} {missing:<15} {extra:<15}")
        
        print("\nThreshold Analysis")
        print("-" * 80)
        print(f"{'Method':<10} {'Threshold':<10} {'Precision':<10} {'Categories':<15}")
        print("-" * 80)
        
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for method in methods:
            for threshold in thresholds:
                results = accuracy["threshold_results"][method][threshold]
                total = results["correct"] + results["incorrect"]
                precision = results["correct"] / total if total else 0
                
                print(f"{method:<10} {threshold:<10.1f} {precision:<10.1%} {results['total'] / accuracy['total_notes']:<15.1f} per note")

def plot_results(accuracy: Dict[str, Any], ensemble_accuracy: Dict[str, Any], output_dir: str) -> None:
    """Generate plots for accuracy results."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping plots.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    methods = ["keyword", "llm", "embedding"]
    ensemble_methods = ["majority_vote", "weighted_vote", "any_method"]
    
    # 1. Exact and partial match bar chart
    plt.figure(figsize=(12, 6))
    
    # Individual methods
    x = np.arange(len(methods))
    width = 0.35
    
    exact_matches = [accuracy["exact_match"][method] for method in methods]
    partial_matches = [accuracy["partial_match"][method] for method in methods]
    
    plt.bar(x - width/2, exact_matches, width, label='Exact Match')
    plt.bar(x + width/2, partial_matches, width, label='Partial Match')
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Method')
    plt.xticks(x, methods)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, 'method_accuracy.png'), dpi=300, bbox_inches='tight')
    
    # 2. Ensemble methods comparison
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(ensemble_methods))
    
    exact_matches = [ensemble_accuracy["exact_match"][method] for method in ensemble_methods]
    partial_matches = [ensemble_accuracy["partial_match"][method] for method in ensemble_methods]
    
    plt.bar(x - width/2, exact_matches, width, label='Exact Match')
    plt.bar(x + width/2, partial_matches, width, label='Partial Match')
    
    plt.xlabel('Ensemble Method')
    plt.ylabel('Accuracy')
    plt.title('Ensemble Method Accuracy')
    plt.xticks(x, [m.replace('_', ' ').title() for m in ensemble_methods])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, 'ensemble_accuracy.png'), dpi=300, bbox_inches='tight')
    
    # 3. Threshold precision curve
    plt.figure(figsize=(12, 6))
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for method in methods:
        precision_values = []
        for threshold in thresholds:
            results = accuracy["threshold_results"][method][threshold]
            total = results["correct"] + results["incorrect"]
            precision = results["correct"] / total if total else 0
            precision_values.append(precision)
        
        plt.plot(thresholds, precision_values, marker='o', linewidth=2, label=method)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision at Different Confidence Thresholds')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'threshold_precision.png'), dpi=300, bbox_inches='tight')
    
    # 4. Category-level accuracy heatmap
    # Get top 20 categories by frequency
    category_freq = defaultdict(int)
    for category, metrics in accuracy["category_metrics"].items():
        for method in methods:
            category_freq[category] += metrics[method]["correct"] + metrics[method]["missed"]
    
    top_categories = sorted(category_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    top_category_names = [c[0] for c in top_categories]
    
    if top_category_names:  # Only proceed if we have categories
        plt.figure(figsize=(14, 10))
        
        data = np.zeros((len(top_category_names), len(methods)))
        
        for i, category in enumerate(top_category_names):
            for j, method in enumerate(methods):
                metrics = accuracy["category_metrics"][category][method]
                total = metrics["correct"] + metrics["missed"]
                accuracy_value = metrics["correct"] / total if total else 0
                data[i, j] = accuracy_value
        
        plt.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        
        plt.yticks(np.arange(len(top_category_names)), top_category_names)
        plt.xticks(np.arange(len(methods)), methods)
        
        plt.title('Category-Level Accuracy by Method')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_accuracy.png'), dpi=300, bbox_inches='tight')
    
    print(f"Plots saved to {output_dir}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare note classification methods.")
    parser.add_argument("--input", type=str, default="processed_notes.json", help="Input file with processed notes")
    parser.add_argument("--plot-dir", type=str, default="./plots", help="Directory for saving plots")
    args = parser.parse_args()
    
    # Load processed notes
    notes = load_processed_notes(args.input)
    if not notes:
        print(f"No notes found in {args.input}")
        return
    
    print(f"Loaded {len(notes)} processed notes from {args.input}")
    
    # Calculate accuracy
    accuracy = calculate_accuracy(notes)
    ensemble_accuracy = calculate_ensemble_performance(notes)
    
    # Display results
    display_results(accuracy, ensemble_accuracy)
    
    # Generate plots
    plot_results(accuracy, ensemble_accuracy, args.plot_dir)
    
    # Save analysis results
    output_file = "classifier_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "ensemble_accuracy": ensemble_accuracy
        }, f, indent=2)
    
    print(f"Analysis results saved to {output_file}")

if __name__ == "__main__":
    main()