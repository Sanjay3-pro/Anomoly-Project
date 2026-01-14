"""
Visualization utilities for anomaly detection results
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List
import os

class TimeSeriesPlotter:
    """Handles visualization of time-series anomaly detection results"""
    
    def __init__(self, style: str = "default", figsize: tuple = (14, 6)):
        """
        Initialize plotter
        
        Args:
            style: Matplotlib style
            figsize: Figure size for matplotlib plots
        """
        self.style = style
        self.figsize = figsize
        if style != "default":
            plt.style.use(style)
    
    def plot_anomalies_matplotlib(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
        scores: Optional[np.ndarray] = None,
        title: str = "Time Series Anomaly Detection",
        savepath: Optional[str] = None
    ) -> None:
        """
        Plot time-series with anomalies using Matplotlib
        
        Args:
            data: Time-series data
            predictions: Binary predictions (1 = anomaly, 0 = normal)
            scores: Anomaly scores (optional)
            title: Plot title
            savepath: Path to save figure
        """
        fig, axes = plt.subplots(2 if scores is not None else 1, 1, figsize=self.figsize)
        
        if scores is None:
            axes = [axes]
        
        # Plot 1: Time series with anomalies highlighted
        ax = axes[0]
        x = np.arange(len(data))
        normal_mask = predictions == 0
        anomaly_mask = predictions == 1
        
        ax.plot(x[normal_mask], data[normal_mask], 'b-', label='Normal', linewidth=1.5)
        ax.scatter(x[anomaly_mask], data[anomaly_mask], color='red', s=50, 
                  label='Anomaly', zorder=5, marker='X')
        
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{title} - Data with Anomalies', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores
        if scores is not None:
            ax = axes[1]
            ax.bar(x, scores, color=['red' if pred == 1 else 'blue' for pred in predictions],
                  alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Time Index', fontsize=12)
            ax.set_ylabel('Anomaly Score', fontsize=12)
            ax.set_title(f'{title} - Anomaly Scores', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if savepath:
            os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {savepath}")
        
        plt.show()
    
    def plot_anomalies_plotly(
        self,
        data: np.ndarray,
        predictions: np.ndarray,
        scores: Optional[np.ndarray] = None,
        title: str = "Time Series Anomaly Detection",
        savepath: Optional[str] = None
    ) -> None:
        """
        Plot time-series with anomalies using Plotly (interactive)
        
        Args:
            data: Time-series data
            predictions: Binary predictions
            scores: Anomaly scores (optional)
            title: Plot title
            savepath: Path to save HTML
        """
        x = np.arange(len(data))
        normal_mask = predictions == 0
        anomaly_mask = predictions == 1
        
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter(
            x=x[normal_mask],
            y=data[normal_mask],
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>'
        ))
        
        # Anomalies
        fig.add_trace(go.Scatter(
            x=x[anomaly_mask],
            y=data[anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x'),
            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{title} - Data with Anomalies',
            xaxis_title='Time Index',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        if savepath:
            os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
            fig.write_html(savepath)
            print(f"Interactive plot saved to {savepath}")
        
        fig.show()
    
    def plot_comparison(
        self,
        data: np.ndarray,
        results_dict: dict,
        title: str = "Anomaly Detection Comparison",
        savepath: Optional[str] = None
    ) -> None:
        """
        Compare results from multiple detectors
        
        Args:
            data: Time-series data
            results_dict: Dict with detector names as keys and predictions as values
            title: Plot title
            savepath: Path to save figure
        """
        n_detectors = len(results_dict)
        fig, axes = plt.subplots(n_detectors + 1, 1, figsize=(14, 3 * (n_detectors + 1)))
        
        if n_detectors == 1:
            axes = [axes]
        else:
            axes = list(axes)
        
        x = np.arange(len(data))
        
        # Plot original data
        ax = axes[0]
        ax.plot(x, data, 'b-', linewidth=1.5)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{title} - Original Data', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot each detector's results
        for idx, (name, predictions) in enumerate(results_dict.items()):
            ax = axes[idx + 1]
            colors = ['red' if pred == 1 else 'blue' for pred in predictions]
            ax.scatter(x, data, c=colors, s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'{name} - Anomalies: {np.sum(predictions)}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Index', fontsize=12)
        plt.tight_layout()
        
        if savepath:
            os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {savepath}")
        
        plt.show()
    
    def plot_statistics(self, results: dict, savepath: Optional[str] = None) -> None:
        """
        Plot statistics about anomaly detection results
        
        Args:
            results: Results dictionary from detector
            savepath: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Anomaly rate
        ax = axes[0]
        ax.bar(['Anomalies', 'Normal'], 
              [results['anomaly_count'], 
               len(results['predictions']) - results['anomaly_count']],
              color=['red', 'blue'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Anomaly Distribution', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Score distribution
        ax = axes[1]
        ax.hist(results['scores'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(results['threshold'], color='red', linestyle='--', linewidth=2, label=f"Threshold: {results['threshold']:.2f}")
        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Score Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if savepath:
            os.makedirs(os.path.dirname(savepath) or '.', exist_ok=True)
            plt.savefig(savepath, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {savepath}")
        
        plt.show()
