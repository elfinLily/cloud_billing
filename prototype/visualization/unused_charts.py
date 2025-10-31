# -*- coding: utf-8 -*-
"""
Unused Resources Chart Generator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .chart_styles import (
    set_preview_style, 
    set_paper_style, 
    COLORS, 
    COLOR_PALETTE,
    format_number,
    format_currency
)


class UnusedResourceCharts:
    """
    Unused Resources Chart Generator
    """
    
    def __init__(self, df, output_dir='results/charts'):
        """
        Initialize
        
        Args:
            df: Unused resources DataFrame
            output_dir: Chart output directory
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 UnusedResourceCharts initialized")
        print(f"   Data: {len(self.df):,} records")
        print(f"   Output: {self.output_dir}")
    
    
    def plot_reason_distribution(self, save=True):
        """
        Chart 1: Distribution by Unused Reason
        
        Commitment-Unused vs Zero-Cost-Zero-Usage
        Count + Cost in two side-by-side charts
        """
        print("\n" + "="*80)
        print("📊 Chart 1: Distribution by Unused Reason")
        print("="*80)
        
        # Aggregate
        reason_stats = self.df.groupby('UnusedReason').agg({
            'ResourceId': 'count',
            'WastedCost': 'sum'
        }).reset_index()
        
        reason_stats.columns = ['Reason', 'Count', 'Cost']
        
        # Sort by count (descending, for horizontal bar bottom-to-top)
        reason_stats = reason_stats.sort_values('Count', ascending=True)
        
        # Create chart (2 subplots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
        # === Left: Count ===
        bars1 = ax1.barh(reason_stats['Reason'], 
                         reason_stats['Count'],
                         color=COLORS['primary'],
                         edgecolor='white',
                         linewidth=1.5)
        
        ax1.set_xlabel('Resource Count', fontsize=13, weight='bold')
        ax1.set_title('Distribution by Unused Reason - Count', 
                     fontsize=15, weight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Value labels
        for i, (idx, row) in enumerate(reason_stats.iterrows()):
            count = row['Count']
            pct = count / reason_stats['Count'].sum() * 100
            ax1.text(count + reason_stats['Count'].max() * 0.02, i,
                    f"{format_number(count)} ({pct:.1f}%)",
                    va='center', fontsize=11, weight='bold')
        
        # === Right: Cost ===
        bars2 = ax2.barh(reason_stats['Reason'], 
                         reason_stats['Cost'],
                         color=COLORS['secondary'],
                         edgecolor='white',
                         linewidth=1.5)
        
        ax2.set_xlabel('Wasted Cost (USD/month)', fontsize=13, weight='bold')
        ax2.set_title('Distribution by Unused Reason - Cost', 
                     fontsize=15, weight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Value labels
        for i, (idx, row) in enumerate(reason_stats.iterrows()):
            cost = row['Cost']
            if cost > 0:
                pct = cost / reason_stats['Cost'].sum() * 100
                ax2.text(cost + reason_stats['Cost'].max() * 0.02, i,
                        f"{format_currency(cost)} ({pct:.1f}%)",
                        va='center', fontsize=11, weight='bold')
            else:
                ax2.text(0, i, " $0.00", va='center', fontsize=11)
        
        plt.tight_layout()
        
        # Save
        if save:
            self._save_chart(fig, 'unused_reason_distribution')
        
        plt.show()
        
        print(f"✅ Chart 1 completed")
        return fig
    
    
    def plot_service_distribution(self, top_n=10, save=True):
        """
        Chart 2: Top N Services Distribution
        
        Args:
            top_n: Number of top services to display (default 10)
        """
        print("\n" + "="*80)
        print(f"📊 Chart 2: Top {top_n} Services Distribution")
        print("="*80)
        
        # Aggregate
        service_stats = self.df.groupby('ServiceName').agg({
            'ResourceId': 'count',
            'WastedCost': 'sum'
        }).reset_index()
        
        service_stats.columns = ['Service', 'Count', 'Cost']
        
        # Top N + Others
        service_stats = service_stats.sort_values('Count', ascending=False)
        top_services = service_stats.head(top_n).copy()
        
        if len(service_stats) > top_n:
            others_count = service_stats.iloc[top_n:]['Count'].sum()
            others_cost = service_stats.iloc[top_n:]['Cost'].sum()
            others_row = pd.DataFrame({
                'Service': ['Others'],
                'Count': [others_count],
                'Cost': [others_cost]
            })
            top_services = pd.concat([top_services, others_row], ignore_index=True)
        
        # Reverse sort for chart (bottom to top descending)
        top_services = top_services.sort_values('Count', ascending=True)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Bar chart
        bars = ax.barh(top_services['Service'], 
                       top_services['Count'],
                       color=COLOR_PALETTE[:len(top_services)],
                       edgecolor='white',
                       linewidth=1.5)
        
        ax.set_xlabel('Resource Count', fontsize=13, weight='bold')
        ax.set_ylabel('Service', fontsize=13, weight='bold')
        ax.set_title(f'Top {top_n} Services - Unused Resources', 
                     fontsize=15, weight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Value labels
        for i, (idx, row) in enumerate(top_services.iterrows()):
            count = row['Count']
            pct = count / self.df['ResourceId'].count() * 100
            ax.text(count + top_services['Count'].max() * 0.01, i,
                   f"{format_number(count)} ({pct:.1f}%)",
                   va='center', fontsize=10, weight='bold')
        
        plt.tight_layout()
        
        # Save
        if save:
            self._save_chart(fig, 'unused_service_distribution')
        
        plt.show()
        
        print(f"✅ Chart 2 completed")
        return fig
    
    
    def plot_resource_type_distribution(self, save=True):
        """
        Chart 3: Resource Type Distribution (Donut Chart)
        """
        print("\n" + "="*80)
        print("📊 Chart 3: Resource Type Distribution (Donut)")
        print("="*80)
        
        if 'ResourceType' not in self.df.columns:
            print("❌ ResourceType column not found. Skipping.")
            return None
        
        # Aggregate
        type_stats = self.df['ResourceType'].value_counts()
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Donut chart
        wedges, texts, autotexts = ax.pie(
            type_stats.values,
            labels=type_stats.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=COLOR_PALETTE[:len(type_stats)],
            pctdistance=0.85,
            explode=[0.05] * len(type_stats),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        # Center circle (donut effect)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white', 
                                   edgecolor='#666666', linewidth=2)
        ax.add_artist(centre_circle)
        
        # Center text
        ax.text(0, 0, f"Total\n{format_number(len(self.df))}", 
               ha='center', va='center', fontsize=18, weight='bold')
        
        # Percentage text style
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        # Label text style
        for text in texts:
            text.set_fontsize(12)
            text.set_weight('bold')
        
        ax.set_title('Resource Type Distribution', fontsize=15, weight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        if save:
            self._save_chart(fig, 'unused_resource_type_distribution')
        
        plt.show()
        
        print(f"✅ Chart 3 completed")
        return fig
    
    
    def plot_cost_distribution(self, save=True):
        """
        Chart 4: Cost Distribution (Box plot + Histogram)
        
        Only for Commitment-Unused resources (Cost > 0)
        """
        print("\n" + "="*80)
        print("📊 Chart 4: Cost Distribution (Box plot)")
        print("="*80)
        
        # Commitment-Unused only (cost > 0)
        cost_data = self.df[
            (self.df['UnusedReason'] == 'Commitment-Unused') &
            (self.df['WastedCost'] > 0)
        ].copy()
        
        if len(cost_data) == 0:
            print("❌ No cost data available.")
            return None
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # === Left: Box plot ===
        bp = ax1.boxplot(
            cost_data['WastedCost'],
            vert=True,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=COLORS['primary'], alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.5),
            capprops=dict(color='black', linewidth=1.5)
        )
        
        ax1.set_ylabel('Wasted Cost (USD/month)', fontsize=13, weight='bold')
        ax1.set_title('Cost Distribution (Box plot)', fontsize=15, weight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_xticklabels(['Commitment-Unused'])
        
        # Statistics text
        stats_text = f"""
        Median: {format_currency(cost_data['WastedCost'].median())}
        Mean: {format_currency(cost_data['WastedCost'].mean())}
        Min: {format_currency(cost_data['WastedCost'].min())}
        Max: {format_currency(cost_data['WastedCost'].max())}
        """
        ax1.text(1.3, cost_data['WastedCost'].median(), stats_text,
                fontsize=10, va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # === Right: Histogram ===
        ax2.hist(cost_data['WastedCost'], bins=30, 
                color=COLORS['secondary'], alpha=0.7,
                edgecolor='white', linewidth=1.5)
        
        ax2.set_xlabel('Wasted Cost (USD/month)', fontsize=13, weight='bold')
        ax2.set_ylabel('Resource Count', fontsize=13, weight='bold')
        ax2.set_title('Cost Distribution (Histogram)', fontsize=15, weight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Mean line
        mean_cost = cost_data['WastedCost'].mean()
        ax2.axvline(mean_cost, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {format_currency(mean_cost)}')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save
        if save:
            self._save_chart(fig, 'unused_cost_distribution')
        
        plt.show()
        
        print(f"✅ Chart 4 completed")
        return fig
    
    
    def plot_cloud_comparison(self, save=True):
        """
        Chart 5: GCP vs AWS Comparison (Grouped Bar Chart)
        """
        print("\n" + "="*80)
        print("📊 Chart 5: GCP vs AWS Comparison")
        print("="*80)
        
        if 'ProviderName' not in self.df.columns:
            print("❌ ProviderName column not found. Skipping.")
            return None
        
        # Aggregate
        cloud_stats = self.df.groupby('ProviderName').agg({
            'ResourceId': 'count',
            'WastedCost': 'sum'
        }).reset_index()
        
        cloud_stats.columns = ['Cloud', 'Count', 'TotalCost']
        cloud_stats['AvgCost'] = cloud_stats['TotalCost'] / cloud_stats['Count']
        
        # Create chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Count', 'TotalCost', 'AvgCost']
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
        ylabels = ['Resource Count', 'Total Wasted Cost (USD/month)', 'Avg Cost per Resource (USD)']
        titles = ['Count Comparison', 'Total Cost Comparison', 'Avg Cost Comparison']
        
        for i, (metric, color, ylabel, title) in enumerate(zip(metrics, colors, ylabels, titles)):
            ax = axes[i]
            
            bars = ax.bar(cloud_stats['Cloud'], 
                         cloud_stats[metric],
                         color=color,
                         edgecolor='white',
                         linewidth=1.5,
                         alpha=0.8)
            
            ax.set_ylabel(ylabel, fontsize=12, weight='bold')
            ax.set_title(title, fontsize=14, weight='bold', pad=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Value labels
            for bar, (idx, row) in zip(bars, cloud_stats.iterrows()):
                height = bar.get_height()
                if metric == 'Count':
                    label = format_number(height)
                else:
                    label = format_currency(height)
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom',
                       fontsize=11, weight='bold')
        
        plt.tight_layout()
        
        # Save
        if save:
            self._save_chart(fig, 'unused_cloud_comparison')
        
        plt.show()
        
        print(f"✅ Chart 5 completed")
        return fig
    
    
    def _save_chart(self, fig, filename):
        """
        Save chart in dark mode + PDF versions
        """
        # Dark mode version
        dark_path = self.output_dir / f"{filename}_dark.png"
        fig.savefig(dark_path, dpi=300, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
        print(f"   💾 Saved: {dark_path}")
        
        # PDF version
        pdf_path = self.output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight',
                   format='pdf')
        print(f"   💾 Saved: {pdf_path}")
    
    
    def generate_all_charts(self):
        """
        Generate all charts at once
        """
        print("\n" + "="*80)
        print("🎨 Generating all charts")
        print("="*80)
        
        self.plot_reason_distribution(save=True)
        self.plot_service_distribution(top_n=10, save=True)
        self.plot_resource_type_distribution(save=True)
        self.plot_cost_distribution(save=True)
        self.plot_cloud_comparison(save=True)
        
        print("\n" + "="*80)
        print("✅ All charts generated!")
        print(f"   Output: {self.output_dir}")
        print("="*80)