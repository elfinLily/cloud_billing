# -*- coding: utf-8 -*-
"""
Transfer Learning 결과 시각화

GCP 학습 패턴 및 AWS 사용률 추정 결과를 시각화합니다.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization.chart_styles import set_preview_style, COLORS, COLOR_PALETTE


class TransferLearningVisualizer:
    """
    Transfer Learning 결과 시각화 클래스
    
    주요 차트:
    1. 서비스 매칭 현황 (Exact Match vs Global Average)
    2. 신뢰도 분포
    3. 추정 CPU/Memory 사용률 분포
    4. GCP vs AWS 서비스별 비교
    5. 과다 프로비저닝 탐지 결과
    """
    
    def __init__(self, output_dir='results/charts/transfer_learning'):
        """
        초기화
        
        Args:
            output_dir: 차트 저장 경로
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 스타일 설정 (다크모드)
        set_preview_style()
        
        print(f"📊 TransferLearningVisualizer 초기화")
        print(f"   출력 경로: {self.output_dir}")
    
    
    def plot_matching_status(self, df_estimation, save=True):
        """
        차트 1: 서비스 매칭 현황
        
        Exact Match vs Global Average 비율
        
        Args:
            df_estimation: estimate_batch() 결과 DataFrame
            save: 저장 여부
        
        Returns:
            matplotlib.figure.Figure
        """
        print("\n" + "="*80)
        print("📊 차트 1: 서비스 매칭 현황")
        print("="*80)
        
        # 매칭 방법별 집계
        method_counts = df_estimation['method'].value_counts()
        
        # 색상 설정 (다크모드용)
        colors = [COLORS['success'], COLORS['warning']]
        
        # 차트 생성 (2개 서브플롯)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # === 좌측: 파이 차트 ===
        wedges, texts, autotexts = ax1.pie(
            method_counts.values,
            labels=['Exact Match', 'Global Average'],
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(method_counts)],
            explode=[0.05] * len(method_counts),
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        # 텍스트 스타일
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_weight('bold')
        
        for text in texts:
            text.set_fontsize(11)
            text.set_color('#CCCCCC')
        
        ax1.set_title('Service Matching Method Distribution', 
                     fontsize=14, weight='bold', pad=15, color='#FFFFFF')
        
        # === 우측: 바 차트 ===
        bars = ax2.bar(
            method_counts.index,
            method_counts.values,
            color=colors[:len(method_counts)],
            edgecolor='white',
            linewidth=1.5
        )
        
        ax2.set_xlabel('Matching Method', fontsize=12, weight='bold', color='#CCCCCC')
        ax2.set_ylabel('Service Count', fontsize=12, weight='bold', color='#CCCCCC')
        ax2.set_title('Service Count by Matching Method', 
                     fontsize=14, weight='bold', pad=15, color='#FFFFFF')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 값 레이블
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom',
                    fontsize=11, weight='bold', color='#FFFFFF')
        
        plt.tight_layout()
        
        # 저장
        if save:
            self._save_chart(fig, 'matching_status')
        
        plt.show()
        
        print(f"✅ 차트 1 완료")
        return fig
    
    
    def plot_confidence_distribution(self, df_estimation, save=True):
        """
        차트 2: 신뢰도 분포
        
        Args:
            df_estimation: estimate_batch() 결과 DataFrame
            save: 저장 여부
        
        Returns:
            matplotlib.figure.Figure
        """
        print("\n" + "="*80)
        print("📊 차트 2: 신뢰도 분포")
        print("="*80)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # === 좌측: 히스토그램 ===
        ax1.hist(
            df_estimation['confidence'],
            bins=20,
            color=COLORS['primary'],
            edgecolor='white',
            linewidth=1.5,
            alpha=0.8
        )
        
        # 평균선
        mean_conf = df_estimation['confidence'].mean()
        ax1.axvline(mean_conf, color=COLORS['danger'], linestyle='--', linewidth=2,
                   label=f'Mean: {mean_conf:.2f}')
        
        ax1.set_xlabel('Confidence Score', fontsize=12, weight='bold', color='#CCCCCC')
        ax1.set_ylabel('Frequency', fontsize=12, weight='bold', color='#CCCCCC')
        ax1.set_title('Confidence Score Distribution', 
                     fontsize=14, weight='bold', pad=15, color='#FFFFFF')
        ax1.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # === 우측: 신뢰도 구간별 바 차트 ===
        # 신뢰도 구간 분류
        bins = [0, 0.3, 0.5, 0.8, 1.0]
        labels = ['Low\n(0-0.3)', 'Medium\n(0.3-0.5)', 'High\n(0.5-0.8)', 'Very High\n(0.8-1.0)']
        df_estimation['conf_category'] = pd.cut(
            df_estimation['confidence'], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
        
        category_counts = df_estimation['conf_category'].value_counts().sort_index()
        
        bar_colors = [COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['success']]
        
        bars = ax2.bar(
            category_counts.index,
            category_counts.values,
            color=bar_colors[:len(category_counts)],
            edgecolor='white',
            linewidth=1.5
        )
        
        ax2.set_xlabel('Confidence Level', fontsize=12, weight='bold', color='#CCCCCC')
        ax2.set_ylabel('Service Count', fontsize=12, weight='bold', color='#CCCCCC')
        ax2.set_title('Services by Confidence Level', 
                     fontsize=14, weight='bold', pad=15, color='#FFFFFF')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 값 레이블
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom',
                    fontsize=10, weight='bold', color='#FFFFFF')
        
        plt.tight_layout()
        
        # 저장
        if save:
            self._save_chart(fig, 'confidence_distribution')
        
        plt.show()
        
        print(f"✅ 차트 2 완료")
        return fig
    
    
    def plot_usage_estimation(self, df_estimation, save=True):
        """
        차트 3: 추정된 CPU/Memory 사용률 분포
        
        Args:
            df_estimation: estimate_batch() 결과 DataFrame
            save: 저장 여부
        
        Returns:
            matplotlib.figure.Figure
        """
        print("\n" + "="*80)
        print("📊 차트 3: 추정 CPU/Memory 사용률 분포")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # === 좌상단: CPU 히스토그램 ===
        if 'cpu_mean' in df_estimation.columns:
            ax = axes[0, 0]
            cpu_data = df_estimation['cpu_mean'].dropna() * 100
            
            ax.hist(cpu_data, bins=25, color=COLORS['primary'], 
                   edgecolor='white', linewidth=1.5, alpha=0.8)
            
            mean_cpu = cpu_data.mean()
            ax.axvline(mean_cpu, color=COLORS['danger'], linestyle='--', linewidth=2,
                      label=f'Mean: {mean_cpu:.1f}%')
            
            # 임계값 표시 (30%)
            ax.axvline(30, color=COLORS['warning'], linestyle=':', linewidth=2,
                      label='Threshold: 30%')
            
            ax.set_xlabel('CPU Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_ylabel('Frequency', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('Estimated CPU Usage Distribution', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # === 우상단: Memory 히스토그램 ===
        if 'memory_mean' in df_estimation.columns:
            ax = axes[0, 1]
            mem_data = df_estimation['memory_mean'].dropna() * 100
            
            ax.hist(mem_data, bins=25, color=COLORS['secondary'], 
                   edgecolor='white', linewidth=1.5, alpha=0.8)
            
            mean_mem = mem_data.mean()
            ax.axvline(mean_mem, color=COLORS['danger'], linestyle='--', linewidth=2,
                      label=f'Mean: {mean_mem:.1f}%')
            
            # 임계값 표시 (30%)
            ax.axvline(30, color=COLORS['warning'], linestyle=':', linewidth=2,
                      label='Threshold: 30%')
            
            ax.set_xlabel('Memory Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_ylabel('Frequency', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('Estimated Memory Usage Distribution', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # === 좌하단: CPU vs Memory 산점도 ===
        ax = axes[1, 0]
        if 'cpu_mean' in df_estimation.columns and 'memory_mean' in df_estimation.columns:
            scatter = ax.scatter(
                df_estimation['cpu_mean'] * 100,
                df_estimation['memory_mean'] * 100,
                c=df_estimation['confidence'],
                cmap='viridis',
                alpha=0.6,
                s=50,
                edgecolor='white',
                linewidth=0.5
            )
            
            # 임계값 영역 표시
            ax.axvline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
            
            # 과다 프로비저닝 영역 표시
            ax.fill_between([0, 30], 0, 30, color=COLORS['danger'], alpha=0.15)
            ax.text(15, 15, 'Over-\nProvisioned', ha='center', va='center', 
                   fontsize=10, color=COLORS['danger'], weight='bold')
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Confidence', fontsize=11, color='#CCCCCC')
            cbar.ax.yaxis.set_tick_params(color='#CCCCCC')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#CCCCCC')
            
            ax.set_xlabel('CPU Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_ylabel('Memory Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('CPU vs Memory Usage (colored by Confidence)', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
        
        # === 우하단: 박스플롯 ===
        ax = axes[1, 1]
        if 'cpu_mean' in df_estimation.columns and 'memory_mean' in df_estimation.columns:
            data_to_plot = [
                df_estimation['cpu_mean'].dropna() * 100,
                df_estimation['memory_mean'].dropna() * 100
            ]
            
            bp = ax.boxplot(
                data_to_plot,
                labels=['CPU', 'Memory'],
                patch_artist=True,
                widths=0.5
            )
            
            colors_box = [COLORS['primary'], COLORS['secondary']]
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(2)
            
            # 임계값 표시
            ax.axhline(30, color=COLORS['danger'], linestyle='--', linewidth=2,
                      label='Threshold: 30%')
            
            ax.set_ylabel('Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('CPU vs Memory Usage Comparison', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 저장
        if save:
            self._save_chart(fig, 'usage_estimation')
        
        plt.show()
        
        print(f"✅ 차트 3 완료")
        return fig
    
    
    def plot_overprovisioning_summary(self, df_overprovisioned, df_total, save=True):
        """
        차트 4: 과다 프로비저닝 탐지 결과 요약
        
        Args:
            df_overprovisioned: 과다 프로비저닝 DataFrame
            df_total: 전체 DataFrame
            save: 저장 여부
        
        Returns:
            matplotlib.figure.Figure
        """
        print("\n" + "="*80)
        print("📊 차트 4: 과다 프로비저닝 탐지 결과")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # === 좌상단: 탐지율 파이 차트 ===
        ax = axes[0, 0]
        
        normal_count = len(df_total) - len(df_overprovisioned)
        over_count = len(df_overprovisioned)
        
        sizes = [normal_count, over_count]
        labels = ['Normal', 'Over-Provisioned']
        colors = [COLORS['success'], COLORS['danger']]
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0, 0.1],
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_weight('bold')
        
        for text in texts:
            text.set_fontsize(11)
            text.set_color('#CCCCCC')
        
        ax.set_title(f'Detection Result\n(Total: {len(df_total):,} records)', 
                    fontsize=13, weight='bold', pad=15, color='#FFFFFF')
        
        # === 우상단: 서비스별 Top 10 ===
        ax = axes[0, 1]
        
        if 'ServiceName' in df_overprovisioned.columns and len(df_overprovisioned) > 0:
            service_counts = df_overprovisioned['ServiceName'].value_counts().head(10)
            
            bars = ax.barh(
                range(len(service_counts)),
                service_counts.values,
                color=COLORS['danger'],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.8
            )
            
            ax.set_yticks(range(len(service_counts)))
            ax.set_yticklabels([s[:35] + '...' if len(s) > 35 else s 
                               for s in service_counts.index],
                              fontsize=9, color='#CCCCCC')
            
            ax.set_xlabel('Record Count', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('Top 10 Over-Provisioned Services', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # 값 레이블
            for i, (bar, count) in enumerate(zip(bars, service_counts.values)):
                ax.text(bar.get_width() + max(service_counts.values) * 0.02, i,
                       f'{count:,}', va='center', fontsize=9, color='#FFFFFF')
        
        # === 좌하단: 신뢰도별 분포 ===
        ax = axes[1, 0]
        
        if 'confidence' in df_overprovisioned.columns and len(df_overprovisioned) > 0:
            # Exact Match vs Global Average
            exact_match = df_overprovisioned[df_overprovisioned['method'] == 'exact_match']
            global_avg = df_overprovisioned[df_overprovisioned['method'] == 'global_average']
            
            x = np.arange(2)
            width = 0.35
            
            bars1 = ax.bar(x - width/2, [len(exact_match), 0], width, 
                          label='Exact Match', color=COLORS['success'],
                          edgecolor='white', linewidth=1.5)
            bars2 = ax.bar(x + width/2, [0, len(global_avg)], width,
                          label='Global Average', color=COLORS['warning'],
                          edgecolor='white', linewidth=1.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Exact Match', 'Global Average'], 
                              fontsize=11, color='#CCCCCC')
            ax.set_ylabel('Count', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('Over-Provisioned by Matching Method', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 값 레이블
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height):,}', ha='center', va='bottom',
                               fontsize=10, weight='bold', color='#FFFFFF')
        
        # === 우하단: 낭비율 히스토그램 ===
        ax = axes[1, 1]
        
        if 'CPUWastePercent' in df_overprovisioned.columns and len(df_overprovisioned) > 0:
            ax.hist(df_overprovisioned['CPUWastePercent'], bins=20, 
                   color=COLORS['primary'], alpha=0.7, label='CPU Waste %',
                   edgecolor='white', linewidth=1)
            ax.hist(df_overprovisioned['MemoryWastePercent'], bins=20, 
                   color=COLORS['secondary'], alpha=0.7, label='Memory Waste %',
                   edgecolor='white', linewidth=1)
            
            ax.set_xlabel('Waste Percentage (%)', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_ylabel('Frequency', fontsize=12, weight='bold', color='#CCCCCC')
            ax.set_title('Resource Waste Distribution', 
                        fontsize=13, weight='bold', pad=10, color='#FFFFFF')
            ax.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 저장
        if save:
            self._save_chart(fig, 'overprovisioning_summary')
        
        plt.show()
        
        print(f"✅ 차트 4 완료")
        return fig
    
    
    def plot_gcp_patterns_summary(self, gcp_patterns, save=True):
        """
        차트 5: GCP 학습 패턴 요약
        
        Args:
            gcp_patterns: GCPPatternLearner 학습 결과 (dict)
            save: 저장 여부
        
        Returns:
            matplotlib.figure.Figure
        """
        print("\n" + "="*80)
        print("📊 차트 5: GCP 학습 패턴 요약")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 데이터 추출
        services = list(gcp_patterns.keys())
        cpu_means = [gcp_patterns[s].get('cpu', {}).get('mean', 0) * 100 for s in services]
        mem_means = [gcp_patterns[s].get('memory', {}).get('mean', 0) * 100 for s in services]
        sample_counts = [gcp_patterns[s].get('sample_count', 0) for s in services]
        
        # === 좌상단: 서비스별 CPU 사용률 Top 15 ===
        ax = axes[0, 0]
        
        # CPU 기준 정렬
        sorted_idx = np.argsort(cpu_means)[::-1][:15]
        top_services = [services[i][:25] for i in sorted_idx]
        top_cpu = [cpu_means[i] for i in sorted_idx]
        
        bars = ax.barh(range(len(top_services)), top_cpu, 
                      color=COLORS['primary'], edgecolor='white', linewidth=1)
        ax.set_yticks(range(len(top_services)))
        ax.set_yticklabels(top_services, fontsize=9, color='#CCCCCC')
        ax.set_xlabel('CPU Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_title('Top 15 Services by CPU Usage (GCP)', 
                    fontsize=13, weight='bold', pad=10, color='#FFFFFF')
        ax.axvline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # === 우상단: 서비스별 Memory 사용률 Top 15 ===
        ax = axes[0, 1]
        
        sorted_idx = np.argsort(mem_means)[::-1][:15]
        top_services = [services[i][:25] for i in sorted_idx]
        top_mem = [mem_means[i] for i in sorted_idx]
        
        bars = ax.barh(range(len(top_services)), top_mem, 
                      color=COLORS['secondary'], edgecolor='white', linewidth=1)
        ax.set_yticks(range(len(top_services)))
        ax.set_yticklabels(top_services, fontsize=9, color='#CCCCCC')
        ax.set_xlabel('Memory Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_title('Top 15 Services by Memory Usage (GCP)', 
                    fontsize=13, weight='bold', pad=10, color='#FFFFFF')
        ax.axvline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # === 좌하단: CPU vs Memory 산점도 ===
        ax = axes[1, 0]
        
        scatter = ax.scatter(cpu_means, mem_means, 
                            c=np.log1p(sample_counts), cmap='plasma',
                            s=50, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        ax.axvline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(30, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Log(Sample Count)', fontsize=11, color='#CCCCCC')
        cbar.ax.yaxis.set_tick_params(color='#CCCCCC')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#CCCCCC')
        
        ax.set_xlabel('CPU Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_ylabel('Memory Usage (%)', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_title('CPU vs Memory by Service (GCP Learned Patterns)', 
                    fontsize=13, weight='bold', pad=10, color='#FFFFFF')
        ax.grid(alpha=0.3, linestyle='--')
        
        # === 우하단: 샘플 수 분포 ===
        ax = axes[1, 1]
        
        ax.hist(sample_counts, bins=30, color=COLORS['info'], 
               edgecolor='white', linewidth=1, alpha=0.8)
        
        mean_samples = np.mean(sample_counts)
        ax.axvline(mean_samples, color=COLORS['danger'], linestyle='--', linewidth=2,
                  label=f'Mean: {mean_samples:.0f}')
        
        ax.set_xlabel('Sample Count', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_ylabel('Service Count', fontsize=12, weight='bold', color='#CCCCCC')
        ax.set_title('Sample Count Distribution per Service', 
                    fontsize=13, weight='bold', pad=10, color='#FFFFFF')
        ax.legend(facecolor='#2e2e2e', edgecolor='#666666', labelcolor='#CCCCCC')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # 저장
        if save:
            self._save_chart(fig, 'gcp_patterns_summary')
        
        plt.show()
        
        print(f"✅ 차트 5 완료")
        return fig
    
    
    def _save_chart(self, fig, filename):
        """
        차트 저장 (PNG + PDF)
        
        Args:
            fig: matplotlib Figure
            filename: 파일명 (확장자 제외)
        """
        # PNG 저장
        png_path = self.output_dir / f"{filename}.png"
        fig.savefig(png_path, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        print(f"   💾 저장: {png_path}")
        
        # PDF 저장
        pdf_path = self.output_dir / f"{filename}.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight',
                   format='pdf')
        print(f"   💾 저장: {pdf_path}")
    
    
    def generate_all_charts(self, df_estimation, df_overprovisioned, gcp_patterns):
        """
        모든 차트 생성
        
        Args:
            df_estimation: 추정 결과 DataFrame
            df_overprovisioned: 과다 프로비저닝 DataFrame
            gcp_patterns: GCP 학습 패턴 dict
        """
        print("\n" + "="*100)
        print("🎨 모든 차트 생성 시작")
        print("="*100)
        
        self.plot_matching_status(df_estimation)
        self.plot_confidence_distribution(df_estimation)
        self.plot_usage_estimation(df_estimation)
        self.plot_overprovisioning_summary(df_overprovisioned, df_estimation)
        self.plot_gcp_patterns_summary(gcp_patterns)
        
        print("\n" + "="*100)
        print("✅ 모든 차트 생성 완료!")
        print(f"   출력 경로: {self.output_dir}")
        print("="*100)


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    print("\n🚀 Transfer Learning 시각화 테스트")
    print("="*100)
    
    # 테스트용 더미 데이터 생성
    print("📝 테스트 데이터 생성 중...")
    
    # 더미 추정 결과
    np.random.seed(42)
    n_samples = 100
    
    df_estimation = pd.DataFrame({
        'aws_service': [f'Service_{i}' for i in range(n_samples)],
        'matched_gcp_service': [f'GCP_Service_{i%20}' for i in range(n_samples)],
        'method': np.random.choice(['exact_match', 'global_average'], n_samples, p=[0.7, 0.3]),
        'confidence': np.random.uniform(0.3, 1.0, n_samples),
        'cpu_mean': np.random.uniform(0.1, 0.9, n_samples),
        'memory_mean': np.random.uniform(0.15, 0.85, n_samples),
    })
    
    # 과다 프로비저닝 (30% 이하)
    df_overprovisioned = df_estimation[
        (df_estimation['cpu_mean'] < 0.3) | (df_estimation['memory_mean'] < 0.3)
    ].copy()
    df_overprovisioned['CPUWastePercent'] = (1 - df_overprovisioned['cpu_mean']) * 100
    df_overprovisioned['MemoryWastePercent'] = (1 - df_overprovisioned['memory_mean']) * 100
    df_overprovisioned['ServiceName'] = df_overprovisioned['aws_service']
    
    # 더미 GCP 패턴
    gcp_patterns = {
        f'GCP_Service_{i}': {
            'service_name': f'GCP_Service_{i}',
            'sample_count': np.random.randint(100, 10000),
            'cpu': {
                'mean': np.random.uniform(0.2, 0.8),
                'std': np.random.uniform(0.05, 0.2),
                'median': np.random.uniform(0.2, 0.8),
                'min': np.random.uniform(0.05, 0.2),
                'max': np.random.uniform(0.8, 1.0)
            },
            'memory': {
                'mean': np.random.uniform(0.25, 0.75),
                'std': np.random.uniform(0.05, 0.15),
                'median': np.random.uniform(0.25, 0.75),
                'min': np.random.uniform(0.1, 0.3),
                'max': np.random.uniform(0.7, 0.95)
            }
        }
        for i in range(20)
    }
    
    print(f"   추정 데이터: {len(df_estimation)}건")
    print(f"   과다 프로비저닝: {len(df_overprovisioned)}건")
    print(f"   GCP 패턴: {len(gcp_patterns)}개 서비스")
    
    # 시각화
    visualizer = TransferLearningVisualizer()
    visualizer.generate_all_charts(df_estimation, df_overprovisioned, gcp_patterns)
    
    print("\n✅ 테스트 완료!")