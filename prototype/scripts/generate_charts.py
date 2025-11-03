# -*- coding: utf-8 -*-
"""
차트만 생성하는 스크립트

이미 저장된 CSV 파일에서 차트만 생성
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualization import UnusedResourceCharts, set_preview_style, set_paper_style


def load_config(config_path='config/focus_config.yaml'):
    """설정 로드"""
    config_file = project_root / config_path
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_charts(self, results):
    """차트 생성 (클라우드별 + 비교)"""
    pattern_config = self.config['analysis']['patterns']
    cloud_config = self.config.get('cloud_filter', {})
    
    print("\n" + "="*100)
    print("🎨 차트 생성 시작")
    print("="*100)
    
    # 스타일 설정
    style = self.viz_config.get('style', 'preview')
    if style == 'paper':
        set_paper_style()
        print(f"   스타일: 논문용 (화이트)")
    else:
        set_preview_style()
        print(f"   스타일: 프리뷰용 (다크)")
    
    # 미사용 리소스 차트
    if pattern_config['unused_resources']['generate_charts']:
        
        if cloud_config.get('enabled', False):
            # GCP 차트
            if 'gcp' in results and len(results['gcp']['unused']) > 0:
                print("\n📊 [GCP] 미사용 리소스 차트 생성 중...")
                gcp_dir = self.chart_output_dir / 'gcp'
                gcp_dir.mkdir(exist_ok=True)
                
                gcp_charts = UnusedResourceCharts(
                    results['gcp']['unused'],
                    output_dir=gcp_dir
                )
                gcp_charts.generate_all_charts()
            
            # AWS 차트
            if 'aws' in results and len(results['aws']['unused']) > 0:
                print("\n📊 [AWS] 미사용 리소스 차트 생성 중...")
                aws_dir = self.chart_output_dir / 'aws'
                aws_dir.mkdir(exist_ok=True)
                
                aws_charts = UnusedResourceCharts(
                    results['aws']['unused'],
                    output_dir=aws_dir
                )
                aws_charts.generate_all_charts()
            
            # 비교 차트 (GCP vs AWS)
            if ('gcp' in results and 'aws' in results and
                len(results['gcp']['unused']) > 0 and 
                len(results['aws']['unused']) > 0):
                print("\n📊 [GCP vs AWS] 비교 차트 생성 중...")
                # TODO: 비교 차트 생성 함수 구현
                print("   ⚠️ 비교 차트는 아직 구현되지 않음")
        
        else:
            # 전체 차트 (기존 방식)
            if 'all' in results and len(results['all']['unused']) > 0:
                print("\n📊 미사용 리소스 차트 생성 중...")
                charts = UnusedResourceCharts(
                    results['all']['unused'],
                    output_dir=self.chart_output_dir
                )
                charts.generate_all_charts()
    
    print("\n" + "="*100)
    print("✅ 차트 생성 완료!")
    print(f"   저장 위치: {self.chart_output_dir}")
    print("="*100)


if __name__ == "__main__":
    generate_charts()