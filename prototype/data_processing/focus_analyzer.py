# -*- coding: utf-8 -*-
"""
FOCUS 데이터 메인 분석기
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yaml
import pandas as pd
from pathlib import Path
import logging

from focus_loader import FocusDataLoader
from focus_patterns import OverProvisioningDetector, UnusedResourceDetector

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from visualization import UnusedResourceCharts, set_preview_style, set_paper_style

class FocusAnalyzer:
    """FOCUS 메인 분석기"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """초기화"""
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 로거 설정
        self._setup_logger()
        
        # 데이터 로더
        self.loader = FocusDataLoader(config_path)
        self.df = None
        
        # 결과 저장 경로
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 시각화 설정
        self.viz_config = self.config.get('visualization', {})
        self.chart_output_dir = Path(self.viz_config.get('output_dir', 'results/charts'))
        self.chart_output_dir.mkdir(parents=True, exist_ok=True)
    
    
    def _setup_logger(self):
        """로거 설정"""
        log_config = self.config['logging']
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    
    def load_data(self, use_sample=False):
        """데이터 로드"""
        self.df = self.loader.load(use_sample=use_sample)
        return self.df
    
    
    def analyze_patterns(self):
        """2가지 패턴 분석"""
        if self.df is None:
            raise ValueError("데이터를 먼저 로드하세요: load_data()")
        
        results = {}
        pattern_config = self.config['analysis']['patterns']

        # 패턴 1: 과다 프로비저닝
        if pattern_config['over_provisioning']['enabled']:
            print("\n🔍 패턴 1: 과다 프로비저닝 분석 실행")
            detector1 = OverProvisioningDetector(self.df, self.config)
            results['over_provisioned'] = detector1.detect()
        else:
            print("\n⏭️  패턴 1: 과다 프로비저닝 분석 스킵 (config에서 비활성화됨)")
            results['over_provisioned'] = pd.DataFrame()

        # 패턴 2: 미사용 리소스
        if pattern_config['unused_resources']['enabled']:
            print("\n🔍 패턴 2: 미사용 리소스 분석 실행")
            detector2 = UnusedResourceDetector(self.df, self.config)
            results['unused'] = detector2.detect()
        else:
            print("\n⏭️  패턴 2: 미사용 리소스 분석 스킵 (config에서 비활성화됨)")
            results['unused'] = pd.DataFrame()

        return results
    
    def generate_charts(self, results):
        """
        차트 생성

        Args:
            results: analyze_patterns() 결과 딕셔너리
        """
        pattern_config = self.config['analysis']['patterns']

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
            if len(results['unused']) > 0:
                print("\n📊 미사용 리소스 차트 생성 중...")
                
                unused_charts = UnusedResourceCharts(
                    results['unused'],
                    output_dir=self.chart_output_dir
                )
                
                unused_charts.generate_all_charts()
            else:
                print("\n⚠️ 미사용 리소스 데이터가 없어 차트를 생성하지 않습니다.")
        else:
            print("\n⏭️  미사용 리소스 차트 생성 스킵 (config에서 비활성화됨)")
        
        # 과다 프로비저닝 차트
        if pattern_config['over_provisioning']['generate_charts']:
            if len(results['over_provisioned']) > 0:
                print("\n📊 과다 프로비저닝 차트 생성 중...")
                # TODO: 구현 예정
                print("   ⚠️ 아직 구현되지 않음")
            else:
                print("\n⚠️ 과다 프로비저닝 데이터가 없어 차트를 생성하지 않습니다.")
        else:
            print("\n⏭️  과다 프로비저닝 차트 생성 스킵 (config에서 비활성화됨)")
        
        print("\n" + "="*100)
        print("✅ 차트 생성 완료!")
        print(f"   저장 위치: {self.chart_output_dir}")
        print("="*100)

    def save_results(self, results):
        """
        결과 저장 (압축 CSV)

        Args:
            results: {'over_provisioned': DataFrame, 'unused': DataFrame}
        """
        if not self.config['analysis']['save_results']:
            return

        print("="*100)
        print("💾 결과 저장 중...")
        print("="*100)

        saved_files = []

        # 과다 프로비저닝
        if len(results['over_provisioned']) > 0:
            path1 = self.output_dir / 'over_provisioned_resources.csv.gz'
            results['over_provisioned'].to_csv(
                path1, 
                index=False, 
                encoding='utf-8-sig',
                compression='gzip'
            )
            size_mb = path1.stat().st_size / 1024**2
            print(f"✅ {path1} ({size_mb:.1f} MB)")
            saved_files.append(('과다 프로비저닝', len(results['over_provisioned']), size_mb))
        else:
            print("⚠️  과다 프로비저닝: 탐지된 항목 없음")

        # 미사용 리소스
        if len(results['unused']) > 0:
            path2 = self.output_dir / 'unused_resources.csv.gz'
            results['unused'].to_csv(
                path2, 
                index=False, 
                encoding='utf-8-sig',
                compression='gzip'
            )
            size_mb = path2.stat().st_size / 1024**2
            print(f"✅ {path2} ({size_mb:.1f} MB)")
            saved_files.append(('미사용 리소스', len(results['unused']), size_mb))
        else:
            print("⚠️  미사용 리소스: 탐지된 항목 없음")

        # 요약 통계
        if saved_files:
            print(f"\n📊 저장 요약:")
            for name, count, size in saved_files:
                print(f"   • {name:20s}: {count:,}건 | {size:.1f} MB")

        print("\n" + "="*100)
    
    def _save_cloud_results(self, cloud_results, cloud_name, pattern_config):
        """
        특정 클라우드 결과 저장

        Args:
            cloud_results: 클라우드별 패턴 결과
            cloud_name: 'gcp', 'aws', 또는 'all'
            pattern_config: 패턴 설정
        """
        # 과다 프로비저닝
        if (pattern_config['over_provisioning']['save_csv'] and 
            len(cloud_results['over_provisioned']) > 0):
            filename = f'{cloud_name}_over_provisioned.csv.gz'
            path = self.output_dir / filename
            cloud_results['over_provisioned'].to_csv(
                path, index=False, encoding='utf-8-sig', compression='gzip'
            )
            print(f"✅ [{cloud_name.upper()}] 과다 프로비저닝: {path}")

        # 미사용 리소스
        if (pattern_config['unused_resources']['save_csv'] and 
            len(cloud_results['unused']) > 0):
            filename = f'{cloud_name}_unused_resources.csv.gz'
            path = self.output_dir / filename
            cloud_results['unused'].to_csv(
                path, index=False, encoding='utf-8-sig', compression='gzip'
            )
            print(f"✅ [{cloud_name.upper()}] 미사용 리소스: {path}")
    
    def print_summary(self, results):
        """
        전체 분석 결과 요약 출력
        
        Args:
            results: 분석 결과 딕셔너리
        """
        print("\n" + "="*100)
        print("📊 전체 분석 요약")
        print("="*100)
        
        over_prov = results['over_provisioned']
        unused = results['unused']
        
        # 기본 통계
        print(f"\n1️⃣  탐지 결과:")
        print(f"   • 과다 프로비저닝: {len(over_prov):,}건")
        print(f"   • 미사용 리소스:   {len(unused):,}건")
        print(f"   • 총 문제 리소스:  {len(over_prov) + len(unused):,}건")
        
        # 비용 통계
        total_waste = 0
        
        if len(over_prov) > 0 and 'PotentialSavings' in over_prov.columns:
            over_savings = over_prov['PotentialSavings'].sum()
            total_waste += over_savings
            print(f"\n2️⃣  예상 절감액:")
            print(f"   • 과다 프로비저닝: ${over_savings:,.2f}/월")
        
        if len(unused) > 0 and 'WastedCost' in unused.columns:
            unused_waste = unused['WastedCost'].sum()
            total_waste += unused_waste
            if 'PotentialSavings' not in over_prov.columns:
                print(f"\n2️⃣  예상 절감액:")
            print(f"   • 미사용 리소스:   ${unused_waste:,.2f}/월")
        
        if total_waste > 0:
            print(f"   • 총 절감 가능액:  ${total_waste:,.2f}/월")
            print(f"   • 연간 절감액:     ${total_waste * 12:,.2f}")
            
            # ROI 계산 (시스템 비용 가정: $150/월)
            system_cost = 150
            roi = (total_waste - system_cost) / system_cost * 100
            print(f"\n3️⃣  ROI 분석:")
            print(f"   • 시스템 비용:     ${system_cost}/월")
            print(f"   • 순 절감액:       ${total_waste - system_cost:,.2f}/월")
            print(f"   • ROI:             {roi:,.0f}%")
            print(f"   • 혁신 재투자 가능: ${(total_waste - system_cost) * 0.75:,.2f}/월")
        
        print("\n" + "="*100)


    def run(self, use_sample=False):
        """전체 분석 실행"""
        # 데이터 로드
        self.load_data(use_sample=use_sample)
        
        # 패턴 분석
        results = self.analyze_patterns()
        
        # 결과 저장
        self.save_results(results)
        
        # 요약 출력
        self.print_summary(results)

        # 차트 생성
        # self.generate_charts(results)
        

        

        print("\n✅ 분석 완료!")
        
        return results


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    # 분석기 생성
    analyzer = FocusAnalyzer('config/focus_config.yaml')
    
    # 실행
    results = analyzer.run(use_sample=False)
    
    print("\n🎉 모든 작업 완료!")