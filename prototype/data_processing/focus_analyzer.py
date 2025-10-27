"""
FOCUS 데이터 메인 분석기
"""

import yaml
import pandas as pd
from pathlib import Path
import logging

from focus_loader import FocusDataLoader
from focus_patterns import OverProvisioningDetector, UnusedResourceDetector


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
        
        # 패턴 1: 과다 프로비저닝
        detector1 = OverProvisioningDetector(self.df, self.config)
        results['over_provisioned'] = detector1.detect()
        
        # 패턴 2: 미사용 리소스
        detector2 = UnusedResourceDetector(self.df, self.config)
        results['unused'] = detector2.detect()
        
        return results
    
    
    def save_results(self, results):
        """결과 저장"""
        if not self.config['analysis']['save_results']:
            return
        
        print("="*100)
        print("💾 결과 저장 중...")
        print("="*100)
        
        # 과다 프로비저닝
        if len(results['over_provisioned']) > 0:
            path1 = self.output_dir / 'over_provisioned_resources.csv'
            results['over_provisioned'].to_csv(path1, index=False, encoding='utf-8-sig')
            print(f"✅ {path1}")
        
        # 미사용 리소스
        if len(results['unused']) > 0:
            path2 = self.output_dir / 'unused_resources.csv'
            results['unused'].to_csv(path2, index=False, encoding='utf-8-sig')
            print(f"✅ {path2}")
        
        print("\n" + "="*100)
    
    
    def run(self, use_sample=False):
        """전체 분석 실행"""
        # 데이터 로드
        self.load_data(use_sample=use_sample)
        
        # 패턴 분석
        results = self.analyze_patterns()
        
        # 결과 저장
        self.save_results(results)
        
        print("\n✅ 분석 완료!")
        
        return results


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    # 분석기 생성
    analyzer = FocusAnalyzer('config/focus_config.yaml')
    
    # 실행
    results = analyzer.run(use_sample=False)
    
    print("\n🎉 모든 작업 완료!")