# -*- coding: utf-8 -*-
"""
GCP 패턴 학습 모듈

GCP Hugging Face 데이터에서 CPU/Memory 사용률 패턴을 학습합니다.
서비스 타입별 통계적 특성을 추출하여 AWS 추정에 활용합니다.
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import sys

# PipelineBase 임포트
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing.pipeline_base import PipelineBase


class GCPPatternLearner(PipelineBase):
    """
    GCP 패턴 학습 클래스
    
    주요 기능:
    1. GCP Hugging Face 데이터 로드
    2. CPU/Memory 사용률 컬럼 추출
    3. 서비스 타입별 통계 계산
    4. 학습 패턴 저장
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        
        # 데이터 경로
        data_config = self.config['data']
        self.gcp_data_path = Path(data_config['gcp_raw_path'])
        self.output_path = Path('results/transfer_learning/gcp_learned_patterns.json')
        
        # 결과 저장
        self.df_gcp = None
        self.patterns = None
    
    
    def load(self):
        """
        GCP Hugging Face 데이터 로드
        
        Returns:
            self
        """
        self.print_step("GCP 데이터 로딩", f"{self.gcp_data_path}")
        
        if not self.gcp_data_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.gcp_data_path}")
            raise FileNotFoundError(f"{self.gcp_data_path}")
        
        # CSV 로드
        self.df_gcp = pd.read_csv(self.gcp_data_path)
        
        self.print_success("로드 완료")
        print(f"   📊 레코드: {len(self.df_gcp):,}건")
        print(f"   📋 컬럼: {len(self.df_gcp.columns)}개")
        print(f"   💾 메모리: {self.df_gcp.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return self
    
    
    def _validate_columns(self):
        """
        필수 컬럼 검증 (CPU/Memory 사용률)
        
        Returns:
            self
        """
        print("\n🔍 필수 컬럼 검증 중...")
        
        # CPU 컬럼 찾기
        cpu_cols = [col for col in self.df_gcp.columns if 'cpu' in col.lower()]
        memory_cols = [col for col in self.df_gcp.columns if 'memory' in col.lower()]
        
        if not cpu_cols:
            self.print_warning("CPU 컬럼을 찾을 수 없습니다.")
            # Utilization으로 재검색
            cpu_cols = [col for col in self.df_gcp.columns 
                       if 'cpu' in col.lower() and 'utilization' in col.lower()]
        
        if not memory_cols:
            self.print_warning("Memory 컬럼을 찾을 수 없습니다.")
            # Utilization으로 재검색
            memory_cols = [col for col in self.df_gcp.columns 
                          if 'memory' in col.lower() and 'utilization' in col.lower()]
        
        if not cpu_cols and not memory_cols:
            self.print_error("CPU/Memory 사용률 컬럼을 찾을 수 없습니다!")
            raise ValueError("필수 컬럼 없음")
        
        # 첫 번째 컬럼 선택
        self.cpu_col = cpu_cols[0] if cpu_cols else None
        self.memory_col = memory_cols[0] if memory_cols else None
        
        print(f"   ✅ CPU 컬럼: {self.cpu_col}")
        print(f"   ✅ Memory 컬럼: {self.memory_col}")
        
        # Service Name 컬럼 찾기
        service_cols = [col for col in self.df_gcp.columns 
                       if 'service' in col.lower() and 'name' in col.lower()]
        
        if not service_cols:
            self.print_warning("Service Name 컬럼을 찾을 수 없습니다.")
            self.service_col = None
        else:
            self.service_col = service_cols[0]
            print(f"   ✅ Service 컬럼: {self.service_col}")
        
        return self
    
    
    def _clean_data(self):
        """
        데이터 정제
        
        - Null 값 제거
        - 이상치 제거 (0-100% 범위 밖)
        - 0% 사용률 제거
        
        Returns:
            self
        """
        print("\n🧹 데이터 정제 중...")
        
        original_count = len(self.df_gcp)
        
        # CPU 정제
        if self.cpu_col:
            # Null 제거
            self.df_gcp = self.df_gcp[self.df_gcp[self.cpu_col].notna()]
            
            # 0-1 범위로 정규화 (% → 소수)
            if self.df_gcp[self.cpu_col].max() > 1.5:
                self.df_gcp[self.cpu_col] = self.df_gcp[self.cpu_col] / 100.0
            
            # 이상치 제거 (0-1 범위)
            self.df_gcp = self.df_gcp[
                (self.df_gcp[self.cpu_col] >= 0) & 
                (self.df_gcp[self.cpu_col] <= 1)
            ]
            
            # 0% 제거 (미사용 리소스)
            self.df_gcp = self.df_gcp[self.df_gcp[self.cpu_col] > 0]
        
        # Memory 정제
        if self.memory_col:
            # Null 제거
            self.df_gcp = self.df_gcp[self.df_gcp[self.memory_col].notna()]
            
            # 0-1 범위로 정규화
            if self.df_gcp[self.memory_col].max() > 1.5:
                self.df_gcp[self.memory_col] = self.df_gcp[self.memory_col] / 100.0
            
            # 이상치 제거
            self.df_gcp = self.df_gcp[
                (self.df_gcp[self.memory_col] >= 0) & 
                (self.df_gcp[self.memory_col] <= 1)
            ]
            
            # 0% 제거
            self.df_gcp = self.df_gcp[self.df_gcp[self.memory_col] > 0]
        
        cleaned_count = len(self.df_gcp)
        removed_count = original_count - cleaned_count
        
        self.print_success("데이터 정제 완료")
        print(f"   원본: {original_count:,}건")
        print(f"   정제 후: {cleaned_count:,}건")
        print(f"   제거: {removed_count:,}건 ({removed_count/original_count*100:.1f}%)")
        
        return self
    
    
    def process(self):
        """
        서비스 타입별 패턴 학습
        
        각 서비스별로:
        - CPU/Memory 평균, 중앙값, 표준편차, 최소/최대
        - 분위수 (25%, 50%, 75%)
        - 샘플 수
        
        Returns:
            self
        """
        self.print_step("서비스별 패턴 학습")
        
        if self.service_col is None:
            self.print_error("Service 컬럼이 없어 학습할 수 없습니다.")
            return self
        
        # 서비스별 그룹화
        grouped = self.df_gcp.groupby(self.service_col)
        
        patterns = {}
        
        print(f"\n   총 서비스: {len(grouped)}개")
        print(f"   진행 상황:")
        
        for i, (service, group) in enumerate(grouped, 1):
            if i % 10 == 0 or i == len(grouped):
                print(f"      {i}/{len(grouped)} 완료...", end='\r')
            
            pattern = {
                'service_name': service,
                'sample_count': len(group)
            }
            
            # CPU 통계
            if self.cpu_col:
                cpu_data = group[self.cpu_col].dropna()
                if len(cpu_data) > 0:
                    pattern['cpu'] = {
                        'mean': float(cpu_data.mean()),
                        'median': float(cpu_data.median()),
                        'std': float(cpu_data.std()),
                        'min': float(cpu_data.min()),
                        'max': float(cpu_data.max()),
                        'q25': float(cpu_data.quantile(0.25)),
                        'q50': float(cpu_data.quantile(0.50)),
                        'q75': float(cpu_data.quantile(0.75))
                    }
            
            # Memory 통계
            if self.memory_col:
                mem_data = group[self.memory_col].dropna()
                if len(mem_data) > 0:
                    pattern['memory'] = {
                        'mean': float(mem_data.mean()),
                        'median': float(mem_data.median()),
                        'std': float(mem_data.std()),
                        'min': float(mem_data.min()),
                        'max': float(mem_data.max()),
                        'q25': float(mem_data.quantile(0.25)),
                        'q50': float(mem_data.quantile(0.50)),
                        'q75': float(mem_data.quantile(0.75))
                    }
            
            patterns[service] = pattern
        
        print()  # 줄바꿈
        
        self.patterns = patterns
        self.result = patterns
        
        self.print_success(f"패턴 학습 완료: {len(patterns)}개 서비스")
        
        # 통계 출력
        self._print_pattern_summary()
        
        return self
    
    
    def _print_pattern_summary(self):
        """학습된 패턴 요약 출력"""
        if not self.patterns:
            return
        
        print(f"\n{'='*100}")
        print("📊 학습된 패턴 요약")
        print(f"{'='*100}")
        
        # 전체 통계
        total_samples = sum(p['sample_count'] for p in self.patterns.values())
        print(f"\n   • 총 서비스: {len(self.patterns)}개")
        print(f"   • 총 샘플: {total_samples:,}건")
        
        # CPU 통계
        if self.cpu_col:
            cpu_means = [p['cpu']['mean'] for p in self.patterns.values() 
                        if 'cpu' in p]
            if cpu_means:
                print(f"\n   📊 CPU 사용률 (전체 평균):")
                print(f"      • 평균: {np.mean(cpu_means)*100:.2f}%")
                print(f"      • 중앙값: {np.median(cpu_means)*100:.2f}%")
                print(f"      • 최소: {np.min(cpu_means)*100:.2f}%")
                print(f"      • 최대: {np.max(cpu_means)*100:.2f}%")
        
        # Memory 통계
        if self.memory_col:
            mem_means = [p['memory']['mean'] for p in self.patterns.values() 
                        if 'memory' in p]
            if mem_means:
                print(f"\n   📊 Memory 사용률 (전체 평균):")
                print(f"      • 평균: {np.mean(mem_means)*100:.2f}%")
                print(f"      • 중앙값: {np.median(mem_means)*100:.2f}%")
                print(f"      • 최소: {np.min(mem_means)*100:.2f}%")
                print(f"      • 최대: {np.max(mem_means)*100:.2f}%")
        
        # Top 10 서비스 (샘플 수 기준)
        sorted_patterns = sorted(self.patterns.items(), 
                                key=lambda x: x[1]['sample_count'], 
                                reverse=True)
        
        print(f"\n   📈 Top 10 서비스 (샘플 수):")
        for i, (service, pattern) in enumerate(sorted_patterns[:10], 1):
            count = pattern['sample_count']
            cpu_mean = pattern.get('cpu', {}).get('mean', 0) * 100
            mem_mean = pattern.get('memory', {}).get('mean', 0) * 100
            
            print(f"      {i:2d}. {service[:40]:40s} | "
                  f"{count:6,}건 | CPU: {cpu_mean:5.1f}% | Mem: {mem_mean:5.1f}%")
        
        print(f"\n{'='*100}")
    
    
    def save(self):
        """
        학습된 패턴 JSON 저장
        
        Returns:
            self
        """
        if not self.patterns:
            self.print_error("학습된 패턴이 없습니다.")
            return self
        
        self.print_step("패턴 저장", f"{self.output_path}")
        
        # 디렉토리 생성
        self.ensure_dir(self.output_path.parent)
        
        # JSON 저장
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)
        
        # 파일 크기
        file_size_kb = self.output_path.stat().st_size / 1024
        
        self.print_success("저장 완료")
        print(f"   📂 경로: {self.output_path}")
        print(f"   💾 크기: {file_size_kb:.1f} KB")
        print(f"   📊 서비스: {len(self.patterns)}개")
        
        return self
    
    
    def run(self):
        """
        전체 학습 프로세스 실행
        
        Returns:
            self
        """
        return (self.load()
                ._validate_columns()
                ._clean_data()
                .process()
                .save())
    
    
    def get_results(self):
        """
        학습 결과 반환
        
        Returns:
            dict: 서비스별 패턴 딕셔너리
        """
        return self.patterns


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 GCP 패턴 학습 시작")
    print("="*100)
    
    learner = GCPPatternLearner('config/focus_config.yaml')
    learner.run()
    
    patterns = learner.get_results()
    
    print(f"\n✅ 학습 완료!")
    print(f"   서비스: {len(patterns)}개")
    print(f"   출력: {learner.output_path}")