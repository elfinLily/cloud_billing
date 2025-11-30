# -*- coding: utf-8 -*-
"""
AWS 과다 프로비저닝 탐지기 (Transfer Learning 기반)

GCP에서 학습한 사용률 패턴을 AWS에 적용하여
과다 프로비저닝 리소스를 탐지합니다.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))

from pipeline_base import PipelineBase
from usage_estimator import UsageEstimator


class AWSOverprovisioningDetector(PipelineBase):
    """
    AWS 과다 프로비저닝 탐지 클래스 (Transfer Learning 기반)
    
    주요 기능:
    1. AWS FOCUS 데이터 로드
    2. UsageEstimator로 CPU/Memory 사용률 추정
    3. 추정된 사용률 기반 과다 프로비저닝 탐지
    4. 낭비 비용 계산
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        
        # 경로 설정
        data_config = self.config['data']
        self.aws_data_path = Path(data_config['aws_focus_output'])
        self.output_path = Path('results/transfer_learning/aws_overprovisioned.csv')
        
        # 임계값
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']
        self.memory_threshold = thresholds['memory_threshold']
        
        # 데이터
        self.df_aws = None
        self.df_estimated = None
        self.df_overprovisioned = None
        
        # UsageEstimator
        self.estimator = None
    
    
    def load(self):
        """
        AWS FOCUS 데이터 로드
        
        Returns:
            self
        """
        self.print_step("AWS 데이터 로딩", f"{self.aws_data_path}")
        
        if not self.aws_data_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.aws_data_path}")
            raise FileNotFoundError(f"{self.aws_data_path}")
        
        # CSV 로드
        self.df_aws = pd.read_csv(self.aws_data_path)
        
        self.print_success("로드 완료")
        print(f"   📊 레코드: {len(self.df_aws):,}건")
        print(f"   📋 컬럼: {len(self.df_aws.columns)}개")
        
        # 서비스 현황
        if 'ServiceName' in self.df_aws.columns:
            unique_services = self.df_aws['ServiceName'].nunique()
            print(f"   🔧 고유 서비스: {unique_services}개")
        
        return self
    
    
    def _init_estimator(self):
        """
        UsageEstimator 초기화
        
        Returns:
            self
        """
        print("\n🔧 UsageEstimator 초기화 중...")
        
        self.estimator = UsageEstimator(self.config_path)
        self.estimator.run()
        
        self.print_success("UsageEstimator 준비 완료")
        
        return self
    
    
    def estimate_usage(self):
        """
        AWS 서비스별 CPU/Memory 사용률 추정
        
        Returns:
            self
        """
        self.print_step("사용률 추정 (Transfer Learning)")
        
        if self.estimator is None:
            self._init_estimator()
        
        # 고유 서비스 목록
        services = self.df_aws['ServiceName'].unique().tolist()
        print(f"   📊 추정 대상: {len(services)}개 서비스")
        
        # 일괄 추정
        df_service_estimation = self.estimator.estimate_batch(services)
        
        # AWS 데이터에 추정값 병합
        self.df_estimated = self.df_aws.merge(
            df_service_estimation,
            left_on='ServiceName',
            right_on='aws_service',
            how='left'
        )
        
        self.print_success("사용률 추정 완료")
        print(f"   📊 추정된 레코드: {len(self.df_estimated):,}건")
        
        # 추정 통계
        if 'cpu_mean' in self.df_estimated.columns:
            avg_cpu = self.df_estimated['cpu_mean'].mean()
            avg_mem = self.df_estimated['memory_mean'].mean()
            avg_conf = self.df_estimated['confidence'].mean()
            
            print(f"\n   📈 추정 통계:")
            print(f"      • 평균 CPU 사용률: {avg_cpu*100:.1f}%")
            print(f"      • 평균 Memory 사용률: {avg_mem*100:.1f}%")
            print(f"      • 평균 신뢰도: {avg_conf:.2f}")
        
        return self
    
    
    def process(self):
        """
        과다 프로비저닝 탐지
        
        조건:
        - 추정 CPU 사용률 < cpu_threshold (30%)
        - 또는 추정 Memory 사용률 < memory_threshold (30%)
        
        Returns:
            self
        """
        self.print_step("과다 프로비저닝 탐지")
        
        if self.df_estimated is None:
            self.estimate_usage()
        
        print(f"\n📌 탐지 조건:")
        print(f"   • CPU 임계값: {self.cpu_threshold*100:.0f}% 이하")
        print(f"   • Memory 임계값: {self.memory_threshold*100:.0f}% 이하")
        
        # CPU/Memory 컬럼 확인
        if 'cpu_mean' not in self.df_estimated.columns:
            self.print_error("CPU 추정값이 없습니다.")
            return self
        
        # 과다 프로비저닝 필터링
        mask_cpu = self.df_estimated['cpu_mean'] < self.cpu_threshold
        mask_memory = self.df_estimated['memory_mean'] < self.memory_threshold
        
        self.df_overprovisioned = self.df_estimated[mask_cpu | mask_memory].copy()
        
        # 낭비율 계산
        self.df_overprovisioned['CPUWastePercent'] = (
            (1 - self.df_overprovisioned['cpu_mean']) * 100
        )
        self.df_overprovisioned['MemoryWastePercent'] = (
            (1 - self.df_overprovisioned['memory_mean']) * 100
        )
        
        # 예상 절감액 계산 (비용의 60% 절감 가능 가정)
        if 'BilledCost' in self.df_overprovisioned.columns:
            self.df_overprovisioned['PotentialSavings'] = (
                self.df_overprovisioned['BilledCost'] * 0.6
            )
        elif 'EffectiveCost' in self.df_overprovisioned.columns:
            self.df_overprovisioned['PotentialSavings'] = (
                self.df_overprovisioned['EffectiveCost'] * 0.6
            )
        
        # 결과 통계
        self._print_detection_summary()
        
        self.result = self.df_overprovisioned
        
        return self
    
    
    def _print_detection_summary(self):
        """탐지 결과 요약"""
        print(f"\n{'='*100}")
        print("📊 과다 프로비저닝 탐지 결과 (Transfer Learning 기반)")
        print(f"{'='*100}")
        
        total_records = len(self.df_estimated)
        overprovisioned_count = len(self.df_overprovisioned)
        detection_rate = overprovisioned_count / total_records * 100 if total_records > 0 else 0
        
        print(f"\n   🚨 탐지 현황:")
        print(f"      • 전체 레코드: {total_records:,}건")
        print(f"      • 과다 프로비저닝: {overprovisioned_count:,}건 ({detection_rate:.1f}%)")
        
        # 매칭 방법별
        if 'method' in self.df_overprovisioned.columns:
            print(f"\n   📌 매칭 방법별:")
            for method, count in self.df_overprovisioned['method'].value_counts().items():
                pct = count / overprovisioned_count * 100
                print(f"      • {method:20s}: {count:,}건 ({pct:.1f}%)")
        
        # 신뢰도별
        if 'confidence' in self.df_overprovisioned.columns:
            high_conf = (self.df_overprovisioned['confidence'] >= 0.8).sum()
            medium_conf = ((self.df_overprovisioned['confidence'] >= 0.5) & 
                          (self.df_overprovisioned['confidence'] < 0.8)).sum()
            low_conf = (self.df_overprovisioned['confidence'] < 0.5).sum()
            
            print(f"\n   📌 신뢰도별:")
            print(f"      • 높음 (≥80%): {high_conf:,}건")
            print(f"      • 중간 (50-80%): {medium_conf:,}건")
            print(f"      • 낮음 (<50%): {low_conf:,}건")
        
        # 예상 절감액
        if 'PotentialSavings' in self.df_overprovisioned.columns:
            total_savings = self.df_overprovisioned['PotentialSavings'].sum()
            print(f"\n   💰 예상 절감액:")
            print(f"      • 총 절감 가능: ${total_savings:,.2f}")
            print(f"      • 월간 추정: ${total_savings:,.2f}")
            print(f"      • 연간 추정: ${total_savings * 12:,.2f}")
        
        # 서비스별 Top 5
        if 'ServiceName' in self.df_overprovisioned.columns:
            print(f"\n   📊 서비스별 Top 5:")
            service_counts = self.df_overprovisioned['ServiceName'].value_counts().head(5)
            
            for i, (service, count) in enumerate(service_counts.items(), 1):
                pct = count / overprovisioned_count * 100
                print(f"      {i}. {service[:45]:45s}: {count:,}건 ({pct:.1f}%)")
        
        # 평균 낭비율
        if 'CPUWastePercent' in self.df_overprovisioned.columns:
            avg_cpu_waste = self.df_overprovisioned['CPUWastePercent'].mean()
            avg_mem_waste = self.df_overprovisioned['MemoryWastePercent'].mean()
            
            print(f"\n   📉 평균 낭비율:")
            print(f"      • CPU: {avg_cpu_waste:.1f}%")
            print(f"      • Memory: {avg_mem_waste:.1f}%")
        
        print(f"\n{'='*100}")
    
    
    def save(self):
        """
        탐지 결과 저장
        
        Returns:
            self
        """
        if self.df_overprovisioned is None or len(self.df_overprovisioned) == 0:
            self.print_warning("저장할 데이터가 없습니다.")
            return self
        
        self.print_step("결과 저장", f"{self.output_path}")
        
        # 디렉토리 생성
        self.ensure_dir(self.output_path.parent)
        
        # CSV 저장
        self.df_overprovisioned.to_csv(self.output_path, index=False)
        
        # 파일 크기
        file_size_kb = self.output_path.stat().st_size / 1024
        
        self.print_success("저장 완료")
        print(f"   📂 경로: {self.output_path}")
        print(f"   💾 크기: {file_size_kb:.1f} KB")
        print(f"   📊 레코드: {len(self.df_overprovisioned):,}건")
        
        return self
    
    
    def run(self):
        """
        전체 프로세스 실행
        
        Returns:
            self
        """
        return (self.load()
                .estimate_usage()
                .process()
                .save())
    
    
    def get_results(self):
        """
        탐지 결과 반환
        
        Returns:
            tuple: (과다 프로비저닝 DataFrame, 전체 추정 DataFrame)
        """
        return (self.df_overprovisioned, self.df_estimated)
    
    
    def get_summary_stats(self):
        """
        요약 통계 반환 (논문용)
        
        Returns:
            dict: 요약 통계
        """
        if self.df_overprovisioned is None:
            return {}
        
        stats = {
            'total_records': len(self.df_estimated),
            'overprovisioned_count': len(self.df_overprovisioned),
            'detection_rate': len(self.df_overprovisioned) / len(self.df_estimated) * 100,
            'avg_cpu_usage': self.df_overprovisioned['cpu_mean'].mean() * 100,
            'avg_memory_usage': self.df_overprovisioned['memory_mean'].mean() * 100,
            'avg_cpu_waste': self.df_overprovisioned['CPUWastePercent'].mean(),
            'avg_memory_waste': self.df_overprovisioned['MemoryWastePercent'].mean(),
            'avg_confidence': self.df_overprovisioned['confidence'].mean(),
            'exact_match_count': (self.df_overprovisioned['method'] == 'exact_match').sum(),
            'global_avg_count': (self.df_overprovisioned['method'] == 'global_average').sum(),
        }
        
        if 'PotentialSavings' in self.df_overprovisioned.columns:
            stats['total_potential_savings'] = self.df_overprovisioned['PotentialSavings'].sum()
        
        return stats


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 AWS 과다 프로비저닝 탐지 (Transfer Learning 기반)")
    print("="*100)
    
    detector = AWSOverprovisioningDetector('config/focus_config.yaml')
    detector.run()
    
    # 결과 조회
    df_overprovisioned, df_estimated = detector.get_results()
    
    print(f"\n✅ 탐지 완료!")
    print(f"   과다 프로비저닝: {len(df_overprovisioned):,}건")
    
    # 요약 통계
    stats = detector.get_summary_stats()
    print(f"\n📊 요약 통계:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.2f}")
        else:
            print(f"   • {key}: {value}")