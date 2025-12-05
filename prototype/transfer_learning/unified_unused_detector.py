"""
통합 미사용 리소스 탐지기 (Unified Unused Resource Detector)

============================================================
핵심 로직:
============================================================
GCP/AWS 공통: FOCUS 컬럼으로 직접 탐지 (ML 불필요)

조건 1: CommitmentDiscountStatus = 'Unused'
        → 예약 인스턴스/약정 할인을 구매했으나 사용하지 않음
        
조건 2: EffectiveCost = 0 AND BilledCost = 0 AND ConsumedQuantity = 0/null
        → 비용도 0, 사용량도 0인 유휴 리소스

============================================================
입력: resource_grouped.csv (ProviderName으로 GCP/AWS 구분)
출력: unused_resources.csv
============================================================

Author: Lily
Date: 2025-01
Purpose: 석사 논문 - LLM 기반 클라우드 FinOps 자동화 시스템 성능 비교
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys

# ============================================================
# 프로젝트 경로 설정
# ============================================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))


try:
    from pipeline_base import PipelineBase
except ImportError:
    class PipelineBase:
        def __init__(self, config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        
        def print_step(self, msg, detail=""):
            print(f"\n{'='*80}")
            print(f"📌 {msg}")
            if detail:
                print(f"   {detail}")
            print(f"{'='*80}")
        
        def print_success(self, msg):
            print(f"   ✅ {msg}")
        
        def print_warning(self, msg):
            print(f"   ⚠️ {msg}")
        
        def print_error(self, msg):
            print(f"   ❌ {msg}")
        
        def ensure_dir(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)


class UnifiedUnusedDetector(PipelineBase):
    """
    통합 미사용 리소스 탐지기
    
    ============================================================
    탐지 조건:
    ============================================================
    조건 1: CommitmentDiscountStatus = 'Unused'
            예약 인스턴스/Savings Plan 구매 후 미사용
            
    조건 2: 연속 72시간(3일) 이상 Zero Usage
            EffectiveCost=0 & BilledCost=0 & ConsumedQuantity=0
    """
    
    # 연속 시간 임계값 (72시간 = 3일)
    MIN_CONSECUTIVE_HOURS = 72
    
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        """
        super().__init__(config_path)
        
        # ============================================================
        # 경로 설정 (config 기반)
        # ============================================================
        data_config = self.config['data']
        self.input_path = Path(data_config['resource_grouped_output'])
        self.output_path = Path(data_config['unused_resources_output'])
        
        # ============================================================
        # 임계값 설정 (config 기반)
        # ============================================================
        thresholds = self.config['thresholds']['unused_resources']
        self.idle_days = thresholds.get('idle_days', 3)  # 3일 = 72시간
        self.MIN_CONSECUTIVE_HOURS = self.idle_days * 24
        
        # ============================================================
        # 데이터
        # ============================================================
        self.df = None
        self.df_gcp = None
        self.df_aws = None
        
        self.df_commitment_unused = None
        self.df_zero_usage = None
        self.df_unified = None
        
        # 통계
        self.stats = {
            'total': 0,
            'commitment_unused': {'gcp': 0, 'aws': 0, 'total': 0},
            'zero_usage': {'gcp': 0, 'aws': 0, 'total': 0}
        }
    
    
    def load(self):
        """
        데이터 로드
        """
        self.print_step("데이터 로드", f"{self.input_path}")
        
        if not self.input_path.exists():
            self.print_error(f"파일 없음: {self.input_path}")
            raise FileNotFoundError(f"{self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        self.stats['total'] = len(self.df)
        
        self.print_success(f"데이터 로드: {len(self.df):,}건")
        
        # GCP/AWS 분리
        self._separate_by_provider()
        
        return self
    
    
    def _separate_by_provider(self):
        """
        ProviderName 기준 GCP/AWS 분리
        """
        if 'ProviderName' not in self.df.columns:
            self.print_warning("ProviderName 컬럼 없음")
            self.df_gcp = self.df.copy()
            self.df_aws = pd.DataFrame()
            return
        
        gcp_mask = self.df['ProviderName'].str.lower().str.contains('gcp|google', na=False)
        aws_mask = self.df['ProviderName'].str.lower().str.contains('aws|amazon', na=False)
        
        self.df_gcp = self.df[gcp_mask].copy()
        self.df_aws = self.df[aws_mask].copy()
        
        print(f"   📊 GCP: {len(self.df_gcp):,}건")
        print(f"   📊 AWS: {len(self.df_aws):,}건")
    
    
    def _find_consecutive_hours(self, df, flag_col):
        """
        연속 True인 최대 시간 찾기
        
        Args:
            df: 시간순 정렬된 DataFrame
            flag_col: 체크할 boolean 컬럼명
        
        Returns:
            int: 최대 연속 시간
        """
        if flag_col not in df.columns or len(df) == 0:
            return 0
        
        flags = df[flag_col].values
        max_consecutive = 0
        current_consecutive = 0
        
        for flag in flags:
            if flag:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    
    def process(self):
        """
        미사용 리소스 탐지
        """
        self.print_step("미사용 리소스 탐지")
        
        results = []
        
        # ============================================================
        # 조건 1: CommitmentDiscountStatus = 'Unused'
        # ============================================================
        self.df_commitment_unused = self._detect_commitment_unused()
        if len(self.df_commitment_unused) > 0:
            results.append(self.df_commitment_unused)
        
        # ============================================================
        # 조건 2: 연속 72시간 Zero Usage
        # ============================================================
        self.df_zero_usage = self._detect_zero_usage_consecutive()
        if len(self.df_zero_usage) > 0:
            results.append(self.df_zero_usage)
        
        # ============================================================
        # 결과 통합
        # ============================================================
        if results:
            self.df_unified = pd.concat(results, ignore_index=True)
            
            # 중복 제거 (같은 ResourceId)
            if 'ResourceId' in self.df_unified.columns:
                before = len(self.df_unified)
                self.df_unified = self.df_unified.drop_duplicates(subset=['ResourceId'])
                if before > len(self.df_unified):
                    print(f"   ⚠️ 중복 제거: {before - len(self.df_unified):,}건")
        else:
            self.df_unified = pd.DataFrame()
        
        self._print_summary()
        
        return self
    
    
    def _detect_commitment_unused(self):
        """
        조건 1: CommitmentDiscountStatus = 'Unused' 탐지
        
        예약 인스턴스/Savings Plan 구매했는데 사용 안 함
        """
        print(f"\n   🔍 조건 1: CommitmentDiscountStatus = 'Unused'...")
        
        if 'CommitmentDiscountStatus' not in self.df.columns:
            self.print_warning("CommitmentDiscountStatus 컬럼 없음")
            return pd.DataFrame()
        
        # Unused 필터
        result = self.df[
            self.df['CommitmentDiscountStatus'].str.lower() == 'unused'
        ].copy()
        
        if len(result) == 0:
            print(f"      ℹ️ Commitment Unused 없음")
            return pd.DataFrame()
        
        # 메타 정보 추가
        result['UnusedReason'] = 'Commitment-Unused'
        result['DetectionMethod'] = 'CommitmentStatus'
        
        if 'TotalHourlyCost' in result.columns:
            result['TotalHourlyCost'] = pd.to_numeric(result['TotalHourlyCost'], errors='coerce').fillna(0)
            result['WastedCost'] = result['TotalHourlyCost']
        elif 'TotalEffectiveCost' in result.columns:
            result['TotalEffectiveCost'] = pd.to_numeric(result['TotalEffectiveCost'], errors='coerce').fillna(0)
            result['WastedCost'] = result['TotalEffectiveCost']
        else:
            result['WastedCost'] = 0
        
        # 통계
        gcp_count = len(result[result['ProviderName'].str.lower().str.contains('gcp|google', na=False)])
        aws_count = len(result[result['ProviderName'].str.lower().str.contains('aws|amazon', na=False)])
        
        self.stats['commitment_unused']['gcp'] = gcp_count
        self.stats['commitment_unused']['aws'] = aws_count
        self.stats['commitment_unused']['total'] = len(result)
        
        print(f"      ✅ 탐지: {len(result):,}건 (GCP: {gcp_count}, AWS: {aws_count})")
        print(f"      💸 낭비 비용: ${result['WastedCost'].sum():,.2f}")
        
        return result
    
    
    def _detect_zero_usage_consecutive(self):
        """
        조건 2: 연속 72시간 이상 Zero Usage 탐지
        
        EffectiveCost=0 & BilledCost=0 & ConsumedQuantity=0 연속 72시간
        """
        print(f"\n   🔍 조건 2: 연속 {self.MIN_CONSECUTIVE_HOURS}시간 Zero Usage...")
        
        df = self.df.copy()
        
        # 필요한 컬럼 확인
        cost_cols = ['TotalEffectiveCost', 'TotalBilledCost', 'TotalHourlyCost']
        available_cost_col = None
        for col in cost_cols:
            if col in df.columns:
                available_cost_col = col
                break
        
        if available_cost_col is None:
            self.print_warning("비용 컬럼 없음")
            return pd.DataFrame()
        
        # 타입 변환
        df['HourlyTimestamp'] = pd.to_datetime(df['HourlyTimestamp'])
        df[available_cost_col] = pd.to_numeric(df[available_cost_col], errors='coerce').fillna(0)
        
        if 'TotalConsumedQuantity' in df.columns:
            df['TotalConsumedQuantity'] = pd.to_numeric(df['TotalConsumedQuantity'], errors='coerce').fillna(0)
        else:
            df['TotalConsumedQuantity'] = 0
        
        # Zero Usage 플래그
        df['IsZeroUsage'] = (
            (df[available_cost_col] == 0) & 
            (df['TotalConsumedQuantity'] == 0)
        )
        
        # ResourceId별 연속 Zero Usage 체크 (groupby 최적화)
        unused_resources = []
        
        unique_resources = df['ResourceId'].nunique()
        print(f"      📊 리소스 수: {unique_resources:,}개")
        
        grouped = df.sort_values('HourlyTimestamp').groupby('ResourceId')
        
        for i, (resource_id, resource_df) in enumerate(grouped):
            if (i + 1) % 10000 == 0:
                print(f"         진행: {i+1:,}개 처리...")
            
            consecutive = self._find_consecutive_hours(resource_df, 'IsZeroUsage')
            
            if consecutive >= self.MIN_CONSECUTIVE_HOURS:
                last_record = resource_df.iloc[-1].to_dict()
                last_record['ConsecutiveZeroHours'] = consecutive
                last_record['UnusedReason'] = f'Zero-Usage-{consecutive}h'
                last_record['DetectionMethod'] = 'Consecutive_Zero'
                last_record['WastedCost'] = 0  # 비용 0이지만 리소스 점유
                
                unused_resources.append(last_record)
        
        if unused_resources:
            result = pd.DataFrame(unused_resources)
            
            gcp_count = len(result[result['ProviderName'].str.lower().str.contains('gcp|google', na=False)])
            aws_count = len(result[result['ProviderName'].str.lower().str.contains('aws|amazon', na=False)])
            
            self.stats['zero_usage']['gcp'] = gcp_count
            self.stats['zero_usage']['aws'] = aws_count
            self.stats['zero_usage']['total'] = len(result)
            
            print(f"      ✅ 탐지: {len(result):,}건 (GCP: {gcp_count}, AWS: {aws_count})")
            return result
        else:
            print(f"      ℹ️ 연속 {self.MIN_CONSECUTIVE_HOURS}시간 이상 Zero Usage 없음")
            return pd.DataFrame()
    
    
    def _print_summary(self):
        """
        탐지 결과 요약
        """
        print(f"\n{'='*80}")
        print(f"📊 미사용 리소스 탐지 결과 요약")
        print(f"{'='*80}")
        
        # 조건 1
        c1 = self.stats['commitment_unused']
        print(f"\n   [조건 1] Commitment Unused")
        print(f"      • GCP: {c1['gcp']:,}건")
        print(f"      • AWS: {c1['aws']:,}건")
        print(f"      • 합계: {c1['total']:,}건")
        
        # 조건 2
        c2 = self.stats['zero_usage']
        print(f"\n   [조건 2] 연속 {self.MIN_CONSECUTIVE_HOURS}시간 Zero Usage")
        print(f"      • GCP: {c2['gcp']:,}건")
        print(f"      • AWS: {c2['aws']:,}건")
        print(f"      • 합계: {c2['total']:,}건")
        
        # 총합
        total_unused = c1['total'] + c2['total']
        total_pct = (total_unused / self.stats['total'] * 100) if self.stats['total'] > 0 else 0
        
        print(f"\n   [총합]")
        print(f"      • 전체 레코드: {self.stats['total']:,}건")
        print(f"      • 미사용 리소스: {total_unused:,}건 ({total_pct:.2f}%)")
        
        # 낭비 비용
        if self.df_unified is not None and 'WastedCost' in self.df_unified.columns:
            wasted = self.df_unified['WastedCost'].sum()
            print(f"\n   💰 낭비 비용:")
            print(f"      • 시간당: ${wasted:,.2f}")
            print(f"      • 월간: ${wasted * 24 * 30:,.2f}")
        
        print(f"\n{'='*80}")
    
    
    def save(self):
        """
        결과 저장
        """
        self.print_step("결과 저장")
        
        if self.df_unified is None or len(self.df_unified) == 0:
            self.print_warning("저장할 결과 없음")
            return self
        
        self.ensure_dir(self.output_path.parent)
        
        # 출력 컬럼
        output_cols = [
            'ResourceId', 'ProviderName', 'ServiceName', 'ResourceType',
            'UnusedReason', 'DetectionMethod',
            'ConsecutiveZeroHours', 'WastedCost',
            'CommitmentDiscountStatus', 'TotalHourlyCost'
        ]
        
        available_cols = [col for col in output_cols if col in self.df_unified.columns]
        df_output = self.df_unified[available_cols]
        
        df_output.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        
        self.print_success(f"저장 완료: {self.output_path}")
        print(f"      • 레코드: {len(df_output):,}건")
        
        # 통계 JSON
        stats_path = self.output_path.parent / 'unused_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        return self
    
    
    def run(self):
        """
        전체 파이프라인 실행
        """
        return self.load().process().save()
    
    
    def get_results(self):
        """
        결과 반환
        """
        return (self.df_unified, self.stats)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🚀 통합 미사용 리소스 탐지기")
    print("="*80)
    print("📌 탐지 조건:")
    print("   • 조건 1: CommitmentDiscountStatus = 'Unused'")
    print("   • 조건 2: 연속 72시간 이상 Zero Usage")
    print("="*80)
    
    detector = UnifiedUnusedDetector('config/focus_config.yaml')
    detector.run()
    
    df_result, stats = detector.get_results()
    
    print(f"\n✅ 완료!")
    if df_result is not None:
        print(f"   총 미사용 리소스: {len(df_result):,}건")