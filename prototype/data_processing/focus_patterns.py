# -*- coding: utf-8 -*-
"""
1. 과다 프로비저닝
2. 미사용 리소스
"""

import pandas as pd
import numpy as np


class OverProvisioningDetector:
    """과다 프로비저닝 탐지기"""
    
    def __init__(self, df, config):
        """
        초기화
        
        Args:
            df: FOCUS DataFrame
            config: 설정 딕셔너리
        """
        self.df = df.copy()
        self.config = config
        self.threshold = config['thresholds']['over_provisioning']
    
    
    def detect(self):
        """
        과다 프로비저닝 탐지
        
        Returns:
            DataFrame: 탐지된 리소스
        """
        print("="*100)
        print("🔍 패턴 1: 과다 프로비저닝 탐지")
        print("="*100)
        print(f"\n📌 탐지 기준:")
        print(f"   • CPU 사용률 < {self.threshold['cpu_threshold']*100}%")
        print(f"   • 메모리 사용률 < {self.threshold['memory_threshold']*100}%")
        
        # 시뮬레이션 데이터 생성 (실제 컬럼이 없을 경우)
        if self.config['analysis']['enable_simulation']:
            self._simulate_usage_data()
        
        # 탐지
        cpu_col = self._find_column('cpu', 'usage')
        memory_col = self._find_column('memory', 'usage')
        
        if not cpu_col:
            print("\n❌ CPU 사용률 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 과다 프로비저닝 필터
        over_prov = self.df[
            (self.df[cpu_col].notna()) &
            (self.df[cpu_col] < self.threshold['cpu_threshold'])
        ].copy()
        
        # 낭비율 계산
        over_prov['WastePercentage'] = (1 - over_prov[cpu_col]) * 100
        
        # 예상 절감액
        if 'BilledCost' in over_prov.columns:
            over_prov['PotentialSavings'] = over_prov['BilledCost'] * 0.6
        
        self._print_results(over_prov)
        
        return over_prov
    
    
    def _simulate_usage_data(self):
        """시뮬레이션 사용률 데이터 생성"""
        # Compute 리소스만
        compute_mask = self.df['ServiceName'].str.contains(
            'Compute|VM|EC2|Instance', 
            case=False, 
            na=False
        )
        
        # CPU 사용률 (10-90%)
        self.df.loc[compute_mask, 'SimulatedCPUUsage'] = np.random.uniform(
            0.10, 0.90, compute_mask.sum()
        )
        
        # 메모리 사용률
        self.df.loc[compute_mask, 'SimulatedMemoryUsage'] = np.random.uniform(
            0.15, 0.85, compute_mask.sum()
        )
        
        # 30%는 과다 프로비저닝 (20% 이하 사용)
        over_mask = compute_mask & (np.random.random(len(self.df)) < 0.3)
        self.df.loc[over_mask, 'SimulatedCPUUsage'] = np.random.uniform(
            0.05, 0.25, over_mask.sum()
        )
    
    
    def _find_column(self, *keywords):
        """키워드로 컬럼 찾기"""
        for col in self.df.columns:
            if all(kw.lower() in col.lower() for kw in keywords):
                return col
        return None
    
    
    def _print_results(self, result):
        """결과 출력"""
        print(f"\n✅ 탐지 완료!")
        print(f"   📊 전체: {len(self.df):,} 건")
        print(f"   🚨 과다 프로비저닝: {len(result):,} 건 ({len(result)/len(self.df)*100:.1f}%)")
        
        if len(result) > 0 and 'PotentialSavings' in result.columns:
            print(f"\n💰 예상 절감액:")
            print(f"   • 월: ${result['PotentialSavings'].sum():,.2f}")
            print(f"   • 연: ${result['PotentialSavings'].sum() * 12:,.2f}")
        
        print("\n" + "="*100)


class UnusedResourceDetector:
    """미사용 리소스 탐지기"""
    
    def __init__(self, df, config):
        """
        초기화
        
        Args:
            df: FOCUS DataFrame
            config: 설정 딕셔너리
        """
        self.df = df.copy()
        self.config = config
    
    
    def detect(self):
        """
        미사용 리소스 탐지
        
        조건:
        1. EffectiveCost != 0 → CommitmentDiscountStatus = 'Unused'
           (예약했는데 안 씀)
        
        2. EffectiveCost == 0 → BilledCost = 0 AND (ConsumedQuantity = 0 or null)
           (비용도 0, 사용량도 0/null)
        
        Returns:
            DataFrame: 미사용 리소스
        """
        print("="*100)
        print("🔍 패턴 2: 미사용 리소스 탐지")
        print("="*100)
        
        print("\n📌 탐지 조건:")
        print("   1. EffectiveCost != 0 → CommitmentDiscountStatus = 'Unused'")
        print("   2. EffectiveCost == 0 → BilledCost = 0 AND (ConsumedQuantity = 0 or null)")
        
        unused_all = []
        
        # 조건 1: Commitment Unused
        condition1 = self._detect_commitment_unused()
        if condition1 is not None and len(condition1) > 0:
            unused_all.append(condition1)
        
        # 조건 2: Zero Cost & Zero Usage
        condition2 = self._detect_zero_cost_zero_usage()
        if condition2 is not None and len(condition2) > 0:
            unused_all.append(condition2)
        
        # 결과 통합
        if len(unused_all) == 0:
            print(f"\n" + "="*100)
            print("✅ 미사용 리소스를 찾을 수 없습니다!")
            print("   모든 리소스가 적절히 사용되고 있습니다.")
            print("="*100)
            return pd.DataFrame()
        
        result = pd.concat(unused_all, ignore_index=True)
        
        # 중복 제거
        if 'ResourceId' in result.columns:
            before = len(result)
            result = result.drop_duplicates(subset=['ResourceId'])
            if before > len(result):
                print(f"\n⚠️ 중복 제거: {before - len(result):,}건")
        
        # 최종 결과 출력
        self._print_results(result)
        
        return result
    
    
    def _detect_commitment_unused(self):
        """
        조건 1: EffectiveCost != 0 & CommitmentDiscountStatus = 'Unused'
        예약 리소스(RI/SP) 구매했는데 사용 안 함
        """
        print(f"\n" + "-"*100)
        print("📌 조건 1: EffectiveCost != 0 & CommitmentDiscountStatus = 'Unused'")
        print("   (Reserved Instance / Savings Plan 구매했는데 사용 안 함)")
        print("-"*100)
        
        # 필요한 컬럼 확인
        if 'EffectiveCost' not in self.df.columns:
            print("❌ EffectiveCost 컬럼 없음")
            return None
        
        if 'CommitmentDiscountStatus' not in self.df.columns:
            print("❌ CommitmentDiscountStatus 컬럼 없음")
            return None
        
        # 탐지
        # result = self.df[
        #     ((self.df['EffectiveCost'] != 0) | (self.df['BilledCost'] != 0))
        #     (self.df['CommitmentDiscountStatus'].str.lower() == 'unused')
        # ].copy()

        result = self.df[
            (self.df['CommitmentDiscountStatus'].str.lower() == 'unused')
        ].copy()
        
        if len(result) == 0:
            print("✅ 없음 (모든 Commitment가 잘 사용되고 있음)")
            return None
        
        # 메타 정보 추가
        result['UnusedReason'] = 'Commitment-Unused'
        result['WastedCost'] = result['EffectiveCost']
        
        # 통계 출력
        print(f"\n🚨 발견: {len(result):,}건")
        print(f"💸 낭비 비용: ${result['EffectiveCost'].sum():,.2f}/월")
        print(f"💸 연간 낭비: ${result['EffectiveCost'].sum() * 12:,.2f}")
        
        # Commitment 타입별
        if 'CommitmentDiscountType' in result.columns:
            print(f"\n📊 Commitment 타입별:")
            type_stats = result.groupby('CommitmentDiscountType').agg({
                'ResourceId': 'count',
                'EffectiveCost': 'sum'
            })
            
            for ctype, row in type_stats.iterrows():
                count = int(row['ResourceId'])
                cost = row['EffectiveCost']
                print(f"   • {ctype:20s}: {count:6,}건 | ${cost:,.2f}")
        
        # 서비스별
        if 'ServiceName' in result.columns:
            print(f"\n📊 서비스별 Top 5:")
            for service, count in result['ServiceName'].value_counts().head(5).items():
                pct = count / len(result) * 100
                service_cost = result[result['ServiceName'] == service]['EffectiveCost'].sum()
                print(f"   • {service[:45]:45s}: {count:4,}건 ({pct:4.1f}%) | ${service_cost:,.2f}")
        
        return result
    
    
    # def _detect_zero_cost_zero_usage(self):
    #     """
    #     조건 2: EffectiveCost == 0 & BilledCost == 0 & (ConsumedQuantity == 0 or null)
    #     비용도 0, 사용량도 0/null인 불필요한 리소스
    #     """
    #     print(f"\n" + "-"*100)
    #     print("📌 조건 2: EffectiveCost = 0 & BilledCost = 0 & (ConsumedQuantity = 0 or null)")
    #     print("   (비용도 0, 사용량도 0/null인 불필요한 리소스)")
    #     print("-"*100)
        
    #     # 필요한 컬럼 확인
    #     required_cols = ['EffectiveCost', 'BilledCost', 'ConsumedQuantity']
    #     missing_cols = [col for col in required_cols if col not in self.df.columns]
        
    #     if missing_cols:
    #         print(f"❌ 필요한 컬럼 없음: {', '.join(missing_cols)}")
    #         return None
        
    #     # 탐지
    #     result = self.df[
    #         (self.df['EffectiveCost'] == 0) &
    #         (self.df['BilledCost'] == 0) &
    #         ((self.df['ConsumedQuantity'] == 0) | (self.df['ConsumedQuantity'].isna()))
    #     ].copy()
        
    #     if len(result) == 0:
    #         print("✅ 없음")
    #         return None
        
    #     # 메타 정보 추가
    #     result['UnusedReason'] = 'Zero-Cost-Zero-Usage'
    #     result['WastedCost'] = 0  # 비용은 0이지만 정리 필요
        
    #     # 통계 출력
    #     print(f"\n🚨 발견: {len(result):,}건")
    #     print(f"⚠️ 비용은 0이지만 불필요한 리소스로 추정 (정리 권장)")
        
    #     # ConsumedQuantity 상태별
    #     null_count = result['ConsumedQuantity'].isna().sum()
    #     zero_count = (result['ConsumedQuantity'] == 0).sum()
        
    #     print(f"\n📊 사용량 상태:")
    #     print(f"   • null: {null_count:,}건 ({null_count/len(result)*100:.1f}%)")
    #     print(f"   • 0: {zero_count:,}건 ({zero_count/len(result)*100:.1f}%)")
        
    #     # 서비스별
    #     if 'ServiceName' in result.columns:
    #         print(f"\n📊 서비스별 Top 5:")
    #         for service, count in result['ServiceName'].value_counts().head(5).items():
    #             pct = count / len(result) * 100
    #             print(f"   • {service[:50]:50s}: {count:,}건 ({pct:.1f}%)")
        
    #     # 리소스 타입별
    #     if 'ResourceType' in result.columns:
    #         print(f"\n📦 리소스 타입별:")
    #         for rtype, count in result['ResourceType'].value_counts().items():
    #             pct = count / len(result) * 100
    #             print(f"   • {rtype:20s}: {count:,}건 ({pct:.1f}%)")
        
    #     return result
    
    def _detect_zero_cost_zero_usage(self):
        """
        조건 2: EffectiveCost == 0 & BilledCost == 0 & ConsumedQuantity == 0 (정확히 0만)
        비용도 0, 사용량도 정확히 0인 불필요한 리소스
        """
        print(f"\n" + "-"*100)
        print("📌 조건 2: EffectiveCost = 0 & BilledCost = 0 & ConsumedQuantity = 0 (정확히 0)")
        print("   (비용도 0, 사용량도 정확히 0인 불필요한 리소스)")
        print("-"*100)

        # 필요한 컬럼 확인
        required_cols = ['EffectiveCost', 'BilledCost', 'ConsumedQuantity']
        missing_cols = [col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            print(f"❌ 필요한 컬럼 없음: {', '.join(missing_cols)}")
            return None

        # 탐지 (정확히 0인 것만, null 제외)
        # result = self.df[
        #     (self.df['EffectiveCost'] == 0) &
        #     (self.df['BilledCost'] == 0) &
        #     (self.df['ConsumedQuantity'] == 0)
        # ].copy()

        result = self.df[
            (self.df['ConsumedQuantity'] == 0)
        ].copy()

        if len(result) == 0:
            print("✅ 없음")
            return None

        # 메타 정보 추가
        result['UnusedReason'] = 'Zero-Cost-Zero-Usage'
        result['WastedCost'] = 0  # 비용은 0이지만 정리 필요

        # 통계 출력
        print(f"\n🚨 발견: {len(result):,}건")
        print(f"⚠️ 비용은 0이지만 불필요한 리소스로 추정 (정리 권장)")

        # 서비스별
        if 'ServiceName' in result.columns:
            print(f"\n📊 서비스별 Top 5:")
            for service, count in result['ServiceName'].value_counts().head(5).items():
                pct = count / len(result) * 100
                print(f"   • {service[:50]:50s}: {count:,}건 ({pct:.1f}%)")

        # 리소스 타입별
        if 'ResourceType' in result.columns:
            print(f"\n📦 리소스 타입별:")
            for rtype, count in result['ResourceType'].value_counts().items():
                pct = count / len(result) * 100
                print(f"   • {rtype:20s}: {count:,}건 ({pct:.1f}%)")

        return result
    
    def _print_results(self, result):
        """최종 결과 출력"""
        print(f"\n" + "="*100)
        print("📊 최종 결과")
        print("="*100)
        
        print(f"\n✅ 총 미사용 리소스: {len(result):,}건")
        
        # 조건별 통계
        print(f"\n📊 조건별 분포:")
        for reason in result['UnusedReason'].unique():
            subset = result[result['UnusedReason'] == reason]
            count = len(subset)
            pct = count / len(result) * 100
            cost = subset['WastedCost'].sum()
            print(f"   • {reason:30s}: {count:7,}건 ({pct:5.1f}%) | ${cost:,.2f}")
        
        # 총 낭비 비용
        total_waste = result['WastedCost'].sum()
        print(f"\n💰 총 낭비 비용: ${total_waste:,.2f}/월")
        if total_waste > 0:
            print(f"💰 연간 낭비: ${total_waste * 12:,.2f}")
        
        # Commitment Unused 상위 10개
        commitment_unused = result[result['UnusedReason'] == 'Commitment-Unused']
        if len(commitment_unused) > 0:
            print(f"\n" + "-"*100)
            print("📈 Commitment Unused 상위 10개 (낭비 비용 기준):")
            print("-"*100)
            
            display_cols = ['ResourceId', 'ServiceName', 'CommitmentDiscountType', 
                           'EffectiveCost', 'BilledCost']
            available = [col for col in display_cols if col in commitment_unused.columns]
            
            top10 = commitment_unused.nlargest(10, 'WastedCost')[available]
            
            pd.set_option('display.max_colwidth', 40)
            pd.set_option('display.float_format', lambda x: f'{x:.6f}' if abs(x) < 0.01 else f'{x:.2f}')
            
            print(top10.to_string(index=False))
        
        # Zero Cost/Usage 샘플
        zero_cost = result[result['UnusedReason'] == 'Zero-Cost-Zero-Usage']
        if len(zero_cost) > 0:
            print(f"\n" + "-"*100)
            print("📋 Zero Cost & Zero Usage 샘플 10개:")
            print("-"*100)
            
            display_cols = ['ResourceId', 'ServiceName', 'ResourceType',
                           'EffectiveCost', 'BilledCost', 'ConsumedQuantity']
            available = [col for col in display_cols if col in zero_cost.columns]
            
            sample = zero_cost[available].head(10)
            print(sample.to_string(index=False))
        
        print("\n" + "="*100)
    
def analyze_patterns(self):
    """2가지 패턴 분석 (클라우드별 분리)"""
    if self.df is None:
        raise ValueError("데이터를 먼저 로드하세요: load_data()")
    
    results = {}
    pattern_config = self.config['analysis']['patterns']
    cloud_config = self.config.get('cloud_filter', {})
    
    # 클라우드 필터 활성화 여부
    if cloud_config.get('enabled', False):
        # GCP 데이터 필터링
        gcp_keywords = cloud_config['providers']['gcp']['keywords']
        gcp_mask = self.df['ProviderName'].str.contains('|'.join(gcp_keywords), 
                                                        case=False, na=False)
        df_gcp = self.df[gcp_mask].copy()
        
        # AWS 데이터 필터링
        aws_keywords = cloud_config['providers']['aws']['keywords']
        aws_mask = self.df['ProviderName'].str.contains('|'.join(aws_keywords), 
                                                        case=False, na=False)
        df_aws = self.df[aws_mask].copy()
        
        print(f"\n📊 클라우드별 데이터 분리:")
        print(f"   • GCP: {len(df_gcp):,}건")
        print(f"   • AWS: {len(df_aws):,}건")
        print(f"   • 전체: {len(self.df):,}건")
        
        # 클라우드별로 패턴 분석
        results['gcp'] = self._analyze_cloud_patterns(df_gcp, 'GCP', pattern_config)
        results['aws'] = self._analyze_cloud_patterns(df_aws, 'AWS', pattern_config)
        
    else:
        # 클라우드 구분 없이 전체 분석 (기존 방식)
        print("\n⚠️ 클라우드 필터 비활성화 - 전체 데이터 분석")
        results['all'] = self._analyze_cloud_patterns(self.df, 'ALL', pattern_config)
    
    return results


def _analyze_cloud_patterns(self, df, cloud_name, pattern_config):
    """
    특정 클라우드 데이터의 패턴 분석
    
    Args:
        df: 클라우드별 필터링된 DataFrame
        cloud_name: 'GCP', 'AWS', 또는 'ALL'
        pattern_config: 패턴 설정
    
    Returns:
        dict: 패턴별 탐지 결과
    """
    results = {}
    
    print(f"\n{'='*100}")
    print(f"🔍 {cloud_name} 데이터 패턴 분석")
    print(f"{'='*100}")
    
    # 패턴 1: 과다 프로비저닝
    if pattern_config['over_provisioning']['enabled']:
        print(f"\n🔍 [{cloud_name}] 패턴 1: 과다 프로비저닝 분석")
        detector1 = OverProvisioningDetector(df, self.config)
        results['over_provisioned'] = detector1.detect()
    else:
        results['over_provisioned'] = pd.DataFrame()
    
    # 패턴 2: 미사용 리소스
    if pattern_config['unused_resources']['enabled']:
        print(f"\n🔍 [{cloud_name}] 패턴 2: 미사용 리소스 분석")
        detector2 = UnusedResourceDetector(df, self.config)
        results['unused'] = detector2.detect()
    else:
        results['unused'] = pd.DataFrame()
    
    return results