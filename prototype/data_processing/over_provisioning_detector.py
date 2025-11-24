# -*- coding: utf-8 -*-
"""
4단계: 과다 프로비저닝 탐지 (Over-Provisioning Detection)

- CPU/Memory 사용률이 임계값 이하인 시간 탐지
- 24시간 이상 연속 저사용률 확인
- 낭비 비용 계산
- ProviderName별 구분
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import timedelta


class OverProvisioningDetector:
    """
    과다 프로비저닝 탐지 클래스
    
    주요 기능:
    - 시간대별 CPU/Memory 저사용률 탐지
    - 24시간 이상 연속 저사용 확인
    - 낭비 비용 계산
    - ProviderName별 분석
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        data_config = self.config['data']
        self.input_path = Path(data_config['hourly_aggregated_output'])
        self.output_path = Path(data_config['over_provisioned_output'])
        
        # 임계값 로드
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']
        self.memory_threshold = thresholds['memory_threshold']
        self.min_consecutive_hours = 24  # 24시간 이상 연속
        
        self.df = None
        self.df_over_provisioned = None
    
    
    def print_step(self, message, char='='):
        """단계 출력 헬퍼"""
        print(f"\n{char*100}")
        print(f"🔍 {message}")
        print(f"{char*100}")
    
    
    def load(self):
        """
        HourlyAggregator 결과 로드
        
        Returns:
            DataFrame: 시간대별 집계 데이터
        """
        self.print_step("데이터 로딩")
        
        print(f"   📂 경로: {self.input_path}")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {self.input_path}")
        
        # CSV 로드
        self.df = pl.read_csv(self.input_path, infer_schema_length=0)
        
        print(f"\n✅ 로드 완료!")
        print(f"   📊 레코드: {len(self.df):,}건")
        print(f"   📋 컬럼: {len(self.df.columns)}개")
        
        # 필수 컬럼 확인
        required_cols = ['HourlyTimestamp', 'ProviderName', 'ResourceId', 
                        'HourlyCost', 'AvgCPUUsage', 'AvgMemoryUsage']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            print(f"\n⚠️  누락된 컬럼: {missing}")
        else:
            print(f"\n✅ 필수 컬럼 모두 확인됨")
        
        return self.df
    
    
    def _convert_types(self):
        """
        데이터 타입 변환
        
        - HourlyTimestamp → datetime
        - 사용률/비용 → float
        """
        print(f"\n   🔄 데이터 타입 변환 중...")
        
        # datetime 변환
        self.df = self.df.with_columns([
            pl.col('HourlyTimestamp').str.to_datetime()
        ])
        
        # 숫자형 변환
        numeric_cols = ['HourlyCost', 'AvgCPUUsage', 'AvgMemoryUsage']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df = self.df.with_columns([
                    pl.col(col).cast(pl.Float64, strict=False)
                ])
        
        print(f"   ✅ 타입 변환 완료")
    
    
    def detect(self):
        """
        과다 프로비저닝 탐지 수행
        
        단계:
        1. 각 시간대별로 CPU/Memory 사용률 확인
        2. 임계값 이하인 시간 필터링
        3. 리소스별로 연속 저사용 시간 계산
        4. 24시간 이상 연속인 리소스만 선택
        5. 낭비 비용 계산
        
        Returns:
            DataFrame: 과다 프로비저닝 리소스 목록
        """
        self.print_step("과다 프로비저닝 탐지 시작")
        
        # 타입 변환
        self._convert_types()
        
        print(f"\n📌 탐지 조건:")
        print(f"   • CPU 임계값: {self.cpu_threshold*100:.0f}% 이하")
        print(f"   • Memory 임계값: {self.memory_threshold*100:.0f}% 이하")
        print(f"   • 최소 연속 시간: {self.min_consecutive_hours}시간")
        
        # CPU/Memory 사용률이 없으면 탐지 불가
        if 'AvgCPUUsage' not in self.df.columns or 'AvgMemoryUsage' not in self.df.columns:
            print(f"\n❌ CPU/Memory 사용률 컬럼이 없어 탐지할 수 없습니다.")
            return pl.DataFrame()
        
        # 1. 저사용률 시간대 필터링
        print(f"\n   1️⃣  저사용률 시간대 필터링 중...")
        
        low_usage = self.df.filter(
            (pl.col('AvgCPUUsage') < self.cpu_threshold) |
            (pl.col('AvgMemoryUsage') < self.memory_threshold)
        )
        
        print(f"      • 전체: {len(self.df):,}건")
        print(f"      • 저사용: {len(low_usage):,}건 ({len(low_usage)/len(self.df)*100:.1f}%)")
        
        if len(low_usage) == 0:
            print(f"\n✅ 과다 프로비저닝이 탐지되지 않았습니다!")
            return pl.DataFrame()
        
        # 2. 리소스별 그룹화 및 연속 시간 계산
        print(f"\n   2️⃣  리소스별 연속 저사용 시간 계산 중...")
        
        # 리소스별로 정렬
        low_usage = low_usage.sort(['ProviderName', 'ResourceId', 'HourlyTimestamp'])
        
        # 리소스별 집계
        resource_stats = low_usage.group_by(['ProviderName', 'ResourceId', 'ServiceName']).agg([
            pl.col('HourlyTimestamp').count().alias('LowUsageHours'),
            pl.col('HourlyCost').sum().alias('TotalWastedCost'),
            pl.col('AvgCPUUsage').mean().alias('AvgCPU'),
            pl.col('AvgMemoryUsage').mean().alias('AvgMemory'),
            pl.col('HourlyTimestamp').min().alias('FirstLowUsage'),
            pl.col('HourlyTimestamp').max().alias('LastLowUsage')
        ])
        
        # 3. 24시간 이상 연속인 리소스만 선택
        print(f"\n   3️⃣  {self.min_consecutive_hours}시간 이상 연속 리소스 필터링 중...")
        
        self.df_over_provisioned = resource_stats.filter(
            pl.col('LowUsageHours') >= self.min_consecutive_hours
        )
        
        print(f"      • 저사용 리소스: {len(resource_stats):,}개")
        print(f"      • {self.min_consecutive_hours}시간 이상: {len(self.df_over_provisioned):,}개")
        
        if len(self.df_over_provisioned) == 0:
            print(f"\n✅ {self.min_consecutive_hours}시간 이상 연속 과다 프로비저닝이 없습니다!")
            return pl.DataFrame()
        
        # 4. 낭비 비율 계산
        self.df_over_provisioned = self.df_over_provisioned.with_columns([
            ((1 - pl.col('AvgCPU')) * 100).alias('CPUWastePercent'),
            ((1 - pl.col('AvgMemory')) * 100).alias('MemoryWastePercent')
        ])
        
        # 5. 정렬 (낭비 비용 기준 내림차순)
        self.df_over_provisioned = self.df_over_provisioned.sort('TotalWastedCost', descending=True)
        
        # 결과 출력
        self._print_summary()
        
        return self.df_over_provisioned
    
    
    def _print_summary(self):
        """탐지 결과 요약 통계"""
        print(f"\n{'='*100}")
        print(f"📊 과다 프로비저닝 탐지 결과")
        print(f"{'='*100}")
        
        total_resources = len(self.df_over_provisioned)
        total_wasted_cost = self.df_over_provisioned['TotalWastedCost'].sum()
        
        print(f"\n   🚨 과다 프로비저닝 리소스: {total_resources:,}개")
        print(f"   💸 총 낭비 비용: ${total_wasted_cost:,.2f}/주")
        print(f"   💸 월간 추정: ${total_wasted_cost * 4.33:,.2f}")
        print(f"   💸 연간 추정: ${total_wasted_cost * 52:,.2f}")
        
        # ProviderName별 통계
        if 'ProviderName' in self.df_over_provisioned.columns:
            print(f"\n   ☁️  Provider별 통계:")
            
            provider_stats = self.df_over_provisioned.group_by('ProviderName').agg([
                pl.count().alias('Resources'),
                pl.col('TotalWastedCost').sum().alias('TotalCost'),
                pl.col('LowUsageHours').sum().alias('TotalHours')
            ])
            
            for row in provider_stats.iter_rows(named=True):
                provider = row['ProviderName']
                resources = row['Resources']
                cost = row['TotalCost']
                hours = row['TotalHours']
                cost_pct = (cost / total_wasted_cost * 100) if total_wasted_cost > 0 else 0
                
                print(f"      • {provider:15s}: {resources:>4,}개 | ${cost:>10,.2f} ({cost_pct:5.1f}%) | {hours:>7,}시간")
        
        # ServiceName Top 5
        if 'ServiceName' in self.df_over_provisioned.columns:
            print(f"\n   📊 Service Top 5 (낭비 비용 기준):")
            
            service_stats = self.df_over_provisioned.group_by('ServiceName').agg([
                pl.count().alias('Resources'),
                pl.col('TotalWastedCost').sum().alias('TotalCost')
            ]).sort('TotalCost', descending=True).head(5)
            
            for i, row in enumerate(service_stats.iter_rows(named=True), 1):
                service = row['ServiceName']
                resources = row['Resources']
                cost = row['TotalCost']
                cost_pct = (cost / total_wasted_cost * 100) if total_wasted_cost > 0 else 0
                
                print(f"      {i}. {service[:50]:50s}: {resources:>3,}개 | ${cost:>10,.2f} ({cost_pct:5.1f}%)")
        
        # 사용률 통계
        avg_cpu = self.df_over_provisioned['AvgCPU'].mean()
        avg_mem = self.df_over_provisioned['AvgMemory'].mean()
        
        print(f"\n   📉 평균 사용률:")
        print(f"      • CPU: {avg_cpu*100:.1f}% (낭비율: {(1-avg_cpu)*100:.1f}%)")
        print(f"      • Memory: {avg_mem*100:.1f}% (낭비율: {(1-avg_mem)*100:.1f}%)")
        
        print(f"\n{'='*100}")
    
    
    def save(self):
        """
        탐지 결과 저장
        
        Returns:
            Path: 저장된 파일 경로
        """
        if self.df_over_provisioned is None or len(self.df_over_provisioned) == 0:
            print(f"\n⚠️  저장할 데이터가 없습니다.")
            return None
        
        self.print_step("결과 저장")
        
        print(f"   📂 경로: {self.output_path}")
        
        # 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        self.df_over_provisioned.write_csv(self.output_path)
        
        # 파일 크기
        file_size_mb = self.output_path.stat().st_size / 1024**2
        
        print(f"\n✅ 저장 완료!")
        print(f"   💾 크기: {file_size_mb:.2f} MB")
        
        print(f"\n{'='*100}")
        
        return self.output_path
    
    
    def get_top_offenders(self, n=10):
        """
        상위 N개 낭비 리소스 반환
        
        Args:
            n: 반환할 개수
        
        Returns:
            DataFrame: 상위 N개 리소스
        """
        if self.df_over_provisioned is None or len(self.df_over_provisioned) == 0:
            return pl.DataFrame()
        
        return self.df_over_provisioned.head(n)
    
    
    def run(self):
        """
        전체 탐지 프로세스 실행
        
        Returns:
            tuple: (탐지 결과 DataFrame, 출력 경로)
        """
        # 1. 로드
        self.load()
        
        # 2. 탐지
        self.detect()
        
        # 3. 저장
        output_path = self.save()
        
        print(f"\n✅ 과다 프로비저닝 탐지 완료!")
        
        return self.df_over_provisioned, output_path


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    print("\n🚀 4단계: 과다 프로비저닝 탐지 (Over-Provisioning Detection)")
    print("="*100)
    
    detector = OverProvisioningDetector('config/focus_config.yaml')
    
    df_over_provisioned, output_path = detector.run()
    
    if len(df_over_provisioned) > 0:
        print(f"\n📋 상위 10개 리소스:")
        top10 = detector.get_top_offenders(10)
        print(top10)
    
    print(f"\n🎉 탐지 완료!")
    if output_path:
        print(f"📂 출력 파일: {output_path}")