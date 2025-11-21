# -*- coding: utf-8 -*-
"""
3단계: 시간대별 집계 (Hourly Aggregation)

- BilledCost, EffectiveCost, ConsumedQuantity 합계
- SimulatedCPUUsage, SimulatedMemoryUsage 평균
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import datetime

class HourlyAggregator:
    """
    시간대별 집계 클래스
    
    입력: resource_grouped.csv (ResourceGrouper 출력)
    출력: hourly_aggregated.csv
    
    주요 기능:
    - 시간대별로 비용/사용량 집계
    - CPU/Memory 사용률 평균 계산
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
        self.input_path = Path(data_config.get('resource_grouped_output', 
                                                'data/processed/resource_grouped.csv'))
        self.output_path = Path(data_config.get('hourly_aggregated_output',
                                                 'data/processed/hourly_aggregated.csv'))
        
        self.df = None
        self.df_aggregated = None
    
    
    def print_step(self, message, char='='):
        """단계 출력 헬퍼"""
        print(f"\n{char*100}")
        print(f"🔄 {message}")
        print(f"{char*100}")
    
    
    def load(self):
        """
        ResourceGrouper 결과 로드
        
        Returns:
            DataFrame: 리소스 그룹화된 데이터
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
        
        # 컬럼 확인
        print(f"\n   📝 주요 컬럼:")
        for col in ['ProviderName', 'ResourceId', 'HourlyTimestamp', 'TotalHourlyCost']:
            if col in self.df.columns:
                print(f"      ✅ {col}")
            else:
                print(f"      ❌ {col} (누락)")
        
        return self.df
    
    
    def _convert_types(self):
        """
        데이터 타입 변환
        
        - HourlyTimestamp → datetime
        - 비용/사용량 컬럼 → float
        """
        print(f"\n   🔄 데이터 타입 변환 중...")
        
        # datetime 변환
        if 'HourlyTimestamp' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('HourlyTimestamp').str.to_datetime()
            ])
        
        # 숫자형 변환
        numeric_cols = [
            'TotalHourlyCost',
            'AvgCPUUsage',
            'AvgMemoryUsage',
            'TotalConsumedQuantity'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df = self.df.with_columns([
                    pl.col(col).cast(pl.Float64, strict=False)
                ])
        
        print(f"   ✅ 타입 변환 완료")
    
    
    def aggregate(self):
        """
        시간대별 집계 수행
        
        집계 단위:
        - HourlyTimestamp (1시간)
        - ProviderName (GCP/AWS 구분)
        - ResourceId (리소스별)
        - ServiceName (서비스별)
        
        집계 메트릭:
        - TotalHourlyCost 합계
        - AvgCPUUsage 평균
        - AvgMemoryUsage 평균
        - TotalConsumedQuantity 합계
        
        Returns:
            DataFrame: 시간대별 집계 결과
        """
        self.print_step("시간대별 집계")
        
        # 타입 변환
        self._convert_types()
        
        print(f"\n   원본 레코드: {len(self.df):,}건")
        
        # 집계 키
        group_keys = [
            'HourlyTimestamp',
            'ProviderName',
            'ResourceId',
            'ServiceName'
        ]
        
        # Region이 있으면 추가
        if 'Region' in self.df.columns:
            group_keys.append('Region')
        
        print(f"\n   📌 집계 키: {', '.join(group_keys)}")
        
        # 집계 수행
        agg_exprs = [
            pl.col('TotalHourlyCost').sum().alias('HourlyCost'),
            pl.col('ResourceId').count().alias('RecordCount')
        ]
        
        # CPU 사용률 (있으면)
        if 'AvgCPUUsage' in self.df.columns:
            agg_exprs.append(
                pl.col('AvgCPUUsage').mean().alias('AvgCPUUsage')
            )
        
        # Memory 사용률 (있으면)
        if 'AvgMemoryUsage' in self.df.columns:
            agg_exprs.append(
                pl.col('AvgMemoryUsage').mean().alias('AvgMemoryUsage')
            )
        
        # ConsumedQuantity (있으면)
        if 'TotalConsumedQuantity' in self.df.columns:
            agg_exprs.append(
                pl.col('TotalConsumedQuantity').sum().alias('TotalConsumedQuantity')
            )
        
        # 집계
        self.df_aggregated = self.df.group_by(group_keys).agg(agg_exprs)
        
        # 정렬
        self.df_aggregated = self.df_aggregated.sort(['HourlyTimestamp', 'ProviderName', 'ResourceId'])
        
        print(f"\n✅ 집계 완료!")
        print(f"   📊 집계 후 레코드: {len(self.df_aggregated):,}건")
        print(f"   📋 집계 후 컬럼: {len(self.df_aggregated.columns)}개")
        
        # 통계
        self._print_summary()
        
        return self.df_aggregated
    
    
    def _print_summary(self):
        """집계 결과 요약 통계"""
        print(f"\n{'='*100}")
        print(f"📊 집계 결과 요약")
        print(f"{'='*100}")
        
        # 전체 통계
        total_cost = self.df_aggregated['HourlyCost'].sum()
        total_hours = self.df_aggregated.select(pl.col('HourlyTimestamp').n_unique()).item()
        total_resources = self.df_aggregated.select(pl.col('ResourceId').n_unique()).item()
        
        print(f"\n   ✅ 총 집계 레코드: {len(self.df_aggregated):,}건")
        print(f"   🕐 총 시간 슬롯: {total_hours:,}개")
        print(f"   📦 총 리소스: {total_resources:,}개")
        print(f"   💰 총 비용: ${total_cost:,.2f}")
        
        # ProviderName별 통계
        if 'ProviderName' in self.df_aggregated.columns:
            print(f"\n   ☁️  ProviderName별 통계:")
            
            provider_stats = self.df_aggregated.group_by('ProviderName').agg([
                pl.col('HourlyCost').sum().alias('TotalCost'),
                pl.col('ResourceId').n_unique().alias('Resources'),
                pl.col('HourlyTimestamp').n_unique().alias('Hours')
            ])
            
            for row in provider_stats.iter_rows(named=True):
                provider = row['ProviderName']
                cost = row['TotalCost']
                resources = row['Resources']
                hours = row['Hours']
                cost_pct = (cost / total_cost * 100) if total_cost > 0 else 0
                
                print(f"      • {provider:15s}: ${cost:>12,.2f} ({cost_pct:5.1f}%) | {resources:>6,}개 리소스 | {hours:>6,}시간")
        
        # ServiceName Top 5
        if 'ServiceName' in self.df_aggregated.columns:
            print(f"\n   📊 ServiceName Top 5:")
            
            service_stats = self.df_aggregated.group_by('ServiceName').agg([
                pl.col('HourlyCost').sum().alias('TotalCost')
            ]).sort('TotalCost', descending=True).head(5)
            
            for i, row in enumerate(service_stats.iter_rows(named=True), 1):
                service = row['ServiceName']
                cost = row['TotalCost']
                cost_pct = (cost / total_cost * 100) if total_cost > 0 else 0
                print(f"      {i}. {service[:45]:45s}: ${cost:>12,.2f} ({cost_pct:5.1f}%)")
        
        # CPU/Memory 사용률 (있으면)
        if 'AvgCPUUsage' in self.df_aggregated.columns:
            avg_cpu = self.df_aggregated.select(pl.col('AvgCPUUsage').mean()).item()
            print(f"\n   🖥️  평균 CPU 사용률: {avg_cpu*100:.2f}%")
        
        if 'AvgMemoryUsage' in self.df_aggregated.columns:
            avg_mem = self.df_aggregated.select(pl.col('AvgMemoryUsage').mean()).item()
            print(f"   💾 평균 Memory 사용률: {avg_mem*100:.2f}%")
        
        print(f"\n{'='*100}")
    
    
    def save(self):
        """
        집계 결과 저장
        
        Returns:
            Path: 저장된 파일 경로
        """
        if self.df_aggregated is None:
            raise ValueError("❌ 집계를 먼저 수행하세요: aggregate()")
        
        self.print_step("결과 저장")
        
        print(f"   📂 경로: {self.output_path}")
        
        # 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        self.df_aggregated.write_csv(self.output_path)
        
        # 파일 크기
        file_size_mb = self.output_path.stat().st_size / 1024**2
        
        print(f"\n✅ 저장 완료!")
        print(f"   💾 크기: {file_size_mb:.1f} MB")
        
        print(f"\n{'='*100}")
        
        return self.output_path
    
    
    def get_provider_comparison(self):
        """
        ProviderName별 비교 통계
        
        Returns:
            DataFrame: Provider 비교 통계
        """
        if self.df_aggregated is None:
            raise ValueError("❌ 집계를 먼저 수행하세요: aggregate()")
        
        if 'ProviderName' not in self.df_aggregated.columns:
            print("⚠️  ProviderName 컬럼이 없습니다.")
            return None
        
        print(f"\n{'='*100}")
        print(f"📊 Provider 비교 통계")
        print(f"{'='*100}")
        
        # Provider별 집계
        comparison = self.df_aggregated.group_by('ProviderName').agg([
            pl.col('HourlyCost').sum().alias('TotalCost'),
            pl.col('HourlyCost').mean().alias('AvgCost'),
            pl.col('ResourceId').n_unique().alias('UniqueResources'),
            pl.col('HourlyTimestamp').n_unique().alias('UniqueHours'),
            pl.len().alias('TotalRecords')
        ])
        
        print(f"\n{comparison}")
        
        print(f"\n{'='*100}")
        
        return comparison
    
    
    def run(self):
        """
        전체 집계 프로세스 실행
        
        Returns:
            tuple: (집계 DataFrame, 비교 통계, 출력 경로)
        """
        # 1. 로드
        self.load()
        
        # 2. 집계
        self.aggregate()
        
        # 3. 저장
        output_path = self.save()
        
        # 4. Provider 비교
        comparison = self.get_provider_comparison()
        
        print(f"\n✅ 모든 작업 완료!")
        
        return self.df_aggregated, comparison, output_path


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    print("\n🚀 3단계: 시간대별 집계 (Hourly Aggregation)")
    print("="*100)
    
    aggregator = HourlyAggregator('config/focus_config.yaml')
    
    df_aggregated, comparison, output_path = aggregator.run()
    
    print(f"\n🎉 집계 완료!")
    print(f"📂 출력 파일: {output_path}")