# -*- coding: utf-8 -*-
"""
리소스별 그룹화 (Resource Grouping)
시간 정규화된 데이터를 ResourceId별로 그룹화하여 시간별 집계
"""

import polars as pl
from pathlib import Path
from pipeline_base import PipelineBase


class ResourceGrouper(PipelineBase):
    """
    리소스별 시간 단위 그룹화 클래스
    
    주요 기능:
    - 시간 정규화된 데이터를 ResourceId + HourlyTimestamp로 그룹화
    - 각 리소스의 시간별 비용, 사용량 집계
    - BillingAccountId, ServiceName 등 메타데이터 유지
    
    입력: time_normalized.csv (1시간 단위 확장된 데이터)
    출력: resource_hourly_grouped.csv (리소스별 시간 집계)

    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        super().__init__(config_path)
        
        data_config = self.config['data']
        self.input_path = Path(data_config['time_normalized_output'])
        self.output_path = Path(data_config['resource_grouped_output'])
        
        self.df = None
        self.df_grouped = None
    
    
    def load(self):
        """
        시간 정규화된 데이터 로드
        
        Returns:
            self
        """
        self.print_step("데이터 로딩", f"{self.input_path}")
        
        if not self.input_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.input_path}")
            raise FileNotFoundError(f"{self.input_path}")
        
        self.df = pl.read_csv(self.input_path, infer_schema_length=0)
        
        self.print_success(f"로드 완료: {len(self.df):,}건")
        print(f"   📋 컬럼: {len(self.df.columns)}개")
        
        return self
    
    
    def _validate_columns(self):
        """
        필수 컬럼 검증
        
        Returns:
            self
        """
        print("\n🔍 필수 컬럼 검증 중...")
        
        required_cols = [
            'ResourceId',
            'HourlyTimestamp',
            'HourlyCost',
            'ServiceName',
            'BillingAccountId'
        ]
        
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            self.print_error(f"누락된 컬럼: {missing}")
            raise ValueError(f"필수 컬럼이 없습니다: {missing}")
        
        self.print_success(f"필수 컬럼 검증 완료")
        
        return self
    
    
    def _convert_types(self):
        """
        데이터 타입 변환
        
        Returns:
            self:
        """
        print("\n🔄 데이터 타입 변환 중...")
        
        # HourlyTimestamp를 datetime으로 변환
        self.df = self.df.with_columns([
            pl.col('HourlyTimestamp').str.to_datetime(),
            pl.col('HourlyCost').cast(pl.Float64)
        ])
        
        if 'SimulatedCPUUsage' in self.df.columns:
            self.df = self.df.with_columns([
                pl.col('SimulatedCPUUsage').cast(pl.Float64),
                pl.col('SimulatedMemoryUsage').cast(pl.Float64)
            ])

        self.print_success("타입 변환 완료")
        
        return self
    
    
    def process(self):
        """
        리소스별 그룹화 수행
        
        그룹화 키:
        - ProviderName
        - BillingAccountId
        - ResourceId
        - HourlyTimestamp
        - ServiceName
        - Region (if there is) - option?
        
        집계:
        - TotalHourlyCost: 시간당 총 비용
        - RecordCount: 해당 시간의 레코드 수
        
        Returns:
            self
        """
        self.print_step("리소스별 그룹화 시작")
        
        print(f"   원본 레코드: {len(self.df):,}건")
        
        # 그룹화할 컬럼 결정
        group_cols = [
            'ProviderName',
            'BillingAccountId',
            'ResourceId',
            'HourlyTimestamp',
            'ServiceName'
        ]
        
        # self.df_grouped = self.df.group_by(group_cols).agg([
        #     pl.col('HourlyCost').sum().alias('TotalHourlyCost'),
        #     pl.count().alias('RecordCount')
        # ])

        # 그룹화 및 집계
        agg_exprs = [
            pl.col('HourlyCost').sum().alias('TotalHourlyCost'),
            pl.count().alias('RecordCount')
        ]
        
        if 'ResourceType' in self.df.columns:
            group_cols.append('ResourceType')

        # GCP: SimulatedCPUUsage, SimulatedMemoryUsage 평균
        if 'SimulatedCPUUsage' in self.df.columns:
            agg_exprs.extend([
                pl.col('SimulatedCPUUsage').mean().alias('AvgCPUUsage'),
                pl.col('SimulatedMemoryUsage').mean().alias('AvgMemoryUsage')
            ])

        # AWS: ConsumedQuantity 합계
        # if 'ConsumedQuantity' in self.df.columns:
        #     agg_exprs.append(
        #         pl.col('ConsumedQuantity').sum().alias('TotalConsumedQuantity')
        #     )

        self.df_grouped = self.df.group_by(group_cols).agg(agg_exprs)
        
        self.print_success(f"그룹화 완료: {len(self.df_grouped):,}건")
        print(f"   압축률: {len(self.df) / len(self.df_grouped):.2f}x")
        
        # 요약 통계
        self._print_summary()
        
        return self
    
    
    def _print_summary(self):
        """그룹화 결과 요약 출력"""

        print(f"\n📊 그룹화 요약:")
        if 'ProviderName' in self.df_grouped.columns:
            provider_stats = self.df_grouped.group_by('ProviderName').agg([
                pl.col('ResourceId').n_unique().alias('Resources'),
                pl.col('TotalHourlyCost').sum().alias('TotalCost')
            ])

            print(f"\n   ☁️  Provider별:")
            for row in provider_stats.iter_rows(named=True):
                print(f"      • {row['ProviderName']}: {row['Resources']:,}개 리소스, ${row['TotalCost']:,.2f}")

        # 고유 리소스 수
        unique_resources = self.df_grouped.select('ResourceId').n_unique()
        print(f"   • 고유 리소스: {unique_resources:,}개")
        
        # 고유 빌링 계정 수
        unique_accounts = self.df_grouped.select('BillingAccountId').n_unique()
        print(f"   • 빌링 계정: {unique_accounts:,}개")
        
        # 시간 범위
        min_time = self.df_grouped.select('HourlyTimestamp').min().item()
        max_time = self.df_grouped.select('HourlyTimestamp').max().item()
        print(f"   • 시간 범위: {min_time} ~ {max_time}")
        
        # 총 비용
        total_cost = self.df_grouped.select('TotalHourlyCost').sum().item()
        print(f"   • 총 비용: ${total_cost:,.2f}")
    
    
    def save(self):
        """
        그룹화 결과 저장
        
        Returns:
            self: 체이닝 지원
        """
        if self.df_grouped is None:
            self.print_error("그룹화를 먼저 수행하세요: process()")
            raise ValueError("그룹화 데이터가 없습니다")
        
        self.print_step("결과 저장", f"{self.output_path}")
        
        # 디렉토리 생성
        self.ensure_dir(self.output_path.parent)
        
        # CSV 저장
        self.df_grouped.write_csv(self.output_path)
        
        # 파일 크기 확인
        file_size_mb = self.output_path.stat().st_size / 1024**2
        
        self.print_success("저장 완료")
        print(f"   📂 경로: {self.output_path}")
        print(f"   💾 크기: {file_size_mb:.1f} MB")
        
        return self
    
    
    def run(self):
        """
        전체 그룹화 프로세스 실행
        
        Returns:
            self
        """
        return (self.load()
                ._validate_columns()
                ._convert_types()
                .process()
                .save())
    
    
    def get_results(self):
        """
        그룹화 결과 반환
        
        Returns:
            tuple: (그룹화 데이터, 출력 경로)
        """
        return (self.df_grouped, self.output_path)


if __name__ == "__main__":
    
    print("\n🚀 리소스별 그룹화 시작")
    print("="*100)
    
    grouper = ResourceGrouper('config/focus_config.yaml')
    grouper.run()
    
    # 결과 조회
    df_grouped, output_path = grouper.get_results()
    
    print(f"\n✅ 그룹화 완료!")
    print(f"📂 출력 파일: {output_path}")
    