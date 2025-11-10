# -*- coding: utf-8 -*-
"""
시간 범위 정규화 (Time Range Normalization)
ChargePeriodStart ~ ChargePeriodEnd를 1시간 단위로 확장
"""

import yaml
import polars as pl
from pathlib import Path
from datetime import timedelta
from pipeline_base import PipelineBase

class TimeNormalizer(PipelineBase):
    """
    시간 범위를 1시간 단위로 정규화하는 클래스
    
    주요 기능:
    - ChargePeriodStart ~ ChargePeriodEnd 사이의 모든 시간(hour)을 개별 레코드로 확장
    - 각 시간 슬롯에 원본 데이터의 비용을 시간 비율로 분배
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            df (DataFrame): FOCUS 형식의 청구 데이터
                필수 컬럼: ChargePeriodStart, ChargePeriodEnd, BilledCost
        """
        super().__init__(config_path)

        data_config = self.config['data']
        self.output_path = Path(data_config['time_normalized_output'])
        self.gcp_data_path = Path(data_config['gcp_focus_output'])
        self.aws_data_path = Path(data_config['aws_focus_output'])

        self.df_all = None
        self.df_time_normalized = None

    
    def load(self):
        """
        FOCUS 형식의 billing 데이터 CSV 로드
        
        Returns:
            DataFrame: 원본 데이터
        """
        print("="*100)
        print(f"🔄 데이터 로딩: {self.aws_data_path} / {self.gcp_data_path}")
        print("="*100)
        
        dfs_to_concat = []

        if self.gcp_data_path.exists():
            df_gcp = pl.read_csv(self.gcp_data_path, infer_schema_length=0)
            dfs_to_concat.append(df_gcp)
            print(f"   ✅ GCP: {len(df_gcp):,}건, {len(df_gcp.columns)}개 컬럼")
        
        # AWS 데이터 로드
        if self.aws_data_path.exists():
            df_aws = pl.read_csv(self.aws_data_path, infer_schema_length=0)
            dfs_to_concat.append(df_aws)
            print(f"   ✅ AWS: {len(df_aws):,}건, {len(df_aws.columns)}개 컬럼")
        
        if not dfs_to_concat:
            raise FileNotFoundError(
                f"❌ 파일을 찾을 수 없습니다:\n"
                f"   GCP: {self.gcp_data_path}\n"
                f"   AWS: {self.aws_data_path}"
            )

        self.df_all = pl.concat(dfs_to_concat, how='diagonal')

        print(f"\n✅ 로드 완료!")
        print(f"   📊 총 레코드: {len(self.df_all):,}건")

        return self.df_all
    
    def _validate_columns(self):
        """
        필수 컬럼 존재 여부 검증
        
        필수 컬럼: ChargePeriodStart, ChargePeriodEnd
        """
        required_cols = ['ChargePeriodStart', 'ChargePeriodEnd']
        missing = [col for col in required_cols if col not in self.df_all.columns]
        
        if missing:
            raise ValueError(f"❌ 필수 컬럼 누락: {missing}")
        
        print(f"\n✅ 필수 컬럼 검증 완료: {required_cols}")

        return self
    
    
    def _convert_datetime(self):
        """
        날짜/시간 컬럼을 datetime 타입으로 변환
        """
        print("\n🕐 날짜/시간 형식 변환 중...")
        
        # Polars 날짜 변환
        self.df_all = self.df_all.with_columns([
            pl.col('ChargePeriodStart').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False),
            pl.col('ChargePeriodEnd').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S', strict=False)
        ])
        
        if 'BilledCost' in self.df_all.columns:
            self.df_all = self.df_all.with_columns([
                pl.col('BilledCost').cast(pl.Float64, strict=False)
            ])

        # null 제거
        null_count = self.df_all.filter(
            pl.col('ChargePeriodStart').is_null() | 
            pl.col('ChargePeriodEnd').is_null() |
            pl.col('BilledCost').is_null()
        ).height

        if null_count > 0:
            print(f"⚠️  변환 실패: {null_count}건 제거")
            self.df_all = self.df_all.drop_nulls(subset=['ChargePeriodStart', 'ChargePeriodEnd', 'BilledCost'])

        print(f"✅ 변환 완료: {len(self.df_all):,}건")

        return self
    
    
    def normalize(self, distribute_cost=True):
        """
        시간 범위를 1시간 단위로 확장
        
        Args:
            distribute_cost (bool): True면 비용을 시간별로 균등 분배, False면 원본 비용 유지
        
        Returns:
            DataFrame: 시간별로 확장된 데이터
                - HourlyTimestamp: 각 시간 슬롯의 시작 시각
                - OriginalDurationHours: 원본 레코드의 총 시간
                - HourlyCost: 시간당 분배된 비용 (distribute_cost=True인 경우)
        
        Steps:
            1. 각 레코드의 Start ~ End 시간 차이 계산
            2. 시간 차이를 1시간 단위로 분해
            3. 각 시간 슬롯을 개별 행으로 생성
            4. 비용을 시간 수로 나눠서 분배 (옵션)
        """
        print("\n" + "="*100)
        print("🔄 시간 정규화 시작")
        print("="*100)
        
        total_records = len(self.df_all)
        print(f"📊 처리 대상: {total_records:,}건")
        print(f"💰 비용 분배: {'ON (시간별 균등 분배)' if distribute_cost else 'OFF (원본 유지)'}")
        
        # 시간 차이 계산 (시간 단위, 벡터 연산)
        df_with_duration = self.df_all.with_columns([
            ((pl.col('ChargePeriodEnd') - pl.col('ChargePeriodStart')).dt.total_seconds() / 3600)
            .clip(lower_bound=1)
            .cast(pl.Int64)
            .alias('OriginalDurationHours')
        ])
        
        # 시간당 비용 계산 (벡터 연산)
        if distribute_cost:
            df_with_duration = df_with_duration.with_columns([
                (pl.col('BilledCost') / pl.col('OriginalDurationHours')).alias('HourlyCost')
            ])
        else:
            df_with_duration = df_with_duration.with_columns([
                pl.col('BilledCost').alias('HourlyCost')
            ])
        
        df_with_duration = df_with_duration.with_row_count('_row_id')
        
        print("\n   복제 및 확장 중...")
        
        df_with_duration = df_with_duration.with_columns([
            pl.int_ranges(pl.col('OriginalDurationHours')).alias('hour_offsets')
        ])

        # 행확장
        df_expanded = df_with_duration.explode('hour_offsets')

        # 시간 계산
        self.df_time_normalized = df_expanded.with_columns([
            (pl.col('ChargePeriodStart') + pl.duration(hours=pl.col('hour_offsets')))
            .alias('HourlyTimestamp')
        ])

        # 불필요 컬럼 제거
        cols_to_drop = ['_row_id', 'hour_offsets']
        self.df_time_normalized = self.df_time_normalized.drop([c for c in cols_to_drop if c in self.df_time_normalized.columns])
    
        
        print(f"\n✅ 확장 완료!")
        print(f"   원본 레코드: {total_records:,}건")
        print(f"   확장된 레코드: {len(self.df_time_normalized):,}건")
        print(f"   평균 확장 배율: {len(self.df_time_normalized) / total_records:.1f}x")

        print("="*100)

        return self
    
    
    def get_hourly_summary(self):
        """
        시간별 요약 통계 생성
             
        Returns:
            DataFrame: 시간별 집계 데이터
                컬럼: HourlyTimestamp, RecordCount, TotalCost, AvgCost
        """
        print("\n📊 시간별 요약 통계 생성 중...")
        
        summary = self.df_time_normalized.group_by('HourlyTimestamp').agg([
            pl.count().alias('RecordCount'),
            pl.col('HourlyCost').sum().alias('TotalCost'),
            pl.col('HourlyCost').mean().alias('AvgCost')
        ]).sort('HourlyTimestamp')
        self.summary = summary

        print(f"✅ 요약 완료: {len(summary):,}개 시간 슬롯")
        
        return self
    
    def save(self):
        """
        시간 정규화 데이터 CSV 저장
        """
        if self.df_time_normalized is None:
            raise ValueError("❌ 변환을 먼저 수행하세요: normalize()")
        
        print("="*100)
        print(f"💾 시간 정규화 파일 저장 중: {self.output_path}")
        print("="*100)
        
        # 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        self.df_time_normalized.write_csv(self.output_path)
        
        # 파일 크기 확인
        file_size_mb = self.output_path.stat().st_size / 1024**2
        
        print(f"\n✅ 저장 완료!")
        print(f"   📂 경로: {self.output_path}")
        print(f"   💾 크기: {file_size_mb:.1f} MB")
        print("="*100)
        
        return self
    
    def run(self):
        """
        전체 시간 정규화 프로세스 실행: 로드 → 변환 → 저장 → 요약
        
        Returns:
            tuple: (DataFrame, 요약 통계, 출력 파일 경로)
        """
        return (self.load()
                ._validate_columns()
                ._convert_datetime()
                .normalize(distribute_cost=True)
                .save()
                .get_hourly_summary())
    
    def get_results(self):
        """
        분석 결과 반환

        Returns:
            tuple: (정규화 데이터, 요약 통계, 출력 경로)
        """
        return (
            self.df_time_normalized,
            getattr(self, 'summary', None),
            self.output_path
        )


if __name__ == "__main__":
    import yaml
    
    print("\n🚀 FOCUS 형식 데이터 → 시간 정규화")

    normalizer = TimeNormalizer('config/focus_config.yaml')
    normalizer.run()
    
    # 결과 조회
    df_time_normalized, summary, output_path = normalizer.get_results()
    
    print(f"\n✅ 시간 정규화 완료!")
    print(f"📂 출력 파일: {output_path}")
    
    if summary is not None:
        print(f"\n시간별 요약 (처음 10행):")
        print(summary.head(10))

    
