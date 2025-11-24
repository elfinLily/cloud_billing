"""
GCP 허깅페이스 데이터 → FOCUS 표준 변환기

허깅페이스에서 다운로드한 GCP billing CSV를 
FinOps FOCUS 1.0 표준 형식으로 변환합니다.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class GCPToFocusConverter:
    """GCP 허깅페이스 데이터를 FOCUS 형식으로 변환"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 경로 설정 (config의 data에서 가져오기)
        data_config = self.config['data']
        self.input_path = Path(data_config['gcp_raw_path'])
        self.output_path = Path(data_config['gcp_focus_output'])
        
        self.df_raw = None
        self.df_focus = None
    
    
    def load(self):
        """
        허깅페이스 GCP billing CSV 로드
        
        Returns:
            DataFrame: 원본 데이터
        """
        print("="*100)
        print(f"🔄 데이터 로딩: {self.input_path}")
        print("="*100)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {self.input_path}")
        
        # CSV 로드
        self.df_raw = pd.read_csv(self.input_path)
        
        print(f"✅ 로드 완료!")
        print(f"   📊 총 레코드: {len(self.df_raw):,} 건")
        print(f"   📋 총 컬럼: {len(self.df_raw.columns)} 개")
        print(f"   💾 메모리: {self.df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")
        
        return self.df_raw
    
    
    def convert(self):
        """
        FOCUS 표준 형식으로 변환
        
        Returns:
            DataFrame: FOCUS 형식 데이터
        """
        if self.df_raw is None:
            raise ValueError("❌ 데이터를 먼저 로드하세요: load()")
        
        print("="*100)
        print("🔄 FOCUS 표준 형식으로 변환 중...")
        print("="*100)
        
        # 새 DataFrame 생성
        self.df_focus = pd.DataFrame()
        
        # ========== 필수 컬럼 매핑 ==========
        
        # 1. 청구 기간
        self.df_focus['ChargePeriodStart'] = pd.to_datetime(self.df_raw['Usage Start Date'])
        self.df_focus['ChargePeriodEnd'] = pd.to_datetime(self.df_raw['Usage End Date'])
        
        # 2. 비용 정보
        self.df_focus['BilledCost'] = self.df_raw['Rounded Cost ($)']
        self.df_focus['EffectiveCost'] = self.df_raw['Rounded Cost ($)']
        
        # 3. 클라우드 제공자 정보
        self.df_focus['InvoiceIssuerName'] = 'Google Cloud'
        self.df_focus['ProviderName'] = 'GCP'
        self.df_focus['PublisherName'] = 'Google'
        
        # 4. 서비스 및 리소스
        self.df_focus['ServiceName'] = self.df_raw['Service Name']
        self.df_focus['ResourceId'] = self.df_raw['Resource ID']
        
        # 5. 리소스 타입 추론
        self.df_focus['ResourceType'] = self._infer_resource_type(self.df_raw['Service Name'])
        
        # 6. 사용량 정보
        self.df_focus['ConsumedQuantity'] = self.df_raw['Usage Quantity']
        self.df_focus['ConsumedUnit'] = self.df_raw['Usage Unit']
        
        # 7. 지역 정보
        self.df_focus['Region'] = self.df_raw['Region / Zone']
        
        # 8. ChargeDescription
        self.df_focus['ChargeDescription'] = (
            self.df_raw['Service Name'] + ' - ' + 
            self.df_raw['Usage Quantity'].astype(str) + ' ' + 
            self.df_raw['Usage Unit']
        )
        
        # 9. Commitment
        self.df_focus['CommitmentDiscountStatus'] = 'Used'
        self.df_focus['CommitmentDiscountType'] = None
        
        # 10. CPU/메모리 사용률
        self.df_focus['CPUUsage'] = self.df_raw['CPU Utilization (%)'] / 100.0
        self.df_focus['MemoryUsage'] = self.df_raw['Memory Utilization (%)'] / 100.0
        
        # 11. 네트워크 데이터
        self.df_focus['NetworkInboundBytes'] = self.df_raw['Network Inbound Data (Bytes)']
        self.df_focus['NetworkOutboundBytes'] = self.df_raw['Network Outbound Data (Bytes)']
        
        # 12. 원본 비용 정보
        self.df_focus['UnroundedCost'] = self.df_raw['Unrounded Cost ($)']
        self.df_focus['CostPerQuantity'] = self.df_raw['Cost per Quantity ($)']
        
        print(f"✅ 변환 완료!")
        print(f"   📊 FOCUS 레코드: {len(self.df_focus):,} 건")
        print(f"   📋 FOCUS 컬럼: {len(self.df_focus.columns)} 개\n")
        
        return self.df_focus
    
    
    def _infer_resource_type(self, service_names):
        """
        서비스명으로부터 리소스 타입 추론
        
        Args:
            service_names: 서비스명 Series
            
        Returns:
            Series: 리소스 타입
        """
        def classify(service):
            service_lower = str(service).lower()
            
            # Compute
            if any(kw in service_lower for kw in ['engine', 'run', 'app engine', 'kubernetes']):
                return 'Compute'
            
            # Storage
            if any(kw in service_lower for kw in ['storage', 'filestore', 'persistent disk']):
                return 'Storage'
            
            # Database
            if any(kw in service_lower for kw in ['sql', 'spanner', 'firestore', 'bigtable', 'memorystore']):
                return 'Database'
            
            # Networking
            if any(kw in service_lower for kw in ['cdn', 'load balancing', 'armor', 'vpc']):
                return 'Networking'
            
            # Analytics
            if any(kw in service_lower for kw in ['bigquery', 'dataflow', 'dataproc', 'pub/sub']):
                return 'Analytics'
            
            # AI/ML
            if any(kw in service_lower for kw in ['ai', 'ml', 'vertex', 'dialogflow', 'vision', 'speech']):
                return 'AI/ML'
            
            # Developer Tools
            if any(kw in service_lower for kw in ['build', 'functions', 'scheduler', 'tasks']):
                return 'Developer Tools'
            
            # Monitoring
            if any(kw in service_lower for kw in ['monitoring', 'logging', 'trace', 'profiler']):
                return 'Monitoring'
            
            # Security
            if any(kw in service_lower for kw in ['secret', 'kms', 'security']):
                return 'Security'
            
            # Container Registry
            if 'registry' in service_lower or 'artifact' in service_lower:
                return 'Container Registry'
            
            # 기타
            return 'Other'
        
        return service_names.apply(classify)
    
    
    def save(self):
        """
        FOCUS 형식 CSV 저장
        """
        if self.df_focus is None:
            raise ValueError("❌ 변환을 먼저 수행하세요: convert()")
        
        print("="*100)
        print(f"💾 FOCUS 파일 저장 중: {self.output_path}")
        print("="*100)
        
        # 디렉토리 생성
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        self.df_focus.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        
        # 파일 크기 확인
        file_size_mb = self.output_path.stat().st_size / 1024**2
        
        print(f"✅ 저장 완료!")
        print(f"   📂 경로: {self.output_path}")
        print(f"   💾 크기: {file_size_mb:.1f} MB\n")
        
        return self.output_path
    
    
    def get_summary(self):
        """
        변환된 데이터 요약 통계
        
        Returns:
            dict: 요약 통계
        """
        if self.df_focus is None:
            raise ValueError("❌ 변환을 먼저 수행하세요: convert()")
        
        print("="*100)
        print("📊 FOCUS 데이터 요약")
        print("="*100)
        
        summary = {}
        
        # 기본 통계
        summary['total_records'] = len(self.df_focus)
        summary['total_cost'] = float(self.df_focus['BilledCost'].sum())
        summary['avg_cost'] = float(self.df_focus['BilledCost'].mean())
        
        # 기간
        summary['start_date'] = self.df_focus['ChargePeriodStart'].min()
        summary['end_date'] = self.df_focus['ChargePeriodEnd'].max()
        summary['date_range_days'] = (summary['end_date'] - summary['start_date']).days
        
        # 서비스별 통계
        summary['unique_services'] = self.df_focus['ServiceName'].nunique()
        summary['unique_resources'] = self.df_focus['ResourceId'].nunique()
        summary['unique_regions'] = self.df_focus['Region'].nunique()
        
        # 리소스 타입별 통계
        summary['resource_type_counts'] = self.df_focus['ResourceType'].value_counts().to_dict()
        
        # 출력
        print(f"\n📈 기본 통계:")
        print(f"   • 총 레코드: {summary['total_records']:,} 건")
        print(f"   • 총 비용: ${summary['total_cost']:,.2f}")
        print(f"   • 평균 비용: ${summary['avg_cost']:,.2f}")
        print(f"\n📅 기간:")
        print(f"   • 시작: {summary['start_date']}")
        print(f"   • 종료: {summary['end_date']}")
        print(f"   • 기간: {summary['date_range_days']} 일")
        print(f"\n🔢 고유 값:")
        print(f"   • 서비스: {summary['unique_services']} 개")
        print(f"   • 리소스: {summary['unique_resources']} 개")
        print(f"   • 지역: {summary['unique_regions']} 개")
        print(f"\n📦 리소스 타입별:")
        for rtype, count in sorted(summary['resource_type_counts'].items(), 
                                   key=lambda x: x[1], reverse=True):
            pct = count / summary['total_records'] * 100
            print(f"   • {rtype:20s}: {count:6,}건 ({pct:5.1f}%)")
        
        print("\n" + "="*100)
        
        return summary
    
    
    def run(self):
        """
        전체 변환 프로세스 실행: 로드 → 변환 → 저장 → 요약
        
        Returns:
            tuple: (FOCUS DataFrame, 요약 통계, 출력 파일 경로)
        """
        # 1. 로드
        self.load()
        
        # 2. 변환
        self.convert()
        
        # 3. 저장
        output_path = self.save()
        
        # 4. 요약
        summary = self.get_summary()
        
        print("\n✅ 모든 작업 완료!\n")
        
        return self.df_focus, summary, output_path


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 GCP 허깅페이스 데이터 → FOCUS 변환기")
    print("="*100)
    
    # 변환기 생성
    converter = GCPToFocusConverter('config/focus_config.yaml')
    
    # 실행
    df_focus, summary, output_path = converter.run()
    
    print("\n🎉 변환 완료!")
    print(f"📂 출력 파일: {output_path}")