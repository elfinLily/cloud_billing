"""
GCP Kaggle 데이터를 FOCUS 표준으로 변환

입력: gcp_cost_dataset.csv (Kaggle)
출력: FOCUS 표준 형식 CSV

주요 기능:
1. 컬럼명 매핑 (GCP → FOCUS)
2. ProviderName, ResourceType 추가
3. 시뮬레이션 컬럼 생성 (CommitmentDiscountStatus 등)
4. 날짜 형식 변환

"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


class GCPToFocusConverter:
    """GCP Kaggle 데이터 → FOCUS 표준 변환기"""
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        # Config 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 경로 설정
        self.input_path = self.config['data']['gcp_raw_path']
        self.output_path = self.config['data']['gcp_focus_output']
        self.output_dir = Path(self.config['data']['output_dir'])
        
        self.df = None
        self.focus_df = None
    
    
    def load_data(self):
        """GCP 데이터 로드"""
        print("="*100)
        print("🔄 GCP 데이터 로딩...")
        print("="*100)
        print(f"   📂 경로: {self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        
        print(f"✅ 로드 완료: {len(self.df):,}건")
        print(f"   컬럼: {len(self.df.columns)}개")
        print()
        
        return self.df
    
    
    def convert_to_focus(self):
        """FOCUS 표준 형식으로 변환"""
        print("="*100)
        print("🔄 FOCUS 표준 변환 중...")
        print("="*100)
        
        # 새 DataFrame 생성
        focus = pd.DataFrame()
        
        # ===== 1. 직접 매핑 (컬럼명만 변경) =====
        print("\n1️⃣  직접 매핑...")
        
        focus['ResourceId'] = self.df['Resource ID']
        focus['ServiceName'] = self.df['Service Name']
        focus['ConsumedQuantity'] = self.df['Usage Quantity']
        focus['ConsumedUnit'] = self.df['Usage Unit']
        focus['Region'] = self.df['Region/Zone']
        
        # 비용
        focus['BilledCost'] = self.df['Unrounded Cost ($)']  # USD
        focus['EffectiveCost'] = self.df['Unrounded Cost ($)']  # 동일하게 (할인 없음 가정)
        focus['ListCost'] = self.df['Unrounded Cost ($)']  # 정가
        
        print(f"   ✅ 기본 컬럼 매핑 완료")
        
        # ===== 2. 날짜 변환 =====
        print("\n2️⃣  날짜 형식 변환...")
        
        focus['ChargePeriodStart'] = pd.to_datetime(
            self.df['Usage Start Date'], 
            format='%d-%m-%Y %H:%M'
        )
        focus['ChargePeriodEnd'] = pd.to_datetime(
            self.df['Usage End Date'], 
            format='%d-%m-%Y %H:%M'
        )
        
        print(f"   ✅ 날짜 변환 완료")
        print(f"      기간: {focus['ChargePeriodStart'].min()} ~ {focus['ChargePeriodEnd'].max()}")
        
        # ===== 3. 고정값 추가 =====
        print("\n3️⃣  고정값 추가...")
        
        focus['ProviderName'] = 'Google Cloud'
        focus['PublisherName'] = 'Google'
        focus['InvoiceIssuerName'] = 'Google Cloud'
        focus['BillingAccountId'] = 'GCP-KAGGLE-001'
        focus['BillingCurrency'] = 'USD'
        focus['PricingCategory'] = 'On-Demand'
        
        print(f"   ✅ 고정값 추가 완료")
        
        # ===== 4. ResourceType 생성 (ServiceName 기반) =====
        print("\n4️⃣  ResourceType 생성...")
        
        focus['ResourceType'] = self.df['Service Name'].apply(self._map_resource_type)
        
        type_counts = focus['ResourceType'].value_counts()
        print(f"   ✅ ResourceType 생성 완료:")
        for rtype, count in type_counts.items():
            print(f"      • {rtype}: {count}건")
        
        # ===== 5. 시뮬레이션 컬럼 추가 =====
        print("\n5️⃣  시뮬레이션 컬럼 생성...")
        
        # CommitmentDiscountStatus (미사용 리소스 탐지용)
        # 기본값: 'Used' (모두 사용 중)
        focus['CommitmentDiscountStatus'] = 'Used'
        focus['CommitmentDiscountType'] = 'None'
        
        # CPU/Memory 사용률 (그대로 유지)
        focus['SimulatedCPUUsage'] = self.df['CPU Utilization (%)'] / 100
        focus['SimulatedMemoryUsage'] = self.df['Memory Utilization (%)'] / 100
        
        # 네트워크 데이터
        focus['NetworkInboundBytes'] = self.df['Network Inbound Data (Bytes)']
        focus['NetworkOutboundBytes'] = self.df['Network Outbound Data (Bytes)']
        
        print(f"   ✅ 시뮬레이션 컬럼 추가 완료")
        
        # ===== 6. PricingQuantity 계산 =====
        print("\n6️⃣  가격 관련 필드 계산...")
        
        focus['PricingQuantity'] = focus['ConsumedQuantity']
        focus['PricingUnit'] = focus['ConsumedUnit']
        focus['ContractedCost'] = 0.0  # 계약 할인 없음
        focus['ContractedUnitPrice'] = focus['BilledCost'] / focus['PricingQuantity'].replace(0, 1)
        
        print(f"   ✅ 가격 필드 계산 완료")
        
        # ===== 7. 컬럼 순서 정리 =====
        print("\n7️⃣  컬럼 순서 정리...")
        
        # FOCUS 표준 순서
        column_order = [
            # 청구 기간
            'ChargePeriodStart',
            'ChargePeriodEnd',
            
            # 제공자 정보
            'ProviderName',
            'PublisherName',
            'InvoiceIssuerName',
            
            # 리소스 정보
            'ResourceId',
            'ServiceName',
            'ResourceType',
            'Region',
            
            # 비용
            'BilledCost',
            'EffectiveCost',
            'ListCost',
            'ContractedCost',
            
            # 사용량
            'ConsumedQuantity',
            'ConsumedUnit',
            'PricingQuantity',
            'PricingUnit',
            'ContractedUnitPrice',
            
            # 할인
            'CommitmentDiscountStatus',
            'CommitmentDiscountType',
            
            # 청구 정보
            'BillingAccountId',
            'BillingCurrency',
            'PricingCategory',
            
            # 시뮬레이션 데이터
            'SimulatedCPUUsage',
            'SimulatedMemoryUsage',
            'NetworkInboundBytes',
            'NetworkOutboundBytes',
        ]
        
        self.focus_df = focus[column_order]
        
        print(f"   ✅ 총 {len(self.focus_df.columns)}개 컬럼")
        
        return self.focus_df
    
    
    def _map_resource_type(self, service_name):
        """
        ServiceName → ResourceType 매핑
        
        Args:
            service_name: GCP 서비스명
        
        Returns:
            str: FOCUS ResourceType
        """
        service_lower = service_name.lower()
        
        # Compute 관련
        if any(kw in service_lower for kw in ['compute', 'engine', 'vm']):
            return 'Compute'
        
        # Storage 관련
        if any(kw in service_lower for kw in ['storage', 'disk', 'bucket']):
            return 'Storage'
        
        # Network 관련
        if any(kw in service_lower for kw in ['network', 'cdn', 'interconnect', 'load']):
            return 'Networking'
        
        # Container 관련
        if any(kw in service_lower for kw in ['kubernetes', 'gke', 'container']):
            return 'Container'
        
        # Database 관련
        if any(kw in service_lower for kw in ['sql', 'database', 'firestore', 'datastore', 'bigtable']):
            return 'Database'
        
        # Analytics 관련
        if any(kw in service_lower for kw in ['bigquery', 'dataflow', 'dataproc', 'analytics']):
            return 'Analytics'
        
        # Messaging 관련
        if any(kw in service_lower for kw in ['pub/sub', 'pubsub', 'messaging']):
            return 'Messaging'
        
        # Serverless 관련
        if any(kw in service_lower for kw in ['cloud run', 'cloud functions', 'app engine']):
            return 'Serverless'
        
        # 기타
        return 'Other'
    
    
    def save(self):
        """FOCUS 데이터 저장"""
        print("\n" + "="*100)
        print("💾 저장 중...")
        print("="*100)
        
        # 디렉토리 생성 (config에서 가져온 output_dir 사용)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.focus_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        file_size = output_path.stat().st_size / 1024  # KB
        
        print(f"\n✅ 저장 완료!")
        print(f"   경로: {output_path}")
        print(f"   레코드: {len(self.focus_df):,}건")
        print(f"   컬럼: {len(self.focus_df.columns)}개")
        print(f"   크기: {file_size:.1f} KB")
        print("\n" + "="*100)
    
    
    def show_summary(self):
        """변환 결과 요약"""
        print("\n" + "="*100)
        print("📊 FOCUS 변환 결과 요약")
        print("="*100)
        
        print(f"\n✅ 변환 완료: {len(self.focus_df):,}건")
        
        # 비용 통계
        print(f"\n💰 비용 통계:")
        print(f"   • BilledCost 총합: ${self.focus_df['BilledCost'].sum():,.2f}")
        print(f"   • 평균 비용: ${self.focus_df['BilledCost'].mean():.2f}")
        print(f"   • 최대 비용: ${self.focus_df['BilledCost'].max():,.2f}")
        
        # ProviderName
        print(f"\n☁️  Provider:")
        for provider, count in self.focus_df['ProviderName'].value_counts().items():
            print(f"   • {provider}: {count:,}건")
        
        # ResourceType 분포
        print(f"\n📦 ResourceType 분포:")
        for rtype, count in self.focus_df['ResourceType'].value_counts().items():
            pct = count / len(self.focus_df) * 100
            print(f"   • {rtype:15s}: {count:4,}건 ({pct:5.1f}%)")
        
        # 시뮬레이션 데이터
        print(f"\n🖥️  사용률 (시뮬레이션):")
        print(f"   • CPU 평균: {self.focus_df['SimulatedCPUUsage'].mean()*100:.2f}%")
        print(f"   • Memory 평균: {self.focus_df['SimulatedMemoryUsage'].mean()*100:.2f}%")
        
        # CommitmentDiscountStatus
        print(f"\n💳 CommitmentDiscountStatus:")
        for status, count in self.focus_df['CommitmentDiscountStatus'].value_counts().items():
            print(f"   • {status}: {count:,}건")
        
        print("\n" + "="*100)
    
    
    def run(self):
        """전체 변환 프로세스 실행"""
        # 데이터 로드
        self.load_data()
        
        # FOCUS 변환
        self.convert_to_focus()
        
        # 저장
        self.save()
        
        # 요약
        self.show_summary()
        
        return self.focus_df


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("="*100)
    print("🚀 GCP → FOCUS 변환 시작")
    print("="*100)
    print("\n📋 설정 파일: config/focus_config.yaml")
    
    converter = GCPToFocusConverter(config_path='config/focus_config.yaml')
    
    print(f"   • 입력: {converter.input_path}")
    print(f"   • 출력: {converter.output_path}")
    print()
    
    # 실행
    focus_df = converter.run()
    
    print("\n" + "="*100)
    print("🎉 모든 변환 완료!")
    print("="*100)
    print(f"   👉 결과 파일: {converter.output_path}")
    print("="*100)