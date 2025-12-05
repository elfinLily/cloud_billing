# -*- coding: utf-8 -*-
"""
통합 과다 프로비저닝 탐지기 (Unified Over-Provisioning Detector)

============================================================
핵심 로직:
============================================================
GCP: AvgCPUUsage/AvgMemoryUsage 직접 비교 (< 30% → 과다)
AWS: ML Classifier 예측 등급 사용 (Low → 과다)

============================================================
입력: resource_grouped.csv (ProviderName으로 GCP/AWS 구분)
출력: unified_overprovisioned.csv
============================================================

Author: Lily
Date: 2025-01
Purpose: 석사 논문 - LLM 기반 클라우드 FinOps 자동화 시스템 성능 비교
"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
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
    # PipelineBase 없을 경우 기본 구현
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


class UnifiedOverProvisioningDetector(PipelineBase):
    """
    통합 과다 프로비저닝 탐지기
    
    ============================================================
    탐지 방법:
    ============================================================
    
    [GCP] 직접 임계값 비교
        - AvgCPUUsage < 30% → CPU 과다 프로비저닝
        - AvgMemoryUsage < 30% → Memory 과다 프로비저닝
        - 정확도: 100% (실제 사용률 데이터 있음)
    
    [AWS] ML Classification 기반
        - Transfer Learning으로 학습된 모델 사용
        - PredictedCPUClass == 'Low' → CPU 과다 프로비저닝
        - PredictedMemoryClass == 'Low' → Memory 과다 프로비저닝
        - 정확도: 97% (ML 모델)
    
    ============================================================
    출력 컬럼:
    ============================================================
    - ResourceId: 리소스 식별자
    - ProviderName: GCP / AWS
    - ServiceName: 서비스명
    - DetectionMethod: 'Direct' (GCP) / 'ML_Classification' (AWS)
    - CPUStatus: 'OverProvisioned' / 'Normal'
    - MemoryStatus: 'OverProvisioned' / 'Normal'
    - CPUValue: 실제값(GCP) / 예측등급(AWS)
    - MemoryValue: 실제값(GCP) / 예측등급(AWS)
    - TotalHourlyCost: 시간당 비용
    - PotentialSavings: 예상 절감액
    """
    
    # ============================================================
    # 서비스 → UnifiedCategory 매핑 (ML 예측용)
    # ============================================================
    SERVICE_CATEGORY_MAP = {
        # Compute
        'compute engine': 'Compute', 'ec2': 'Compute', 'amazon ec2': 'Compute',
        'cloud functions': 'Compute', 'lambda': 'Compute', 'aws lambda': 'Compute',
        'cloud run': 'Compute', 'ecs': 'Compute', 'fargate': 'Compute',
        'app engine': 'Compute', 'elastic beanstalk': 'Compute',
        
        # Container
        'kubernetes engine': 'Container', 'gke': 'Container',
        'eks': 'Container', 'amazon eks': 'Container',
        
        # Database
        'cloud sql': 'Database', 'rds': 'Database', 'amazon rds': 'Database',
        'aurora': 'Database', 'dynamodb': 'Database', 'bigtable': 'Database',
        'firestore': 'Database', 'elasticache': 'Database', 'redshift': 'Database',
        
        # Storage
        'cloud storage': 'Storage', 's3': 'Storage', 'amazon s3': 'Storage',
        'persistent disk': 'Storage', 'ebs': 'Storage', 'efs': 'Storage',
        
        # Analytics
        'bigquery': 'Analytics', 'athena': 'Analytics', 'dataproc': 'Analytics',
        'emr': 'Analytics', 'kinesis': 'Analytics', 'glue': 'Analytics',
        
        # AI/ML
        'vertex ai': 'AI_ML', 'sagemaker': 'AI_ML', 'automl': 'AI_ML',
        
        # Networking
        'vpc': 'Networking', 'cloud load balancing': 'Networking',
        'elb': 'Networking', 'cloudfront': 'Networking', 'cloud cdn': 'Networking',
        
        # Monitoring
        'cloud monitoring': 'Monitoring', 'cloudwatch': 'Monitoring',
        
        # Security
        'iam': 'Security', 'kms': 'Security', 'waf': 'Security',
        
        # Messaging
        'pub/sub': 'Messaging', 'sns': 'Messaging', 'sqs': 'Messaging',
    }
    
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        
        # ============================================================
        # 경로 설정 (config 기반)
        # ============================================================
        data_config = self.config['data']
        self.input_path = Path(data_config['resource_grouped_output'])
        self.output_path = Path('results/transfer_learning/unified_overprovisioned.csv')
        
        # ML 모델 경로
        self.model_dir = Path('results/transfer_learning/models/classifier')
        
        # ============================================================
        # 임계값 설정 (config 기반)
        # ============================================================
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']      # 0.30 (30%)
        self.memory_threshold = thresholds['memory_threshold']  # 0.30 (30%)
        
        # ============================================================
        # 데이터 & 모델
        # ============================================================
        self.df = None              # 전체 데이터
        self.df_gcp = None          # GCP 데이터
        self.df_aws = None          # AWS 데이터
        
        self.df_gcp_result = None   # GCP 탐지 결과
        self.df_aws_result = None   # AWS 탐지 결과
        self.df_unified = None      # 통합 결과
        
        # ML 모델 (AWS용)
        self.cpu_model = None
        self.memory_model = None
        self.label_encoders = {}
        self.scaler = None
        self.class_encoder = None
        
        # 통계
        self.stats = {
            'gcp': {'total': 0, 'over_provisioned': 0},
            'aws': {'total': 0, 'over_provisioned': 0}
        }
    
    
    def load(self):
        """
        데이터 및 ML 모델 로드
        
        Returns:
            self
        """
        self.print_step("데이터 로드", f"{self.input_path}")
        
        # ============================================================
        # 1. resource_grouped.csv 로드
        # ============================================================
        if not self.input_path.exists():
            self.print_error(f"파일 없음: {self.input_path}")
            raise FileNotFoundError(f"{self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        self.print_success(f"데이터 로드: {len(self.df):,}건")
        
        # ============================================================
        # 2. GCP/AWS 분리
        # ============================================================
        self._separate_by_provider()
        
        # ============================================================
        # 3. ML 모델 로드 (AWS 예측용)
        # ============================================================
        if len(self.df_aws) > 0:
            self._load_ml_models()
        
        return self
    
    
    def _separate_by_provider(self):
        """
        ProviderName으로 GCP/AWS 데이터 분리
        """
        print(f"\n   🔄 클라우드 제공자별 분리...")
        
        # ProviderName 컬럼 확인
        if 'ProviderName' not in self.df.columns:
            self.print_warning("ProviderName 컬럼 없음 - 전체를 GCP로 처리")
            self.df_gcp = self.df.copy()
            self.df_aws = pd.DataFrame()
            return
        
        # GCP 필터
        gcp_keywords = ['GCP', 'Google', 'google']
        gcp_mask = self.df['ProviderName'].str.contains(
            '|'.join(gcp_keywords), case=False, na=False
        )
        self.df_gcp = self.df[gcp_mask].copy()
        
        # AWS 필터
        aws_keywords = ['AWS', 'Amazon', 'amazon']
        aws_mask = self.df['ProviderName'].str.contains(
            '|'.join(aws_keywords), case=False, na=False
        )
        self.df_aws = self.df[aws_mask].copy()
        
        # 통계 저장
        self.stats['gcp']['total'] = len(self.df_gcp)
        self.stats['aws']['total'] = len(self.df_aws)
        
        print(f"   ✅ GCP: {len(self.df_gcp):,}건")
        print(f"   ✅ AWS: {len(self.df_aws):,}건")
        
        # 기타 데이터 경고
        other_count = len(self.df) - len(self.df_gcp) - len(self.df_aws)
        if other_count > 0:
            self.print_warning(f"기타 Provider: {other_count:,}건 (제외됨)")
    
    
    def _load_ml_models(self):
        """
        AWS 예측용 ML 모델 로드
        """
        print(f"\n   🔄 ML 모델 로드...")
        
        # 모델 파일 확인
        cpu_model_path = self.model_dir / 'cpu_classifier.joblib'
        memory_model_path = self.model_dir / 'memory_classifier.joblib'
        
        if not cpu_model_path.exists():
            self.print_error(f"CPU 모델 없음: {cpu_model_path}")
            self.print_warning("ML 모델 없음 - AWS 데이터는 처리 불가")
            self.print_warning("먼저 ml_usage_classifier.py 실행 필요")
            return
        
        # 모델 로드
        self.cpu_model = joblib.load(cpu_model_path)
        self.memory_model = joblib.load(memory_model_path)
        self.label_encoders = joblib.load(self.model_dir / 'label_encoders.joblib')
        self.scaler = joblib.load(self.model_dir / 'scaler.joblib')
        self.class_encoder = joblib.load(self.model_dir / 'class_encoder.joblib')
        
        self.print_success("ML 모델 로드 완료")
        print(f"      • CPU Classifier: ✅")
        print(f"      • Memory Classifier: ✅")
        print(f"      • Label Encoders: ✅")
        print(f"      • Scaler: ✅")
    
    
    def process(self):
        """
        과다 프로비저닝 탐지 실행
        
        Returns:
            self
        """
        self.print_step("과다 프로비저닝 탐지")
        
        results = []
        
        # ============================================================
        # 1. GCP 직접 탐지
        # ============================================================
        if len(self.df_gcp) > 0:
            self.df_gcp_result = self._detect_gcp()
            if len(self.df_gcp_result) > 0:
                results.append(self.df_gcp_result)
        
        # ============================================================
        # 2. AWS ML 기반 탐지
        # ============================================================
        if len(self.df_aws) > 0 and self.cpu_model is not None:
            self.df_aws_result = self._detect_aws()
            if len(self.df_aws_result) > 0:
                results.append(self.df_aws_result)
        elif len(self.df_aws) > 0:
            self.print_warning("AWS 데이터 있으나 ML 모델 없음 - 스킵")
        
        # ============================================================
        # 3. 결과 통합
        # ============================================================
        if results:
            self.df_unified = pd.concat(results, ignore_index=True)
        else:
            self.df_unified = pd.DataFrame()
        
        # 통계 출력
        self._print_summary()
        
        return self
    
    
    def _detect_gcp(self):
        """
        GCP 과다 프로비저닝 탐지 (직접 임계값 비교)
        
        조건:
        - AvgCPUUsage < 30% → CPU 과다 프로비저닝
        - AvgMemoryUsage < 30% → Memory 과다 프로비저닝
        
        Returns:
            DataFrame: 탐지된 과다 프로비저닝 리소스
        """
        print(f"\n   🔍 [GCP] 직접 임계값 비교...")
        print(f"      • CPU 임계값: < {self.cpu_threshold*100:.0f}%")
        print(f"      • Memory 임계값: < {self.memory_threshold*100:.0f}%")
        
        df = self.df_gcp.copy()
        
        # 사용률 컬럼 확인
        cpu_col = 'AvgCPUUsage'
        mem_col = 'AvgMemoryUsage'
        
        if cpu_col not in df.columns:
            self.print_warning(f"GCP 데이터에 {cpu_col} 없음")
            return pd.DataFrame()
        
        # 숫자 변환
        df[cpu_col] = pd.to_numeric(df[cpu_col], errors='coerce').fillna(0)
        df[mem_col] = pd.to_numeric(df[mem_col], errors='coerce').fillna(0)
        
        # 과다 프로비저닝 판정
        cpu_over = df[cpu_col] < self.cpu_threshold
        mem_over = df[mem_col] < self.memory_threshold
        is_over = cpu_over | mem_over
        
        df_over = df[is_over].copy()
        
        # 결과 컬럼 추가
        df_over['DetectionMethod'] = 'Direct'
        df_over['CPUStatus'] = np.where(
            df_over[cpu_col] < self.cpu_threshold,
            'OverProvisioned', 'Normal'
        )
        df_over['MemoryStatus'] = np.where(
            df_over[mem_col] < self.memory_threshold,
            'OverProvisioned', 'Normal'
        )
        df_over['CPUValue'] = df_over[cpu_col].apply(lambda x: f"{x*100:.1f}%")
        df_over['MemoryValue'] = df_over[mem_col].apply(lambda x: f"{x*100:.1f}%")
        
        # 예상 절감액 계산 (낭비 비율 기반)
        cost_col = 'TotalHourlyCost'
        if cost_col in df_over.columns:
            df_over[cost_col] = pd.to_numeric(df_over[cost_col], errors='coerce').fillna(0)
            
            # 낭비 비율 = 1 - 실제 사용률
            cpu_waste = 1 - df_over[cpu_col]
            mem_waste = 1 - df_over[mem_col]
            avg_waste = (cpu_waste + mem_waste) / 2
            
            df_over['WasteRatio'] = avg_waste
            df_over['PotentialSavings'] = df_over[cost_col] * avg_waste
        else:
            df_over['WasteRatio'] = 0
            df_over['PotentialSavings'] = 0
        
        # 통계 저장
        self.stats['gcp']['over_provisioned'] = len(df_over)
        
        print(f"      ✅ 탐지: {len(df_over):,}건 / {len(self.df_gcp):,}건")
        print(f"         ({len(df_over)/len(self.df_gcp)*100:.1f}%)")
        
        return df_over
    
    
    def _detect_aws(self):
        """
        AWS 과다 프로비저닝 탐지 (ML Classification 기반)
        
        조건:
        - PredictedCPUClass == 'Low' → CPU 과다 프로비저닝
        - PredictedMemoryClass == 'Low' → Memory 과다 프로비저닝
        
        Returns:
            DataFrame: 탐지된 과다 프로비저닝 리소스
        """
        print(f"\n   🔍 [AWS] ML Classification 기반 탐지...")
        print(f"      • 모델: RandomForestClassifier (97% Accuracy)")
        print(f"      • 기준: 'Low' 등급 → 과다 프로비저닝")
        
        df = self.df_aws.copy()
        
        # ============================================================
        # Feature 준비
        # ============================================================
        df = self._prepare_features(df)
        
        # Feature 인코딩
        X = self._encode_features(df)
        
        if X is None or len(X) == 0:
            self.print_warning("Feature 인코딩 실패")
            return pd.DataFrame()
        
        # ============================================================
        # ML 예측
        # ============================================================
        print(f"      🔄 ML 예측 중...")
        
        cpu_pred_encoded = self.cpu_model.predict(X)
        mem_pred_encoded = self.memory_model.predict(X)
        
        # 인코딩 → 등급 변환
        cpu_pred = self.class_encoder.inverse_transform(cpu_pred_encoded)
        mem_pred = self.class_encoder.inverse_transform(mem_pred_encoded)
        
        df['PredictedCPUClass'] = cpu_pred
        df['PredictedMemoryClass'] = mem_pred
        
        # 등급 분포 출력
        print(f"\n      📊 예측 등급 분포:")
        print(f"         CPU:")
        for cls in ['Low', 'Medium', 'High']:
            cnt = (cpu_pred == cls).sum()
            print(f"            • {cls}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)")
        print(f"         Memory:")
        for cls in ['Low', 'Medium', 'High']:
            cnt = (mem_pred == cls).sum()
            print(f"            • {cls}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)")
        
        # ============================================================
        # 과다 프로비저닝 판정 (Low 등급)
        # ============================================================
        cpu_over = df['PredictedCPUClass'] == 'Low'
        mem_over = df['PredictedMemoryClass'] == 'Low'
        is_over = cpu_over | mem_over
        
        df_over = df[is_over].copy()
        
        # 결과 컬럼 추가
        df_over['DetectionMethod'] = 'ML_Classification'
        df_over['CPUStatus'] = np.where(
            df_over['PredictedCPUClass'] == 'Low',
            'OverProvisioned', 'Normal'
        )
        df_over['MemoryStatus'] = np.where(
            df_over['PredictedMemoryClass'] == 'Low',
            'OverProvisioned', 'Normal'
        )
        df_over['CPUValue'] = df_over['PredictedCPUClass']
        df_over['MemoryValue'] = df_over['PredictedMemoryClass']
        
        # 예상 절감액 계산
        cost_col = 'TotalHourlyCost'
        if cost_col in df_over.columns:
            df_over[cost_col] = pd.to_numeric(df_over[cost_col], errors='coerce').fillna(0)
            
            # Low 등급 → 약 50% 절감 가정
            # (Low = 하위 25% 사용률 → 평균 ~12.5% → 87.5% 낭비)
            df_over['WasteRatio'] = 0.5  # 보수적 추정
            df_over['PotentialSavings'] = df_over[cost_col] * 0.5
        else:
            df_over['WasteRatio'] = 0
            df_over['PotentialSavings'] = 0
        
        # 통계 저장
        self.stats['aws']['over_provisioned'] = len(df_over)
        
        print(f"\n      ✅ 탐지: {len(df_over):,}건 / {len(self.df_aws):,}건")
        print(f"         ({len(df_over)/len(self.df_aws)*100:.1f}%)")
        
        return df_over
    
    
    def _prepare_features(self, df):
        """
        AWS 데이터 Feature 준비
        """
        # ServiceName → UnifiedCategory
        if 'ServiceName' in df.columns:
            df['UnifiedCategory'] = df['ServiceName'].apply(self._map_to_category)
        else:
            df['UnifiedCategory'] = 'Other'
        
        # LogCost
        cost_col = 'TotalHourlyCost'
        if cost_col in df.columns:
            df['LogCost'] = np.log1p(
                pd.to_numeric(df[cost_col], errors='coerce').fillna(0)
            )
        else:
            df['LogCost'] = 0
        
        # HourOfDay, DayOfWeek
        if 'HourlyTimestamp' in df.columns:
            df['HourlyTimestamp'] = pd.to_datetime(df['HourlyTimestamp'], errors='coerce')
            df['HourOfDay'] = df['HourlyTimestamp'].dt.hour.fillna(12).astype(int)
            df['DayOfWeek'] = df['HourlyTimestamp'].dt.dayofweek.fillna(3).astype(int)
        else:
            df['HourOfDay'] = 12
            df['DayOfWeek'] = 3
        
        # ResourceType 기본값
        if 'ResourceType' not in df.columns:
            df['ResourceType'] = 'Unknown'
        
        return df
    
    
    def _map_to_category(self, service_name):
        """
        서비스명 → UnifiedCategory 매핑
        """
        if pd.isna(service_name):
            return 'Other'
        
        service_lower = str(service_name).lower()
        
        for keyword, category in self.SERVICE_CATEGORY_MAP.items():
            if keyword in service_lower:
                return category
        
        return 'Other'
    
    
    def _encode_features(self, df):
        """
        Feature 인코딩 (학습된 인코더 사용)
        """
        try:
            encoded_data = []
            
            # Categorical
            categorical_cols = ['UnifiedCategory', 'ResourceType']
            for col in categorical_cols:
                if col not in self.label_encoders:
                    continue
                
                encoder = self.label_encoders[col]
                known_classes = set(encoder.classes_)
                
                values = df[col].fillna('Unknown').astype(str)
                values = values.apply(lambda x: x if x in known_classes else 'Unknown')
                
                encoded = encoder.transform(values)
                encoded_data.append(encoded.reshape(-1, 1))
            
            # Numerical
            numerical_cols = ['LogCost', 'HourOfDay', 'DayOfWeek']
            numerical_data = df[numerical_cols].fillna(0).values
            numerical_scaled = self.scaler.transform(numerical_data)
            encoded_data.append(numerical_scaled)
            
            return np.hstack(encoded_data)
        
        except Exception as e:
            self.print_error(f"Feature 인코딩 실패: {e}")
            return None
    
    
    def _print_summary(self):
        """
        탐지 결과 요약 출력
        """
        print(f"\n{'='*80}")
        print(f"📊 과다 프로비저닝 탐지 결과 요약")
        print(f"{'='*80}")
        
        # GCP 통계
        gcp_total = self.stats['gcp']['total']
        gcp_over = self.stats['gcp']['over_provisioned']
        gcp_pct = (gcp_over / gcp_total * 100) if gcp_total > 0 else 0
        
        print(f"\n   [GCP] 직접 탐지 (임계값 < 30%)")
        print(f"      • 전체: {gcp_total:,}건")
        print(f"      • 과다 프로비저닝: {gcp_over:,}건 ({gcp_pct:.1f}%)")
        
        # AWS 통계
        aws_total = self.stats['aws']['total']
        aws_over = self.stats['aws']['over_provisioned']
        aws_pct = (aws_over / aws_total * 100) if aws_total > 0 else 0
        
        print(f"\n   [AWS] ML Classification (Low 등급)")
        print(f"      • 전체: {aws_total:,}건")
        print(f"      • 과다 프로비저닝: {aws_over:,}건 ({aws_pct:.1f}%)")
        
        # 통합 통계
        total = gcp_total + aws_total
        total_over = gcp_over + aws_over
        total_pct = (total_over / total * 100) if total > 0 else 0
        
        print(f"\n   [통합]")
        print(f"      • 전체: {total:,}건")
        print(f"      • 과다 프로비저닝: {total_over:,}건 ({total_pct:.1f}%)")
        
        # 예상 절감액
        if self.df_unified is not None and 'PotentialSavings' in self.df_unified.columns:
            savings = self.df_unified['PotentialSavings'].sum()
            print(f"\n   💰 예상 절감액:")
            print(f"      • 시간당: ${savings:,.2f}")
            print(f"      • 월간: ${savings * 24 * 30:,.2f}")
            print(f"      • 연간: ${savings * 24 * 365:,.2f}")
        
        print(f"\n{'='*80}")
    
    
    def save(self):
        """
        결과 저장
        
        Returns:
            self
        """
        self.print_step("결과 저장")
        
        if self.df_unified is None or len(self.df_unified) == 0:
            self.print_warning("저장할 결과 없음")
            return self
        
        # 디렉토리 생성
        self.ensure_dir(self.output_path.parent)
        
        # 출력 컬럼 선택
        output_cols = [
            'ResourceId', 'ProviderName', 'ServiceName', 'ResourceType',
            'DetectionMethod', 'CPUStatus', 'MemoryStatus',
            'CPUValue', 'MemoryValue',
            'TotalHourlyCost', 'WasteRatio', 'PotentialSavings'
        ]
        
        # 존재하는 컬럼만 선택
        available_cols = [col for col in output_cols if col in self.df_unified.columns]
        df_output = self.df_unified[available_cols]
        
        # CSV 저장
        df_output.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        
        self.print_success(f"저장 완료: {self.output_path}")
        print(f"      • 레코드: {len(df_output):,}건")
        
        # GCP/AWS 별도 저장
        if self.df_gcp_result is not None and len(self.df_gcp_result) > 0:
            gcp_path = self.output_path.parent / 'gcp_overprovisioned.csv'
            self.df_gcp_result.to_csv(gcp_path, index=False, encoding='utf-8-sig')
            print(f"      • GCP: {gcp_path}")
        
        if self.df_aws_result is not None and len(self.df_aws_result) > 0:
            aws_path = self.output_path.parent / 'aws_overprovisioned.csv'
            self.df_aws_result.to_csv(aws_path, index=False, encoding='utf-8-sig')
            print(f"      • AWS: {aws_path}")
        
        # 통계 JSON 저장
        stats_path = self.output_path.parent / 'detection_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"      • 통계: {stats_path}")
        
        return self
    
    
    def run(self):
        """
        전체 파이프라인 실행
        
        Returns:
            self
        """
        return self.load().process().save()
    
    
    def get_results(self):
        """
        결과 반환
        
        Returns:
            tuple: (통합 결과 DataFrame, 통계 딕셔너리)
        """
        return (self.df_unified, self.stats)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🚀 통합 과다 프로비저닝 탐지기")
    print("="*80)
    print("📌 탐지 방법:")
    print("   • GCP: 직접 임계값 비교 (AvgCPU/Memory < 30%)")
    print("   • AWS: ML Classification (Low 등급 = 과다)")
    print("="*80)
    
    detector = UnifiedOverProvisioningDetector('config/focus_config.yaml')
    detector.run()
    
    df_result, stats = detector.get_results()
    
    print(f"\n✅ 완료!")
    if df_result is not None:
        print(f"   총 과다 프로비저닝: {len(df_result):,}건")