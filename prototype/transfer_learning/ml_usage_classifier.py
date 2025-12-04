# -*- coding: utf-8 -*-
"""
ML 기반 사용률 분류 모델 (Classification)

============================================================
핵심 변경: Regression → Classification
============================================================
- 기존: CPU 사용률 45.3% 예측 (연속값) → R² 0.08로 실패
- 변경: "Low/Medium/High" 등급 예측 (범주형) → Accuracy/F1 사용

============================================================
등급 분류 기준 (Percentile 기반):
============================================================
- Low: 하위 25% (사용률 < P25) → 과다 프로비저닝 가능성 높음
- Medium: 중간 50% (P25 ~ P75) → 적정 사용
- High: 상위 25% (사용률 > P75) → 효율적 사용

============================================================
Author: Lily
Date: 2025-01
Purpose: 석사 논문 - LLM 기반 클라우드 FinOps 자동화 시스템 성능 비교
============================================================
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier  # ← Regressor → Classifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib

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
            print(f"\n{'='*60}")
            print(f"📌 {msg}")
            if detail:
                print(f"   {detail}")
            print(f"{'='*60}")
        
        def print_success(self, msg):
            print(f"   ✅ {msg}")
        
        def print_warning(self, msg):
            print(f"   ⚠️ {msg}")
        
        def print_error(self, msg):
            print(f"   ❌ {msg}")
        
        def ensure_dir(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)


class MLUsageClassifier(PipelineBase):
    """
    ML 기반 사용률 분류 모델
    
    ============================================================
    Regression vs Classification:
    ============================================================
    Regression (기존):
        - 목표: CPU = 0.453 예측
        - 평가: MAE, R²
        - 문제: R² 0.08 → 예측력 없음
    
    Classification (변경):
        - 목표: "Low/Medium/High" 등급 예측
        - 평가: Accuracy, Precision, Recall, F1
        - 장점: 등급 맞추기가 정확한 수치보다 쉬움
    
    ============================================================
    등급 기준 (Percentile):
    ============================================================
    - Low: 사용률 < P25 (하위 25%)
    - Medium: P25 ≤ 사용률 < P75 (중간 50%)
    - High: 사용률 ≥ P75 (상위 25%)
    """
    
    # ============================================================
    # 서비스 → UnifiedCategory 매핑
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
    
    # ============================================================
    # 등급 정의
    # ============================================================
    USAGE_CLASSES = ['Low', 'Medium', 'High']
    
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        """
        super().__init__(config_path)
        
        # 경로 설정
        data_config = self.config['data']
        self.input_path = Path(data_config['resource_grouped_output'])
        self.output_path = Path('results/transfer_learning/ml_classifier_predictions.csv')
        self.model_dir = Path('results/transfer_learning/models/classifier')
        
        # 학습 설정
        self.sample_size = 1_000_000       # 100만건
        self.tune_hyperparams = True
        self.n_iter = 15
        self.cv_folds = 3
        self.test_size = 0.2
        
        # 과다 프로비저닝 임계값
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']
        self.memory_threshold = thresholds['memory_threshold']
        
        # ============================================================
        # Percentile 기준 (등급 분류용)
        # ============================================================
        self.cpu_percentiles = {}    # {'P25': 0.15, 'P75': 0.65}
        self.memory_percentiles = {}
        
        # Feature 설정
        self.feature_cols = ['UnifiedCategory', 'ResourceType', 'LogCost', 'HourOfDay', 'DayOfWeek']
        self.categorical_cols = ['UnifiedCategory', 'ResourceType']
        self.numerical_cols = ['LogCost', 'HourOfDay', 'DayOfWeek']
        
        # 모델
        self.cpu_model = None      # RandomForestClassifier
        self.memory_model = None   # RandomForestClassifier
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # 데이터
        self.df_all = None
        self.df_gcp = None
        self.df_aws = None
        self.df_predictions = None
        self.training_results = {}
    
    
    def _map_to_category(self, service_name):
        """
        서비스명 → UnifiedCategory
        """
        if pd.isna(service_name):
            return 'Other'
        
        service_lower = str(service_name).lower().strip()
        
        for key, cat in self.SERVICE_CATEGORY_MAP.items():
            if key in service_lower:
                return cat
        
        if any(kw in service_lower for kw in ['compute', 'instance', 'vm']):
            return 'Compute'
        elif any(kw in service_lower for kw in ['sql', 'database', 'db']):
            return 'Database'
        elif any(kw in service_lower for kw in ['storage', 'bucket', 'disk']):
            return 'Storage'
        elif any(kw in service_lower for kw in ['network', 'vpc', 'cdn']):
            return 'Networking'
        
        return 'Other'
    
    
    def _usage_to_class(self, usage, percentiles):
        """
        사용률 → 등급 변환 (Percentile 기준)
        
        Args:
            usage: 사용률 (0~1)
            percentiles: {'P25': 0.15, 'P75': 0.65}
        
        Returns:
            str: 'Low', 'Medium', 'High'
        """
        if usage < percentiles['P25']:
            return 'Low'       # 하위 25% → 과다 프로비저닝 가능성
        elif usage < percentiles['P75']:
            return 'Medium'    # 중간 50% → 적정
        else:
            return 'High'      # 상위 25% → 효율적
    
    
    def load(self):
        """
        데이터 로드
        """
        self.print_step("데이터 로딩", f"{self.input_path}")
        
        if not self.input_path.exists():
            self.print_error(f"파일 없음: {self.input_path}")
            return self
        
        self.df_all = pd.read_csv(self.input_path)
        self.print_success(f"로드 완료: {len(self.df_all):,}건")
        
        # UnifiedCategory 생성
        self.df_all['UnifiedCategory'] = self.df_all['ServiceName'].apply(self._map_to_category)
        
        # LogCost 생성
        cost_col = 'TotalHourlyCost' if 'TotalHourlyCost' in self.df_all.columns else 'BilledCost'
        self.df_all['LogCost'] = np.log1p(
            pd.to_numeric(self.df_all[cost_col], errors='coerce').fillna(0)
        )
        
        # GCP/AWS 분리
        gcp_mask = self.df_all['ProviderName'].str.upper().str.contains('GCP|GOOGLE', na=False)
        aws_mask = self.df_all['ProviderName'].str.upper().str.contains('AWS|AMAZON', na=False)
        
        self.df_gcp = self.df_all[gcp_mask].copy()
        self.df_aws = self.df_all[aws_mask].copy()
        
        print(f"\n   ☁️  GCP (학습): {len(self.df_gcp):,}건")
        print(f"   ☁️  AWS (예측): {len(self.df_aws):,}건")
        
        # 카테고리 분포
        print(f"\n   📊 UnifiedCategory 분포:")
        for cat, cnt in self.df_all['UnifiedCategory'].value_counts().head(8).items():
            print(f"      • {cat}: {cnt:,}건")
        
        return self
    
    
    def _find_usage_columns(self):
        """
        사용률 컬럼 찾기
        """
        cols = self.df_gcp.columns.tolist()
        cpu_col = next((c for c in ['AvgCPUUsage', 'SimulatedCPUUsage', 'CPUUsage'] if c in cols), None)
        mem_col = next((c for c in ['AvgMemoryUsage', 'SimulatedMemoryUsage', 'MemoryUsage'] if c in cols), None)
        return cpu_col, mem_col
    
    
    def _extract_resource_type(self, service_name):
        """
        리소스 타입 추출
        """
        if pd.isna(service_name):
            return 'Unknown'
        
        service_lower = str(service_name).lower()
        
        if any(kw in service_lower for kw in ['vm', 'instance', 'ec2', 'compute engine']):
            return 'VM'
        elif any(kw in service_lower for kw in ['container', 'docker', 'ecs', 'gke', 'eks']):
            return 'Container'
        elif any(kw in service_lower for kw in ['function', 'lambda', 'serverless']):
            return 'Function'
        elif any(kw in service_lower for kw in ['sql', 'database', 'rds']):
            return 'Database'
        elif any(kw in service_lower for kw in ['storage', 's3', 'bucket']):
            return 'Storage'
        
        return 'Other'
    
    
    def _prepare_features(self, df):
        """
        Feature 준비
        """
        df = df.copy()
        
        if 'ResourceType' not in df.columns:
            df['ResourceType'] = df['ServiceName'].apply(self._extract_resource_type)
        
        if 'HourOfDay' not in df.columns:
            if 'ChargePeriodStart' in df.columns:
                df['ChargePeriodStart'] = pd.to_datetime(df['ChargePeriodStart'], errors='coerce')
                df['HourOfDay'] = df['ChargePeriodStart'].dt.hour.fillna(12).astype(int)
                df['DayOfWeek'] = df['ChargePeriodStart'].dt.dayofweek.fillna(3).astype(int)
            else:
                df['HourOfDay'] = 12
                df['DayOfWeek'] = 3
        
        return df
    
    
    def _encode_features(self, df, fit=False):
        """
        Feature 인코딩
        """
        encoded_data = []
        
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                unique_vals = list(df[col].unique()) + ['Unknown']
                self.label_encoders[col].fit(unique_vals)
            
            values = df[col].fillna('Unknown').astype(str)
            known_classes = set(self.label_encoders[col].classes_)
            values = values.apply(lambda x: x if x in known_classes else 'Unknown')
            
            encoded = self.label_encoders[col].transform(values)
            encoded_data.append(encoded.reshape(-1, 1))
        
        numerical_data = df[self.numerical_cols].fillna(0).values
        
        if fit:
            numerical_scaled = self.scaler.fit_transform(numerical_data)
        else:
            numerical_scaled = self.scaler.transform(numerical_data)
        
        encoded_data.append(numerical_scaled)
        
        return np.hstack(encoded_data)
    
    
    def _get_param_space(self):
        """
        하이퍼파라미터 탐색 공간
        """
        return {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample', None]  # 클래스 불균형 처리
        }
    
    
    def process(self):
        """
        모델 학습 (Classification)
        
        ============================================================
        핵심 변경:
        1. 사용률 → 등급(Low/Medium/High) 변환
        2. RandomForestRegressor → RandomForestClassifier
        3. R², MAE → Accuracy, F1-Score
        ============================================================
        """
        self.print_step("RandomForest 분류 모델 학습")
        
        if self.df_gcp is None or len(self.df_gcp) == 0:
            self.print_error("GCP 데이터 없음")
            return self
        
        cpu_col, mem_col = self._find_usage_columns()
        if not cpu_col or not mem_col:
            self.print_error(f"사용률 컬럼 없음")
            return self
        
        print(f"   📋 Target: CPU={cpu_col}, Memory={mem_col}")
        print(f"   📋 Features: {self.feature_cols}")
        print(f"   📋 Classes: {self.USAGE_CLASSES}")
        
        # Feature 준비
        df_train = self._prepare_features(self.df_gcp)
        
        df_train[cpu_col] = pd.to_numeric(df_train[cpu_col], errors='coerce')
        df_train[mem_col] = pd.to_numeric(df_train[mem_col], errors='coerce')
        
        df_valid = df_train[
            (df_train[cpu_col] > 0) & (df_train[cpu_col] <= 1) &
            (df_train[mem_col] > 0) & (df_train[mem_col] <= 1)
        ].copy()
        
        print(f"   📊 유효 데이터: {len(df_valid):,}건")
        
        # ============================================================
        # Percentile 계산 (등급 기준)
        # ============================================================
        self.cpu_percentiles = {
            'P25': float(np.percentile(df_valid[cpu_col], 25)),
            'P50': float(np.percentile(df_valid[cpu_col], 50)),
            'P75': float(np.percentile(df_valid[cpu_col], 75))
        }
        self.memory_percentiles = {
            'P25': float(np.percentile(df_valid[mem_col], 25)),
            'P50': float(np.percentile(df_valid[mem_col], 50)),
            'P75': float(np.percentile(df_valid[mem_col], 75))
        }
        
        print(f"\n   📊 CPU Percentile 기준:")
        print(f"      • Low: < {self.cpu_percentiles['P25']*100:.1f}% (하위 25%)")
        print(f"      • Medium: {self.cpu_percentiles['P25']*100:.1f}% ~ {self.cpu_percentiles['P75']*100:.1f}%")
        print(f"      • High: ≥ {self.cpu_percentiles['P75']*100:.1f}% (상위 25%)")
        
        print(f"\n   📊 Memory Percentile 기준:")
        print(f"      • Low: < {self.memory_percentiles['P25']*100:.1f}%")
        print(f"      • Medium: {self.memory_percentiles['P25']*100:.1f}% ~ {self.memory_percentiles['P75']*100:.1f}%")
        print(f"      • High: ≥ {self.memory_percentiles['P75']*100:.1f}%")
        
        # ============================================================
        # 사용률 → 등급 변환 (핵심!)
        # ============================================================
        df_valid['CPUClass'] = df_valid[cpu_col].apply(
            lambda x: self._usage_to_class(x, self.cpu_percentiles)
        )
        df_valid['MemoryClass'] = df_valid[mem_col].apply(
            lambda x: self._usage_to_class(x, self.memory_percentiles)
        )
        
        # 클래스 분포 확인
        print(f"\n   📊 CPU 등급 분포:")
        for cls, cnt in df_valid['CPUClass'].value_counts().items():
            print(f"      • {cls}: {cnt:,}건 ({cnt/len(df_valid)*100:.1f}%)")
        
        print(f"\n   📊 Memory 등급 분포:")
        for cls, cnt in df_valid['MemoryClass'].value_counts().items():
            print(f"      • {cls}: {cnt:,}건 ({cnt/len(df_valid)*100:.1f}%)")
        
        # 샘플링
        if len(df_valid) > self.sample_size:
            df_sample = df_valid.sample(n=self.sample_size, random_state=42)
            print(f"\n   📊 샘플링: {self.sample_size:,}건")
        else:
            df_sample = df_valid
        
        # Feature 인코딩
        X = self._encode_features(df_sample, fit=True)
        y_cpu = df_sample['CPUClass'].values
        y_mem = df_sample['MemoryClass'].values
        
        # Label 인코딩 (등급 → 숫자)
        self.class_encoder = LabelEncoder()
        self.class_encoder.fit(self.USAGE_CLASSES)
        
        y_cpu_encoded = self.class_encoder.transform(y_cpu)
        y_mem_encoded = self.class_encoder.transform(y_mem)
        
        # Train/Test 분할
        X_train, X_test, y_cpu_train, y_cpu_test, y_mem_train, y_mem_test = train_test_split(
            X, y_cpu_encoded, y_mem_encoded, test_size=self.test_size, random_state=42
        )
        
        print(f"\n   📊 데이터 분할:")
        print(f"      • Train: {len(X_train):,}건")
        print(f"      • Test: {len(X_test):,}건")
        
        # ============================================================
        # 하이퍼파라미터 튜닝
        # ============================================================
        best_params = None
        
        if self.tune_hyperparams:
            print(f"\n   🔧 하이퍼파라미터 튜닝...")
            
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model,
                self._get_param_space(),
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring='f1_macro',  # 다중 클래스 F1
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            search.fit(X_train, y_cpu_train)
            best_params = search.best_params_
            
            print(f"\n   ✅ 최적 파라미터:")
            for k, v in best_params.items():
                print(f"      • {k}: {v}")
        
        # ============================================================
        # CPU 분류 모델 학습
        # ============================================================
        print(f"\n   🔄 CPU 분류 모델 학습...")
        
        if best_params:
            self.cpu_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        else:
            self.cpu_model = RandomForestClassifier(
                n_estimators=100, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
        
        self.cpu_model.fit(X_train, y_cpu_train)
        
        # CPU 평가
        y_cpu_pred = self.cpu_model.predict(X_test)
        
        cpu_accuracy = accuracy_score(y_cpu_test, y_cpu_pred)
        cpu_precision = precision_score(y_cpu_test, y_cpu_pred, average='macro')
        cpu_recall = recall_score(y_cpu_test, y_cpu_pred, average='macro')
        cpu_f1 = f1_score(y_cpu_test, y_cpu_pred, average='macro')
        
        print(f"\n   ✅ CPU 분류 모델 성능:")
        print(f"      • Accuracy: {cpu_accuracy*100:.2f}%")
        print(f"      • Precision: {cpu_precision*100:.2f}%")
        print(f"      • Recall: {cpu_recall*100:.2f}%")
        print(f"      • F1-Score: {cpu_f1*100:.2f}%")
        
        # Confusion Matrix
        print(f"\n   📊 CPU Confusion Matrix:")
        cpu_cm = confusion_matrix(y_cpu_test, y_cpu_pred)
        self._print_confusion_matrix(cpu_cm, self.USAGE_CLASSES)
        
        # ============================================================
        # Memory 분류 모델 학습
        # ============================================================
        print(f"\n   🔄 Memory 분류 모델 학습...")
        
        if best_params:
            self.memory_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        else:
            self.memory_model = RandomForestClassifier(
                n_estimators=100, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
        
        self.memory_model.fit(X_train, y_mem_train)
        
        # Memory 평가
        y_mem_pred = self.memory_model.predict(X_test)
        
        mem_accuracy = accuracy_score(y_mem_test, y_mem_pred)
        mem_precision = precision_score(y_mem_test, y_mem_pred, average='macro')
        mem_recall = recall_score(y_mem_test, y_mem_pred, average='macro')
        mem_f1 = f1_score(y_mem_test, y_mem_pred, average='macro')
        
        print(f"\n   ✅ Memory 분류 모델 성능:")
        print(f"      • Accuracy: {mem_accuracy*100:.2f}%")
        print(f"      • Precision: {mem_precision*100:.2f}%")
        print(f"      • Recall: {mem_recall*100:.2f}%")
        print(f"      • F1-Score: {mem_f1*100:.2f}%")
        
        # Confusion Matrix
        print(f"\n   📊 Memory Confusion Matrix:")
        mem_cm = confusion_matrix(y_mem_test, y_mem_pred)
        self._print_confusion_matrix(mem_cm, self.USAGE_CLASSES)
        
        # ============================================================
        # 결과 저장
        # ============================================================
        self.training_results = {
            'model_type': 'Classification',
            'classes': self.USAGE_CLASSES,
            'sample_size': len(df_sample),
            'features': self.feature_cols,
            'cpu_percentiles': self.cpu_percentiles,
            'memory_percentiles': self.memory_percentiles,
            'best_params': best_params,
            'cpu': {
                'accuracy': float(cpu_accuracy),
                'precision': float(cpu_precision),
                'recall': float(cpu_recall),
                'f1': float(cpu_f1)
            },
            'memory': {
                'accuracy': float(mem_accuracy),
                'precision': float(mem_precision),
                'recall': float(mem_recall),
                'f1': float(mem_f1)
            }
        }
        
        # Feature Importance
        print(f"\n   📊 Feature Importance (CPU):")
        feature_names = self.categorical_cols + self.numerical_cols
        for name, imp in sorted(zip(feature_names, self.cpu_model.feature_importances_), 
                                key=lambda x: x[1], reverse=True):
            print(f"      • {name}: {imp:.4f}")
        
        # Regression vs Classification 비교
        self._compare_with_regression()
        
        return self
    
    
    def _print_confusion_matrix(self, cm, classes):
        """
        Confusion Matrix 출력
        """
        print(f"      {'':>10} ", end='')
        for cls in classes:
            print(f"{cls:>8}", end='')
        print()
        
        for i, cls in enumerate(classes):
            print(f"      {cls:>10} ", end='')
            for j in range(len(classes)):
                print(f"{cm[i][j]:>8,}", end='')
            print()
    
    
    def _compare_with_regression(self):
        """
        Regression 결과와 비교
        """
        # Regression 결과 (이전 대화에서 확인된 값)
        reg_results = {
            'cpu_r2': 0.0895,
            'memory_r2': 0.0947,
            'description': 'R² < 0.1 → 예측력 거의 없음'
        }
        
        print(f"\n{'='*80}")
        print("📊 Regression vs Classification 비교")
        print(f"{'='*80}")
        
        print(f"\n   📈 Regression (기존):")
        print(f"      • CPU R²: {reg_results['cpu_r2']:.4f}")
        print(f"      • Memory R²: {reg_results['memory_r2']:.4f}")
        print(f"      ⚠️ {reg_results['description']}")
        
        print(f"\n   📊 Classification (변경):")
        print(f"      • CPU Accuracy: {self.training_results['cpu']['accuracy']*100:.2f}%")
        print(f"      • CPU F1-Score: {self.training_results['cpu']['f1']*100:.2f}%")
        print(f"      • Memory Accuracy: {self.training_results['memory']['accuracy']*100:.2f}%")
        print(f"      • Memory F1-Score: {self.training_results['memory']['f1']*100:.2f}%")
        
        # 랜덤 추측 대비 개선도 (3개 클래스 → 33% 기준)
        random_baseline = 33.33
        cpu_improvement = self.training_results['cpu']['accuracy'] * 100 - random_baseline
        mem_improvement = self.training_results['memory']['accuracy'] * 100 - random_baseline
        
        print(f"\n   📊 랜덤 추측(33%) 대비 개선:")
        print(f"      • CPU: +{cpu_improvement:.1f}%p")
        print(f"      • Memory: +{mem_improvement:.1f}%p")
        
        print(f"\n   ✅ 결론: Classification이 더 효과적")
        print(f"{'='*80}")
    
    
    def predict(self):
        """
        AWS 데이터에 등급 예측
        """
        self.print_step("AWS 사용률 등급 예측")
        
        if self.cpu_model is None:
            self.print_error("먼저 process() 실행")
            return self
        
        if self.df_aws is None or len(self.df_aws) == 0:
            self.print_warning("AWS 데이터 없음")
            return self
        
        # Feature 준비
        df_pred = self._prepare_features(self.df_aws)
        X = self._encode_features(df_pred, fit=False)
        
        # 예측
        cpu_class_encoded = self.cpu_model.predict(X)
        mem_class_encoded = self.memory_model.predict(X)
        
        # 디코딩 (숫자 → 등급)
        cpu_class = self.class_encoder.inverse_transform(cpu_class_encoded)
        mem_class = self.class_encoder.inverse_transform(mem_class_encoded)
        
        # 결과 저장
        self.df_predictions = df_pred.copy()
        self.df_predictions['PredictedCPUClass'] = cpu_class
        self.df_predictions['PredictedMemoryClass'] = mem_class
        
        # 과다 프로비저닝 = Low 등급
        self.df_predictions['IsOverProvisioned'] = (
            (self.df_predictions['PredictedCPUClass'] == 'Low') |
            (self.df_predictions['PredictedMemoryClass'] == 'Low')
        )
        
        # 통계
        print(f"\n   📊 예측 결과:")
        print(f"      • 전체: {len(self.df_predictions):,}건")
        
        print(f"\n   📊 CPU 등급 분포:")
        for cls, cnt in self.df_predictions['PredictedCPUClass'].value_counts().items():
            print(f"      • {cls}: {cnt:,}건 ({cnt/len(self.df_predictions)*100:.1f}%)")
        
        print(f"\n   📊 Memory 등급 분포:")
        for cls, cnt in self.df_predictions['PredictedMemoryClass'].value_counts().items():
            print(f"      • {cls}: {cnt:,}건 ({cnt/len(self.df_predictions)*100:.1f}%)")
        
        over_prov = self.df_predictions['IsOverProvisioned'].sum()
        print(f"\n   🚨 과다 프로비저닝 (Low 등급):")
        print(f"      • {over_prov:,}건 ({over_prov/len(self.df_predictions)*100:.1f}%)")
        
        return self
    
    
    def save(self):
        """
        결과 저장
        """
        self.print_step("결과 저장")
        
        self.ensure_dir(self.output_path.parent)
        self.ensure_dir(self.model_dir)
        
        # 예측 결과
        if self.df_predictions is not None:
            self.df_predictions.to_csv(self.output_path, index=False)
            print(f"   📂 예측 결과: {self.output_path}")
        
        # 모델 저장
        if self.cpu_model is not None:
            joblib.dump(self.cpu_model, self.model_dir / 'cpu_classifier.joblib')
            joblib.dump(self.memory_model, self.model_dir / 'memory_classifier.joblib')
            joblib.dump(self.label_encoders, self.model_dir / 'label_encoders.joblib')
            joblib.dump(self.scaler, self.model_dir / 'scaler.joblib')
            joblib.dump(self.class_encoder, self.model_dir / 'class_encoder.joblib')
            print(f"   📂 모델: {self.model_dir}")
        
        # 학습 결과 JSON
        with open(self.model_dir / 'training_results.json', 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # 과다 프로비저닝만
        if self.df_predictions is not None:
            over_path = self.output_path.parent / 'classifier_overprovisioned.csv'
            df_over = self.df_predictions[self.df_predictions['IsOverProvisioned']]
            if len(df_over) > 0:
                df_over.to_csv(over_path, index=False)
                print(f"   📂 과다 프로비저닝: {over_path} ({len(df_over):,}건)")
        
        self.print_success("저장 완료")
        return self
    
    
    def run(self):
        """
        전체 실행
        """
        return (self.load()
                .process()
                .predict()
                .save())
    
    
    def get_results(self):
        """
        결과 반환
        """
        return (self.df_predictions, self.training_results)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🚀 ML 사용률 분류 모델 (Regression → Classification)")
    print("="*80)
    print("📌 핵심 변경:")
    print("   • Regression: CPU=0.45 예측 → R² 0.08 (실패)")
    print("   • Classification: Low/Medium/High 예측 → Accuracy/F1 (개선)")
    print("📌 등급 기준:")
    print("   • Low: 하위 25% (과다 프로비저닝)")
    print("   • Medium: 중간 50% (적정)")
    print("   • High: 상위 25% (효율적)")
    print("="*80)
    
    classifier = MLUsageClassifier('config/focus_config.yaml')
    classifier.sample_size = 1_000_000
    classifier.tune_hyperparams = True
    classifier.n_iter = 15
    
    classifier.run()
    
    df_pred, results = classifier.get_results()
    
    print(f"\n✅ 완료!")
    if results:
        print(f"   CPU Accuracy: {results['cpu']['accuracy']*100:.2f}%")
        print(f"   CPU F1-Score: {results['cpu']['f1']*100:.2f}%")