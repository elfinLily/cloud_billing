# -*- coding: utf-8 -*-
"""
XGBoost 기반 사용률 예측 모델

============================================================
데이터 흐름:
============================================================
1. resource_grouped.csv 로드 (TimeNormalizer → ResourceGrouper 결과)
2. ProviderName으로 GCP/AWS 분리
3. GCP 데이터로 학습 (AvgCPUUsage, AvgMemoryUsage 있음)
4. AWS 데이터에 적용하여 CPU/Memory 사용률 예측
5. 과다 프로비저닝 탐지 (사용률 < 30%)

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
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# XGBoost 및 sklearn 임포트
# ============================================================
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# ============================================================
# 프로젝트 경로 설정
# ============================================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))

from pipeline_base import PipelineBase


class XGBUsagePredictor(PipelineBase):
    """
    XGBoost 기반 사용률 예측 클래스
    
    ============================================================
    주요 기능:
    ============================================================
    1. resource_grouped.csv에서 GCP/AWS 분리
    2. GCP 데이터로 XGBoost 모델 학습
    3. RandomizedSearchCV로 하이퍼파라미터 튜닝
    4. AWS 데이터에 적용하여 CPU/Memory 사용률 예측
    5. 과다 프로비저닝 탐지 (사용률 < 30%)
    6. RandomForest 결과와 비교
    
    ============================================================
    RandomForest 대비 XGBoost 장점:
    ============================================================
    - Gradient Boosting으로 순차적 오차 보정
    - 정형 데이터에서 일반적으로 더 높은 성능
    - 정규화(L1/L2) 내장으로 과적합 방지
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화 - config에서 경로와 임계값 로드
        
        Args:
            config_path (str): 설정 파일 경로
        """
        super().__init__(config_path)
        
        # --------------------------------------------------------
        # 경로 설정 (config 기반, 하드코딩 금지)
        # --------------------------------------------------------
        data_config = self.config['data']
        
        # 입력: resource_grouped.csv (ResourceGrouper 결과)
        self.input_path = Path(data_config['resource_grouped_output'])
        
        # 출력 경로
        self.model_output_dir = Path('results/transfer_learning/models/xgboost')
        self.result_output_path = Path('results/transfer_learning/xgb_predictions.csv')
        
        # --------------------------------------------------------
        # 임계값 (config 기반)
        # --------------------------------------------------------
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']      # 0.30
        self.memory_threshold = thresholds['memory_threshold']  # 0.30
        
        # --------------------------------------------------------
        # 모델 및 전처리기
        # --------------------------------------------------------
        self.cpu_model = None
        self.memory_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # --------------------------------------------------------
        # 데이터 저장용
        # --------------------------------------------------------
        self.df_all = None       # 전체 데이터
        self.df_gcp = None       # GCP 데이터 (학습용)
        self.df_aws = None       # AWS 데이터 (예측 대상)
        self.df_predictions = None
        self.training_results = None
        
        # --------------------------------------------------------
        # Feature 컬럼 정의 (이전 대화에서 확정된 5개)
        # --------------------------------------------------------
        self.feature_cols = []
        self.categorical_cols = ['ServiceName', 'ResourceType']
        self.numerical_cols = ['TotalHourlyCost', 'HourOfDay', 'DayOfWeek']
        
        # --------------------------------------------------------
        # 하이퍼파라미터 튜닝 설정
        # --------------------------------------------------------
        self.tune_hyperparams = True
        self.sample_size = 5_000_000  # 튜닝용 샘플 크기
        self.n_iter = 15              # RandomizedSearchCV 반복 횟수
        self.cv_folds = 3             # Cross-validation 폴드 수
    
    
    def load(self):
        """
        resource_grouped.csv 로드 및 GCP/AWS 분리
        
        --------------------------------------------------------
        기능:
        - resource_grouped.csv 파일 로드
        - ProviderName 기준으로 GCP/AWS 분리
        - GCP: 학습용 (AvgCPUUsage, AvgMemoryUsage 있음)
        - AWS: 예측 대상 (사용률 없음)
        --------------------------------------------------------
        
        Returns:
            self: 메서드 체이닝용
        """
        self.print_step("데이터 로딩", f"{self.input_path}")
        
        if not self.input_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.input_path}")
            self.print_warning("먼저 ResourceGrouper를 실행하세요.")
            return self
        
        # CSV 로드
        self.df_all = pd.read_csv(self.input_path)
        
        self.print_success("로드 완료")
        print(f"   📊 전체 레코드: {len(self.df_all):,}건")
        print(f"   📋 컬럼: {list(self.df_all.columns)}")
        
        # --------------------------------------------------------
        # ProviderName으로 GCP/AWS 분리
        # --------------------------------------------------------
        if 'ProviderName' not in self.df_all.columns:
            self.print_error("ProviderName 컬럼이 없습니다.")
            return self
        
        # GCP 데이터 필터링
        gcp_mask = self.df_all['ProviderName'].str.upper().str.contains('GCP|GOOGLE', na=False)
        self.df_gcp = self.df_all[gcp_mask].copy()
        
        # AWS 데이터 필터링
        aws_mask = self.df_all['ProviderName'].str.upper().str.contains('AWS|AMAZON', na=False)
        self.df_aws = self.df_all[aws_mask].copy()
        
        print(f"\n   ☁️  Provider별 분리:")
        print(f"      • GCP (학습용): {len(self.df_gcp):,}건")
        print(f"      • AWS (예측 대상): {len(self.df_aws):,}건")
        
        # CPU/Memory 컬럼 확인
        cpu_col = self._find_column(['AvgCPUUsage', 'SimulatedCPUUsage', 'CPUUsage'])
        mem_col = self._find_column(['AvgMemoryUsage', 'SimulatedMemoryUsage', 'MemoryUsage'])
        
        print(f"\n   📋 사용률 컬럼:")
        print(f"      • CPU: {cpu_col}")
        print(f"      • Memory: {mem_col}")
        
        return self
    
    
    def _find_column(self, candidates):
        """
        후보 컬럼 중 존재하는 컬럼 찾기
        
        Args:
            candidates (list): 후보 컬럼명 리스트
        
        Returns:
            str or None: 찾은 컬럼명
        """
        if self.df_all is None:
            return None
        
        for col in candidates:
            if col in self.df_all.columns:
                return col
        return None
    
    
    def _extract_features(self, df, is_training=True):
        """
        Feature 추출 및 전처리
        
        --------------------------------------------------------
        Feature 5개:
        1. ServiceName (범주형)
        2. ResourceType (범주형) - 없으면 ServiceName에서 추출
        3. TotalHourlyCost (수치형)
        4. HourOfDay (수치형)
        5. DayOfWeek (수치형)
        --------------------------------------------------------
        
        Args:
            df (DataFrame): 원본 데이터
            is_training (bool): 학습 데이터 여부
        
        Returns:
            DataFrame: Feature DataFrame
        """
        features = pd.DataFrame()
        
        # 1. ServiceName
        if 'ServiceName' in df.columns:
            features['ServiceName'] = df['ServiceName'].fillna('Unknown')
        else:
            features['ServiceName'] = 'Unknown'
        
        # 2. ResourceType (없으면 ServiceName에서 추출)
        if 'ResourceType' in df.columns:
            features['ResourceType'] = df['ResourceType'].fillna('Other')
        else:
            features['ResourceType'] = features['ServiceName'].apply(self._extract_resource_type)
        
        # 3. TotalHourlyCost
        if 'TotalHourlyCost' in df.columns:
            features['TotalHourlyCost'] = pd.to_numeric(
                df['TotalHourlyCost'], errors='coerce'
            ).fillna(0)
        elif 'BilledCost' in df.columns:
            features['TotalHourlyCost'] = pd.to_numeric(
                df['BilledCost'], errors='coerce'
            ).fillna(0)
        else:
            features['TotalHourlyCost'] = 0
        
        # 4 & 5. HourOfDay, DayOfWeek
        time_col = self._find_time_column(df)
        if time_col:
            try:
                dt = pd.to_datetime(df[time_col], errors='coerce')
                features['HourOfDay'] = dt.dt.hour.fillna(12).astype(int)
                features['DayOfWeek'] = dt.dt.dayofweek.fillna(3).astype(int)
            except:
                features['HourOfDay'] = 12
                features['DayOfWeek'] = 3
        else:
            features['HourOfDay'] = 12
            features['DayOfWeek'] = 3
        
        # Target 컬럼 (학습 시에만)
        if is_training:
            # CPU 사용률
            cpu_col = self._find_column(['AvgCPUUsage', 'SimulatedCPUUsage', 'CPUUsage'])
            if cpu_col and cpu_col in df.columns:
                features['CPUUsage'] = pd.to_numeric(df[cpu_col], errors='coerce')
            
            # Memory 사용률
            mem_col = self._find_column(['AvgMemoryUsage', 'SimulatedMemoryUsage', 'MemoryUsage'])
            if mem_col and mem_col in df.columns:
                features['MemoryUsage'] = pd.to_numeric(df[mem_col], errors='coerce')
        
        return features
    
    
    def _find_time_column(self, df):
        """
        시간 관련 컬럼 찾기
        
        Args:
            df (DataFrame): 데이터프레임
        
        Returns:
            str or None: 시간 컬럼명
        """
        candidates = ['HourlyTimestamp', 'ChargePeriodStart', 'Date', 'Timestamp', 'Hour']
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    
    def _extract_resource_type(self, service_name):
        """
        서비스명에서 리소스 타입 추출
        
        Args:
            service_name: 서비스명
        
        Returns:
            str: 리소스 타입 (VM, Container, Function 등)
        """
        if pd.isna(service_name):
            return 'Other'
        
        service_lower = str(service_name).lower()
        
        if any(kw in service_lower for kw in ['vm', 'instance', 'engine', 'compute', 'ec2']):
            return 'VM'
        elif any(kw in service_lower for kw in ['container', 'kubernetes', 'ecs', 'eks', 'gke']):
            return 'Container'
        elif any(kw in service_lower for kw in ['function', 'lambda', 'cloud functions']):
            return 'Function'
        elif any(kw in service_lower for kw in ['storage', 'bucket', 's3', 'gcs']):
            return 'ObjectStorage'
        elif any(kw in service_lower for kw in ['disk', 'volume', 'ebs']):
            return 'BlockStorage'
        elif any(kw in service_lower for kw in ['sql', 'database', 'rds', 'spanner', 'dynamodb']):
            return 'Database'
        elif any(kw in service_lower for kw in ['network', 'vpc', 'load balancer', 'cdn']):
            return 'Network'
        else:
            return 'Other'
    
    
    def _encode_features(self, features, fit=True):
        """
        카테고리 Feature 인코딩 + 수치형 정규화
        
        --------------------------------------------------------
        처리:
        - LabelEncoder: 범주형 → 정수
        - StandardScaler: 수치형 정규화
        - 학습 시 없던 카테고리 → 첫 번째 클래스로 대체
        --------------------------------------------------------
        
        Args:
            features (DataFrame): Feature DataFrame
            fit (bool): 인코더 학습 여부
        
        Returns:
            numpy.ndarray: 인코딩된 Feature 배열
        """
        df_encoded = features.copy()
        
        # 카테고리 컬럼 인코딩
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    # 학습 시 없던 카테고리 → 첫 번째 클래스로 대체
                    le = self.label_encoders.get(col)
                    if le:
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ 
                            else le.transform([le.classes_[0]])[0]
                        )
        
        # Feature 컬럼 선택
        feature_cols = self.categorical_cols + self.numerical_cols
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        
        X = df_encoded[feature_cols].values.astype(np.float32)
        
        # 수치형 정규화
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        self.feature_cols = feature_cols
        
        return X
    
    
    def _get_xgb_param_space(self):
        """
        XGBoost 하이퍼파라미터 탐색 공간 정의
        
        --------------------------------------------------------
        탐색 파라미터:
        - n_estimators: 트리 개수 (50~300)
        - max_depth: 트리 깊이 (3~15)
        - learning_rate: 학습률 (0.01~0.3)
        - subsample: 행 샘플링 비율 (0.6~1.0)
        - colsample_bytree: 컬럼 샘플링 비율 (0.6~1.0)
        - reg_alpha: L1 정규화 (0~1)
        - reg_lambda: L2 정규화 (0~1)
        --------------------------------------------------------
        
        Returns:
            dict: 파라미터 탐색 공간
        """
        param_space = {
            'n_estimators': [50, 100, 150, 200, 300],
            'max_depth': [3, 5, 7, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 7]
        }
        return param_space
    
    
    def _tune_model(self, X_train, y_train, target_name='CPU'):
        """
        RandomizedSearchCV로 하이퍼파라미터 튜닝
        
        --------------------------------------------------------
        프로세스:
        1. 대용량 데이터 샘플링 (500만 건)
        2. RandomizedSearchCV 실행
        3. 최적 파라미터로 전체 데이터 재학습
        --------------------------------------------------------
        
        Args:
            X_train (ndarray): 학습 Feature
            y_train (ndarray): 학습 Target
            target_name (str): 타겟 이름 (CPU/Memory)
        
        Returns:
            tuple: (최적 모델, 최적 파라미터, 최적 점수)
        """
        print(f"\n   🔧 {target_name} 모델 하이퍼파라미터 튜닝...")
        
        # 샘플링 (대용량 데이터 처리)
        if len(X_train) > self.sample_size:
            indices = np.random.choice(len(X_train), self.sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
            print(f"      📊 샘플링: {len(X_train):,} → {len(X_sample):,}건")
        else:
            X_sample = X_train
            y_sample = y_train
        
        # 기본 모델
        base_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # RandomizedSearchCV
        param_space = self._get_xgb_param_space()
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print(f"      🔄 RandomizedSearchCV 실행 (n_iter={self.n_iter}, cv={self.cv_folds})")
        search.fit(X_sample, y_sample)
        
        best_params = search.best_params_
        best_score = -search.best_score_  # MAE (양수로 변환)
        
        print(f"\n      ✅ {target_name} 최적 파라미터:")
        for key, val in best_params.items():
            print(f"         • {key}: {val}")
        print(f"      📊 CV MAE: {best_score*100:.2f}%")
        
        # 전체 데이터로 최적 모델 재학습
        print(f"\n      🔄 전체 데이터로 재학습 중...")
        best_model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **best_params
        )
        best_model.fit(X_train, y_train)
        
        return best_model, best_params, best_score
    
    
    def process(self):
        """
        XGBoost 모델 학습 (메인 프로세스)
        
        --------------------------------------------------------
        프로세스:
        1. GCP 데이터에서 Feature 추출
        2. 결측치 제거 및 유효 데이터 필터링
        3. Train/Test 분할 (80:20)
        4. 하이퍼파라미터 튜닝 (선택적)
        5. CPU/Memory 모델 학습
        6. 평가 및 결과 저장
        --------------------------------------------------------
        
        Returns:
            self: 메서드 체이닝용
        """
        self.print_step("XGBoost 모델 학습 (GCP 데이터)")
        
        if self.df_gcp is None or len(self.df_gcp) == 0:
            self.print_error("GCP 학습 데이터가 없습니다. load()를 먼저 실행하세요.")
            return self
        
        # Feature 추출
        print(f"\n   📊 Feature 추출 중...")
        features = self._extract_features(self.df_gcp, is_training=True)
        
        print(f"      • 추출된 Feature: {list(features.columns)}")
        
        # CPU/Memory 컬럼 확인
        if 'CPUUsage' not in features.columns or 'MemoryUsage' not in features.columns:
            self.print_error("CPUUsage 또는 MemoryUsage 컬럼을 찾을 수 없습니다.")
            print(f"      • 사용 가능한 컬럼: {list(features.columns)}")
            return self
        
        # 결측치 제거 및 유효 범위 필터링 (0 < 사용률 <= 1)
        features_clean = features.dropna(subset=['CPUUsage', 'MemoryUsage'])
        features_clean = features_clean[
            (features_clean['CPUUsage'] > 0) & 
            (features_clean['CPUUsage'] <= 1) &
            (features_clean['MemoryUsage'] > 0) & 
            (features_clean['MemoryUsage'] <= 1)
        ]
        
        print(f"\n   📊 학습 데이터:")
        print(f"      • 원본: {len(features):,}건")
        print(f"      • 유효: {len(features_clean):,}건")
        
        if len(features_clean) == 0:
            self.print_error("유효한 학습 데이터가 없습니다.")
            return self
        
        # Feature 인코딩
        X = self._encode_features(features_clean, fit=True)
        y_cpu = features_clean['CPUUsage'].values
        y_memory = features_clean['MemoryUsage'].values
        
        print(f"   📋 Feature 컬럼: {self.feature_cols}")
        print(f"   📊 X shape: {X.shape}")
        
        # Train/Test 분할 (80:20)
        X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(
            X, y_cpu, test_size=0.2, random_state=42
        )
        _, _, y_mem_train, y_mem_test = train_test_split(
            X, y_memory, test_size=0.2, random_state=42
        )
        
        print(f"\n   📊 Train/Test 분할:")
        print(f"      • Train: {len(X_train):,}건")
        print(f"      • Test: {len(X_test):,}건")
        
        # ============================================================
        # CPU 모델 학습
        # ============================================================
        if self.tune_hyperparams:
            self.cpu_model, cpu_best_params, _ = self._tune_model(
                X_train, y_cpu_train, target_name='CPU'
            )
        else:
            print(f"\n   🤖 CPU 모델 학습 중 (기본 파라미터)...")
            self.cpu_model = XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            self.cpu_model.fit(X_train, y_cpu_train)
            cpu_best_params = {}
        
        # CPU 모델 평가
        y_cpu_pred = self.cpu_model.predict(X_test)
        cpu_mae = mean_absolute_error(y_cpu_test, y_cpu_pred)
        cpu_rmse = np.sqrt(mean_squared_error(y_cpu_test, y_cpu_pred))
        cpu_r2 = r2_score(y_cpu_test, y_cpu_pred)
        
        print(f"\n   📊 CPU 모델 성능:")
        print(f"      • MAE: {cpu_mae*100:.2f}%")
        print(f"      • RMSE: {cpu_rmse*100:.2f}%")
        print(f"      • R²: {cpu_r2:.4f}")
        
        # ============================================================
        # Memory 모델 학습
        # ============================================================
        if self.tune_hyperparams:
            self.memory_model, mem_best_params, _ = self._tune_model(
                X_train, y_mem_train, target_name='Memory'
            )
        else:
            print(f"\n   🤖 Memory 모델 학습 중 (기본 파라미터)...")
            self.memory_model = XGBRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            self.memory_model.fit(X_train, y_mem_train)
            mem_best_params = {}
        
        # Memory 모델 평가
        y_mem_pred = self.memory_model.predict(X_test)
        mem_mae = mean_absolute_error(y_mem_test, y_mem_pred)
        mem_rmse = np.sqrt(mean_squared_error(y_mem_test, y_mem_pred))
        mem_r2 = r2_score(y_mem_test, y_mem_pred)
        
        print(f"\n   📊 Memory 모델 성능:")
        print(f"      • MAE: {mem_mae*100:.2f}%")
        print(f"      • RMSE: {mem_rmse*100:.2f}%")
        print(f"      • R²: {mem_r2:.4f}")
        
        # ============================================================
        # 학습 결과 저장
        # ============================================================
        self.training_results = {
            'model_type': 'XGBoost',
            'timestamp': datetime.now().isoformat(),
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'feature_cols': self.feature_cols,
            'cpu_mae': float(cpu_mae),
            'cpu_rmse': float(cpu_rmse),
            'cpu_r2': float(cpu_r2),
            'cpu_best_params': cpu_best_params if self.tune_hyperparams else {},
            'memory_mae': float(mem_mae),
            'memory_rmse': float(mem_rmse),
            'memory_r2': float(mem_r2),
            'memory_best_params': mem_best_params if self.tune_hyperparams else {},
            'tuning_config': {
                'tune_hyperparams': self.tune_hyperparams,
                'sample_size': self.sample_size,
                'n_iter': self.n_iter,
                'cv_folds': self.cv_folds
            }
        }
        
        # 결과 요약 출력
        self._print_training_summary()
        
        # 모델 저장
        self._save_models()
        
        return self
    
    
    def _print_training_summary(self):
        """
        학습 결과 요약 출력 (RandomForest 비교 포함)
        """
        print(f"\n{'='*100}")
        print("📊 XGBoost 학습 결과 요약")
        print(f"{'='*100}")
        
        r = self.training_results
        
        print(f"\n   🔢 데이터:")
        print(f"      • Train: {r['train_size']:,}건")
        print(f"      • Test: {r['test_size']:,}건")
        
        print(f"\n   📊 CPU 모델:")
        print(f"      • MAE: {r['cpu_mae']*100:.2f}%")
        print(f"      • R²: {r['cpu_r2']:.4f}")
        
        print(f"\n   📊 Memory 모델:")
        print(f"      • MAE: {r['memory_mae']*100:.2f}%")
        print(f"      • R²: {r['memory_r2']:.4f}")
        
        # RandomForest 결과와 비교 (이전 대화에서 확인된 결과)
        rf_cpu_mae = 0.2369
        rf_cpu_r2 = 0.0895
        rf_mem_mae = 0.2371
        rf_mem_r2 = 0.0947
        
        print(f"\n   📈 RandomForest 대비 비교:")
        
        # MAE 개선 (낮을수록 좋음 → 양수면 개선)
        cpu_mae_diff = (rf_cpu_mae - r['cpu_mae']) * 100
        mem_mae_diff = (rf_mem_mae - r['memory_mae']) * 100
        
        # R² 개선 (높을수록 좋음 → 양수면 개선)
        cpu_r2_diff = r['cpu_r2'] - rf_cpu_r2
        mem_r2_diff = r['memory_r2'] - rf_mem_r2
        
        cpu_mae_sign = "+" if cpu_mae_diff > 0 else ""
        cpu_r2_sign = "+" if cpu_r2_diff > 0 else ""
        mem_mae_sign = "+" if mem_mae_diff > 0 else ""
        mem_r2_sign = "+" if mem_r2_diff > 0 else ""
        
        print(f"      • CPU MAE: {cpu_mae_sign}{cpu_mae_diff:.2f}%p (RF: {rf_cpu_mae*100:.2f}% → XGB: {r['cpu_mae']*100:.2f}%)")
        print(f"      • CPU R²: {cpu_r2_sign}{cpu_r2_diff:.4f} (RF: {rf_cpu_r2:.4f} → XGB: {r['cpu_r2']:.4f})")
        print(f"      • Memory MAE: {mem_mae_sign}{mem_mae_diff:.2f}%p (RF: {rf_mem_mae*100:.2f}% → XGB: {r['memory_mae']*100:.2f}%)")
        print(f"      • Memory R²: {mem_r2_sign}{mem_r2_diff:.4f} (RF: {rf_mem_r2:.4f} → XGB: {r['memory_r2']:.4f})")
        
        print(f"\n{'='*100}")
    
    
    def _save_models(self):
        """
        학습된 모델 및 전처리기 저장
        """
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.cpu_model, self.model_output_dir / 'cpu_model.joblib')
        joblib.dump(self.memory_model, self.model_output_dir / 'memory_model.joblib')
        joblib.dump(self.label_encoders, self.model_output_dir / 'label_encoders.joblib')
        joblib.dump(self.scaler, self.model_output_dir / 'scaler.joblib')
        joblib.dump(self.feature_cols, self.model_output_dir / 'feature_cols.joblib')
        
        # 학습 결과 JSON 저장
        results_path = self.model_output_dir / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n   💾 모델 저장: {self.model_output_dir}")
        print(f"   💾 학습 결과: {results_path}")
    
    
    def load_models(self):
        """
        저장된 모델 로드
        
        Returns:
            self: 메서드 체이닝용
        """
        self.print_step("저장된 모델 로드", f"{self.model_output_dir}")
        
        self.cpu_model = joblib.load(self.model_output_dir / 'cpu_model.joblib')
        self.memory_model = joblib.load(self.model_output_dir / 'memory_model.joblib')
        self.label_encoders = joblib.load(self.model_output_dir / 'label_encoders.joblib')
        self.scaler = joblib.load(self.model_output_dir / 'scaler.joblib')
        self.feature_cols = joblib.load(self.model_output_dir / 'feature_cols.joblib')
        
        self.print_success("모델 로드 완료")
        
        return self
    
    
    def predict_aws(self):
        """
        AWS 데이터에 대해 사용률 예측
        
        --------------------------------------------------------
        프로세스:
        1. AWS 데이터에서 Feature 추출
        2. Feature 인코딩 (학습된 인코더 사용)
        3. CPU/Memory 사용률 예측
        4. 과다 프로비저닝 판정 (< 30%)
        --------------------------------------------------------
        
        Returns:
            self: 메서드 체이닝용
        """
        self.print_step("AWS 데이터 예측")
        
        if self.cpu_model is None:
            self.print_error("먼저 모델을 학습하거나 로드하세요.")
            return self
        
        if self.df_aws is None or len(self.df_aws) == 0:
            self.print_warning("AWS 데이터가 없습니다.")
            return self
        
        print(f"   📊 AWS 데이터: {len(self.df_aws):,}건")
        
        # Feature 추출
        features = self._extract_features(self.df_aws, is_training=False)
        
        # Feature 인코딩
        X = self._encode_features(features, fit=False)
        
        # 예측
        print(f"\n   🔮 예측 중...")
        cpu_predictions = self.cpu_model.predict(X)
        memory_predictions = self.memory_model.predict(X)
        
        # 예측값 클리핑 (0~1 범위)
        cpu_predictions = np.clip(cpu_predictions, 0, 1)
        memory_predictions = np.clip(memory_predictions, 0, 1)
        
        # 결과 저장
        self.df_predictions = self.df_aws.copy()
        self.df_predictions['PredictedCPU'] = cpu_predictions
        self.df_predictions['PredictedMemory'] = memory_predictions
        
        # 과다 프로비저닝 판정 (CPU 또는 Memory < 임계값)
        self.df_predictions['IsOverProvisioned'] = (
            (self.df_predictions['PredictedCPU'] < self.cpu_threshold) |
            (self.df_predictions['PredictedMemory'] < self.memory_threshold)
        )
        
        # 낭비율 계산
        self.df_predictions['CPUWastePercent'] = (
            (1 - self.df_predictions['PredictedCPU']) * 100
        )
        self.df_predictions['MemoryWastePercent'] = (
            (1 - self.df_predictions['PredictedMemory']) * 100
        )
        
        # 예상 절감액 (과다 프로비저닝 리소스의 60% 절감 가정)
        cost_col = 'TotalHourlyCost' if 'TotalHourlyCost' in self.df_predictions.columns else 'BilledCost'
        if cost_col in self.df_predictions.columns:
            self.df_predictions['PotentialSavings'] = np.where(
                self.df_predictions['IsOverProvisioned'],
                pd.to_numeric(self.df_predictions[cost_col], errors='coerce').fillna(0) * 0.6,
                0
            )
        
        # 결과 통계
        self._print_prediction_summary()
        
        return self
    
    
    def _print_prediction_summary(self):
        """
        예측 결과 요약 출력
        """
        print(f"\n{'='*100}")
        print("📊 AWS 사용률 예측 결과 (XGBoost)")
        print(f"{'='*100}")
        
        total = len(self.df_predictions)
        over_prov = self.df_predictions['IsOverProvisioned'].sum()
        over_prov_rate = over_prov / total * 100
        
        print(f"\n   🚨 과다 프로비저닝 탐지:")
        print(f"      • 전체: {total:,}건")
        print(f"      • 과다 프로비저닝: {over_prov:,}건 ({over_prov_rate:.1f}%)")
        print(f"      • 정상: {total - over_prov:,}건")
        
        # 예측값 분포
        print(f"\n   📊 예측 사용률 분포:")
        print(f"      • CPU 평균: {self.df_predictions['PredictedCPU'].mean()*100:.1f}%")
        print(f"      • CPU 중앙값: {self.df_predictions['PredictedCPU'].median()*100:.1f}%")
        print(f"      • Memory 평균: {self.df_predictions['PredictedMemory'].mean()*100:.1f}%")
        print(f"      • Memory 중앙값: {self.df_predictions['PredictedMemory'].median()*100:.1f}%")
        
        # 예상 절감액
        if 'PotentialSavings' in self.df_predictions.columns:
            total_savings = self.df_predictions['PotentialSavings'].sum()
            print(f"\n   💰 예상 절감액:")
            print(f"      • 총 절감 가능: ${total_savings:,.2f}")
            print(f"      • 연간 추정: ${total_savings * 12:,.2f}")
        
        print(f"\n{'='*100}")
    
    
    def save(self):
        """
        예측 결과 저장
        
        Returns:
            self: 메서드 체이닝용
        """
        if self.df_predictions is None:
            self.print_warning("저장할 예측 결과가 없습니다.")
            return self
        
        self.print_step("예측 결과 저장", f"{self.result_output_path}")
        
        # 디렉토리 생성
        self.ensure_dir(self.result_output_path.parent)
        
        # CSV 저장
        self.df_predictions.to_csv(self.result_output_path, index=False)
        
        self.print_success("저장 완료")
        print(f"   📂 경로: {self.result_output_path}")
        print(f"   📊 레코드: {len(self.df_predictions):,}건")
        
        # 과다 프로비저닝만 별도 저장
        over_prov_path = self.result_output_path.parent / 'xgb_overprovisioned.csv'
        df_over = self.df_predictions[self.df_predictions['IsOverProvisioned']]
        
        if len(df_over) > 0:
            df_over.to_csv(over_prov_path, index=False)
            print(f"   📂 과다 프로비저닝: {over_prov_path}")
            print(f"   📊 과다 프로비저닝: {len(df_over):,}건")
        
        return self
    
    
    def run(self):
        """
        전체 프로세스 실행: 로드 → 학습 → 예측 → 저장
        
        Returns:
            self: 메서드 체이닝용
        """
        return (self.load()
                .process()
                .predict_aws()
                .save())
    
    
    def get_results(self):
        """
        결과 반환
        
        Returns:
            tuple: (예측 결과 DataFrame, 학습 결과 dict)
        """
        return (self.df_predictions, self.training_results)
    
    
    def get_overprovisioned(self):
        """
        과다 프로비저닝 데이터만 반환
        
        Returns:
            DataFrame: 과다 프로비저닝 데이터
        """
        if self.df_predictions is None:
            return None
        
        return self.df_predictions[self.df_predictions['IsOverProvisioned']].copy()
    
    
    def compare_with_rf(self, rf_results_path=None):
        """
        RandomForest 결과와 비교
        
        Args:
            rf_results_path: RF 결과 JSON 경로 (없으면 기본값 사용)
        
        Returns:
            dict: 비교 결과
        """
        if self.training_results is None:
            print("⚠️ 먼저 학습을 실행하세요.")
            return None
        
        # RandomForest 기존 결과 (이전 대화에서 확인된 값)
        rf_results = {
            'cpu_mae': 0.2369,
            'cpu_r2': 0.0895,
            'memory_mae': 0.2371,
            'memory_r2': 0.0947
        }
        
        comparison = {
            'model_comparison': 'XGBoost vs RandomForest',
            'cpu': {
                'xgb_mae': self.training_results['cpu_mae'],
                'rf_mae': rf_results['cpu_mae'],
                'mae_improvement': rf_results['cpu_mae'] - self.training_results['cpu_mae'],
                'xgb_r2': self.training_results['cpu_r2'],
                'rf_r2': rf_results['cpu_r2'],
                'r2_improvement': self.training_results['cpu_r2'] - rf_results['cpu_r2']
            },
            'memory': {
                'xgb_mae': self.training_results['memory_mae'],
                'rf_mae': rf_results['memory_mae'],
                'mae_improvement': rf_results['memory_mae'] - self.training_results['memory_mae'],
                'xgb_r2': self.training_results['memory_r2'],
                'rf_r2': rf_results['memory_r2'],
                'r2_improvement': self.training_results['memory_r2'] - rf_results['memory_r2']
            }
        }
        
        return comparison


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    
    print("\n🚀 XGBoost 기반 사용률 예측 시작")
    print("="*100)
    print("📌 논문: LLM 기반 클라우드 FinOps 자동화 시스템 - Gemini vs Claude 성능 비교")
    print("📌 목적: RandomForest 대비 XGBoost 성능 비교")
    print("="*100)
    
    # 예측기 생성
    predictor = XGBUsagePredictor('config/focus_config.yaml')
    
    # 하이퍼파라미터 튜닝 설정
    predictor.tune_hyperparams = True
    predictor.sample_size = 5_000_000
    predictor.n_iter = 15
    predictor.cv_folds = 3
    
    # 실행
    predictor.run()
    
    # 결과 조회
    df_predictions, training_results = predictor.get_results()
    
    print(f"\n✅ 완료!")
    if df_predictions is not None:
        print(f"   전체 예측: {len(df_predictions):,}건")
        
        df_over = predictor.get_overprovisioned()
        if df_over is not None:
            print(f"   과다 프로비저닝: {len(df_over):,}건")
    
    # RF 대비 비교
    comparison = predictor.compare_with_rf()
    if comparison:
        print(f"\n📈 RandomForest 대비 비교:")
        print(f"   CPU MAE 개선: {comparison['cpu']['mae_improvement']*100:.2f}%p")
        print(f"   CPU R² 개선: {comparison['cpu']['r2_improvement']:.4f}")
        print(f"   Memory MAE 개선: {comparison['memory']['mae_improvement']*100:.2f}%p")
        print(f"   Memory R² 개선: {comparison['memory']['r2_improvement']:.4f}")