# -*- coding: utf-8 -*-
"""
ML 기반 사용률 예측 모델

GCP 데이터로 학습하여 AWS 리소스의 CPU/Memory 사용률을 예측합니다.
RandomForest 회귀 모델 사용
"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))

from pipeline_base import PipelineBase


class MLUsagePredictor(PipelineBase):
    """
    ML 기반 사용률 예측 클래스
    
    주요 기능:
    1. GCP 데이터에서 Feature 추출
    2. RandomForest 모델 학습
    3. AWS 데이터에 적용하여 사용률 예측
    4. 과다 프로비저닝 탐지
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        
        # 경로 설정
        data_config = self.config['data']
        self.gcp_data_path = Path(data_config['gcp_raw_path'])
        self.aws_data_path = Path(data_config['aws_focus_output'])
        self.model_output_dir = Path('results/transfer_learning/models')
        self.result_output_path = Path('results/transfer_learning/ml_predictions.csv')
        
        # 임계값
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']
        self.memory_threshold = thresholds['memory_threshold']
        
        # 모델
        self.cpu_model = None
        self.memory_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # 데이터
        self.df_gcp = None
        self.df_aws = None
        self.df_predictions = None
        
        # Feature 컬럼
        self.feature_cols = []
        self.categorical_cols = ['ServiceCategory', 'ResourceType']
        self.numerical_cols = ['HourlyCost', 'HourOfDay', 'DayOfWeek', 'CostPerQuantity']
    
    
    def load(self):
        """
        GCP 데이터 로드
        
        Returns:
            self
        """
        self.print_step("GCP 학습 데이터 로딩", f"{self.gcp_data_path}")
        
        if not self.gcp_data_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.gcp_data_path}")
            raise FileNotFoundError(f"{self.gcp_data_path}")
        
        self.df_gcp = pd.read_csv(self.gcp_data_path)
        
        self.print_success("로드 완료")
        print(f"   📊 레코드: {len(self.df_gcp):,}건")
        print(f"   📋 컬럼: {len(self.df_gcp.columns)}개")
        
        # 컬럼 확인
        print(f"\n   📝 사용 가능한 컬럼:")
        for col in self.df_gcp.columns[:15]:
            print(f"      • {col}")
        if len(self.df_gcp.columns) > 15:
            print(f"      ... 외 {len(self.df_gcp.columns) - 15}개")
        
        return self
    
    
    def _find_columns(self):
        """
        필요한 컬럼 매핑
        
        Returns:
            dict: 컬럼 매핑
        """
        col_mapping = {}
        
        # CPU 사용률
        cpu_cols = [col for col in self.df_gcp.columns 
                   if 'cpu' in col.lower() and ('usage' in col.lower() or 'utilization' in col.lower())]
        col_mapping['cpu'] = cpu_cols[0] if cpu_cols else None
        
        # Memory 사용률
        mem_cols = [col for col in self.df_gcp.columns 
                   if 'memory' in col.lower() and ('usage' in col.lower() or 'utilization' in col.lower())]
        col_mapping['memory'] = mem_cols[0] if mem_cols else None
        
        # 서비스명
        service_cols = [col for col in self.df_gcp.columns 
                       if 'service' in col.lower() and 'name' in col.lower()]
        col_mapping['service'] = service_cols[0] if service_cols else None
        
        # 비용
        cost_cols = [col for col in self.df_gcp.columns 
                   if 'cost' in col.lower() and 'round' in col.lower()]
        if not cost_cols:
            cost_cols = [col for col in self.df_gcp.columns if 'cost' in col.lower()]
        col_mapping['cost'] = cost_cols[0] if cost_cols else None
        
        # 날짜
        date_cols = [col for col in self.df_gcp.columns 
                   if 'date' in col.lower() or 'time' in col.lower() or 'start' in col.lower()]
        col_mapping['date'] = date_cols[0] if date_cols else None
        
        # 단위당 비용
        unit_cost_cols = [col for col in self.df_gcp.columns 
                        if 'cost' in col.lower() and 'per' in col.lower()]
        col_mapping['cost_per_unit'] = unit_cost_cols[0] if unit_cost_cols else None
        
        print(f"\n   🔍 컬럼 매핑:")
        for key, col in col_mapping.items():
            status = "✅" if col else "❌"
            print(f"      {status} {key}: {col}")
        
        return col_mapping
    
    
    def _extract_features(self, df, col_mapping, is_training=True):
        """
        Feature 추출
        
        Args:
            df: 원본 DataFrame
            col_mapping: 컬럼 매핑
            is_training: 학습용 데이터인지 여부
        
        Returns:
            DataFrame: Feature DataFrame
        """
        print(f"\n   🔧 Feature 추출 중...")
        
        features = pd.DataFrame()
        
        # 1. ServiceCategory (서비스명에서 추출)
        if col_mapping['service']:
            features['ServiceCategory'] = df[col_mapping['service']].apply(
                self._categorize_service
            )
        else:
            features['ServiceCategory'] = 'Unknown'
        
        # 2. ResourceType (서비스명에서 추출)
        if col_mapping['service']:
            features['ResourceType'] = df[col_mapping['service']].apply(
                self._extract_resource_type
            )
        else:
            features['ResourceType'] = 'Unknown'
        
        # 3. HourlyCost
        if col_mapping['cost']:
            features['HourlyCost'] = pd.to_numeric(df[col_mapping['cost']], errors='coerce').fillna(0)
        else:
            features['HourlyCost'] = 0
        
        # 4. HourOfDay, DayOfWeek (날짜에서 추출)
        if col_mapping['date']:
            try:
                dates = pd.to_datetime(df[col_mapping['date']], errors='coerce')
                features['HourOfDay'] = dates.dt.hour.fillna(12)
                features['DayOfWeek'] = dates.dt.dayofweek.fillna(3)
            except:
                features['HourOfDay'] = 12
                features['DayOfWeek'] = 3
        else:
            features['HourOfDay'] = 12
            features['DayOfWeek'] = 3
        
        # 5. CostPerQuantity
        if col_mapping['cost_per_unit']:
            features['CostPerQuantity'] = pd.to_numeric(
                df[col_mapping['cost_per_unit']], errors='coerce'
            ).fillna(0)
        else:
            features['CostPerQuantity'] = 0
        
        # 6. Target 변수 (학습용만)
        if is_training:
            if col_mapping['cpu']:
                cpu_vals = pd.to_numeric(df[col_mapping['cpu']], errors='coerce')
                # 0-1 범위로 정규화
                if cpu_vals.max() > 1.5:
                    cpu_vals = cpu_vals / 100.0
                features['CPUUsage'] = cpu_vals
            
            if col_mapping['memory']:
                mem_vals = pd.to_numeric(df[col_mapping['memory']], errors='coerce')
                if mem_vals.max() > 1.5:
                    mem_vals = mem_vals / 100.0
                features['MemoryUsage'] = mem_vals
        
        print(f"      ✅ Feature 추출 완료: {len(features)}건, {len(features.columns)}개 컬럼")
        
        return features
    
    
    def _categorize_service(self, service_name):
        """
        서비스명을 카테고리로 분류
        
        Args:
            service_name: 서비스명
        
        Returns:
            str: 카테고리
        """
        if pd.isna(service_name):
            return 'Other'
        
        service_lower = str(service_name).lower()
        
        if any(kw in service_lower for kw in ['compute', 'engine', 'ec2', 'vm', 'instance']):
            return 'Compute'
        elif any(kw in service_lower for kw in ['storage', 's3', 'disk', 'bucket']):
            return 'Storage'
        elif any(kw in service_lower for kw in ['sql', 'database', 'rds', 'dynamo', 'firestore']):
            return 'Database'
        elif any(kw in service_lower for kw in ['network', 'vpc', 'load', 'cdn', 'cloudfront']):
            return 'Networking'
        elif any(kw in service_lower for kw in ['lambda', 'function', 'run', 'container']):
            return 'Serverless'
        elif any(kw in service_lower for kw in ['ai', 'ml', 'sagemaker', 'vertex']):
            return 'AI_ML'
        elif any(kw in service_lower for kw in ['monitor', 'log', 'cloudwatch', 'trace']):
            return 'Monitoring'
        elif any(kw in service_lower for kw in ['bigquery', 'analytics', 'athena', 'kinesis']):
            return 'Analytics'
        else:
            return 'Other'
    
    
    def _extract_resource_type(self, service_name):
        """
        서비스명에서 리소스 타입 추출
        
        Args:
            service_name: 서비스명
        
        Returns:
            str: 리소스 타입
        """
        if pd.isna(service_name):
            return 'Other'
        
        service_lower = str(service_name).lower()
        
        if any(kw in service_lower for kw in ['vm', 'instance', 'engine']):
            return 'VM'
        elif any(kw in service_lower for kw in ['container', 'kubernetes', 'ecs', 'eks']):
            return 'Container'
        elif any(kw in service_lower for kw in ['function', 'lambda']):
            return 'Function'
        elif any(kw in service_lower for kw in ['storage', 'bucket', 's3']):
            return 'ObjectStorage'
        elif any(kw in service_lower for kw in ['disk', 'volume', 'ebs']):
            return 'BlockStorage'
        elif any(kw in service_lower for kw in ['sql', 'database']):
            return 'Database'
        else:
            return 'Other'
    
    
    def _encode_features(self, features, fit=True):
        """
        카테고리 Feature 인코딩
        
        Args:
            features: Feature DataFrame
            fit: 인코더 학습 여부
        
        Returns:
            numpy array: 인코딩된 Feature
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
                    # 학습 시 없던 카테고리 처리
                    le = self.label_encoders.get(col)
                    if le:
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ 
                            else le.transform([le.classes_[0]])[0]
                        )
        
        # Feature 컬럼 선택
        feature_cols = self.categorical_cols + self.numerical_cols
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        
        X = df_encoded[feature_cols].values
        
        # 수치형 정규화
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        self.feature_cols = feature_cols
        
        return X
    
    
    def process(self):
        """
        ML 모델 학습
        
        Returns:
            self
        """
        self.print_step("ML 모델 학습")
        
        # 컬럼 매핑
        col_mapping = self._find_columns()
        
        if not col_mapping['cpu'] or not col_mapping['memory']:
            self.print_error("CPU/Memory 사용률 컬럼을 찾을 수 없습니다.")
            return self
        
        # Feature 추출
        features = self._extract_features(self.df_gcp, col_mapping, is_training=True)
        
        # 결측치 제거
        features_clean = features.dropna(subset=['CPUUsage', 'MemoryUsage'])
        features_clean = features_clean[
            (features_clean['CPUUsage'] > 0) & 
            (features_clean['CPUUsage'] <= 1) &
            (features_clean['MemoryUsage'] > 0) & 
            (features_clean['MemoryUsage'] <= 1)
        ]
        
        print(f"\n   📊 학습 데이터: {len(features_clean):,}건")
        
        # Feature 인코딩
        X = self._encode_features(features_clean, fit=True)
        y_cpu = features_clean['CPUUsage'].values
        y_memory = features_clean['MemoryUsage'].values
        
        # Train/Test 분할
        X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(
            X, y_cpu, test_size=0.2, random_state=42
        )
        _, _, y_mem_train, y_mem_test = train_test_split(
            X, y_memory, test_size=0.2, random_state=42
        )
        
        print(f"   📊 Train: {len(X_train):,}건, Test: {len(X_test):,}건")
        
        # CPU 모델 학습
        print(f"\n   🤖 CPU 모델 학습 중...")
        self.cpu_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.cpu_model.fit(X_train, y_cpu_train)
        
        # CPU 모델 평가
        y_cpu_pred = self.cpu_model.predict(X_test)
        cpu_mae = mean_absolute_error(y_cpu_test, y_cpu_pred)
        cpu_r2 = r2_score(y_cpu_test, y_cpu_pred)
        
        print(f"      ✅ CPU 모델 MAE: {cpu_mae*100:.2f}%")
        print(f"      ✅ CPU 모델 R²: {cpu_r2:.4f}")
        
        # Memory 모델 학습
        print(f"\n   🤖 Memory 모델 학습 중...")
        self.memory_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.memory_model.fit(X_train, y_mem_train)
        
        # Memory 모델 평가
        y_mem_pred = self.memory_model.predict(X_test)
        mem_mae = mean_absolute_error(y_mem_test, y_mem_pred)
        mem_r2 = r2_score(y_mem_test, y_mem_pred)
        
        print(f"      ✅ Memory 모델 MAE: {mem_mae*100:.2f}%")
        print(f"      ✅ Memory 모델 R²: {mem_r2:.4f}")
        
        # Feature 중요도
        print(f"\n   📊 Feature 중요도 (CPU):")
        importances = self.cpu_model.feature_importances_
        for i, col in enumerate(self.feature_cols):
            print(f"      • {col}: {importances[i]*100:.1f}%")
        
        # 모델 저장
        self._save_models()
        
        # 학습 결과 저장
        self.training_results = {
            'cpu_mae': cpu_mae,
            'cpu_r2': cpu_r2,
            'memory_mae': mem_mae,
            'memory_r2': mem_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': dict(zip(self.feature_cols, importances.tolist()))
        }
        
        return self
    
    
    def _save_models(self):
        """
        학습된 모델 저장
        """
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.cpu_model, self.model_output_dir / 'cpu_model.joblib')
        joblib.dump(self.memory_model, self.model_output_dir / 'memory_model.joblib')
        joblib.dump(self.label_encoders, self.model_output_dir / 'label_encoders.joblib')
        joblib.dump(self.scaler, self.model_output_dir / 'scaler.joblib')
        joblib.dump(self.feature_cols, self.model_output_dir / 'feature_cols.joblib')
        
        print(f"\n   💾 모델 저장: {self.model_output_dir}")
    
    
    def load_models(self):
        """
        저장된 모델 로드
        
        Returns:
            self
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
        
        Returns:
            self
        """
        self.print_step("AWS 데이터 예측")
        
        if self.cpu_model is None:
            self.print_error("먼저 모델을 학습하거나 로드하세요.")
            return self
        
        # AWS 데이터 로드
        if not self.aws_data_path.exists():
            self.print_error(f"AWS 데이터 파일을 찾을 수 없습니다: {self.aws_data_path}")
            return self
        
        self.df_aws = pd.read_csv(self.aws_data_path)
        print(f"   📊 AWS 데이터: {len(self.df_aws):,}건")
        
        # AWS 컬럼 매핑
        aws_col_mapping = {
            'service': 'ServiceName',
            'cost': 'BilledCost',
            'date': 'ChargePeriodStart',
            'cost_per_unit': None,
            'cpu': None,
            'memory': None
        }
        
        # 컬럼 존재 확인
        for key, col in aws_col_mapping.items():
            if col and col not in self.df_aws.columns:
                aws_col_mapping[key] = None
        
        # Feature 추출
        features = self._extract_features(self.df_aws, aws_col_mapping, is_training=False)
        
        # Feature 인코딩
        X = self._encode_features(features, fit=False)
        
        # 예측
        print(f"\n   🔮 예측 중...")
        cpu_predictions = self.cpu_model.predict(X)
        memory_predictions = self.memory_model.predict(X)
        
        # 결과 저장
        self.df_predictions = self.df_aws.copy()
        self.df_predictions['PredictedCPU'] = cpu_predictions
        self.df_predictions['PredictedMemory'] = memory_predictions
        self.df_predictions['ServiceCategory'] = features['ServiceCategory'].values
        self.df_predictions['ResourceType'] = features['ResourceType'].values
        
        # 과다 프로비저닝 판정
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
        
        # 예상 절감액
        if 'BilledCost' in self.df_predictions.columns:
            self.df_predictions['PotentialSavings'] = np.where(
                self.df_predictions['IsOverProvisioned'],
                self.df_predictions['BilledCost'] * 0.6,
                0
            )
        
        # 결과 통계
        self._print_prediction_summary()
        
        return self
    
    
    def _print_prediction_summary(self):
        """예측 결과 요약"""
        print(f"\n{'='*100}")
        print("📊 AWS 사용률 예측 결과 (ML 기반)")
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
        
        # 임계값 이하 비율
        below_cpu = (self.df_predictions['PredictedCPU'] < self.cpu_threshold).sum()
        below_mem = (self.df_predictions['PredictedMemory'] < self.memory_threshold).sum()
        
        print(f"\n   📉 임계값({self.cpu_threshold*100:.0f}%) 이하:")
        print(f"      • CPU < {self.cpu_threshold*100:.0f}%: {below_cpu:,}건 ({below_cpu/total*100:.1f}%)")
        print(f"      • Memory < {self.memory_threshold*100:.0f}%: {below_mem:,}건 ({below_mem/total*100:.1f}%)")
        
        # 카테고리별
        if over_prov > 0:
            print(f"\n   📊 카테고리별 과다 프로비저닝:")
            category_stats = self.df_predictions[self.df_predictions['IsOverProvisioned']].groupby(
                'ServiceCategory'
            ).size().sort_values(ascending=False)
            
            for cat, count in category_stats.head(5).items():
                pct = count / over_prov * 100
                print(f"      • {cat}: {count:,}건 ({pct:.1f}%)")
        
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
            self
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
        over_prov_path = self.result_output_path.parent / 'ml_overprovisioned.csv'
        df_over = self.df_predictions[self.df_predictions['IsOverProvisioned']]
        
        if len(df_over) > 0:
            df_over.to_csv(over_prov_path, index=False)
            print(f"   📂 과다 프로비저닝: {over_prov_path}")
            print(f"   📊 과다 프로비저닝: {len(df_over):,}건")
        
        return self
    
    
    def run(self):
        """
        전체 프로세스 실행: 학습 → 예측 → 저장
        
        Returns:
            self
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
        return (self.df_predictions, getattr(self, 'training_results', None))
    
    
    def get_overprovisioned(self):
        """
        과다 프로비저닝 데이터만 반환
        
        Returns:
            DataFrame: 과다 프로비저닝 데이터
        """
        if self.df_predictions is None:
            return None
        
        return self.df_predictions[self.df_predictions['IsOverProvisioned']].copy()


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 ML 기반 사용률 예측 시작")
    print("="*100)
    
    predictor = MLUsagePredictor('config/focus_config.yaml')
    predictor.run()
    
    # 결과 조회
    df_predictions, training_results = predictor.get_results()
    
    print(f"\n✅ 완료!")
    print(f"   전체 예측: {len(df_predictions):,}건")
    
    df_over = predictor.get_overprovisioned()
    if df_over is not None:
        print(f"   과다 프로비저닝: {len(df_over):,}건")
    
    if training_results:
        print(f"\n📊 학습 결과:")
        print(f"   CPU MAE: {training_results['cpu_mae']*100:.2f}%")
        print(f"   CPU R²: {training_results['cpu_r2']:.4f}")
        print(f"   Memory MAE: {training_results['memory_mae']*100:.2f}%")
        print(f"   Memory R²: {training_results['memory_r2']:.4f}")