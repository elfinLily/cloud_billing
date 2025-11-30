# -*- coding: utf-8 -*-
"""
ML 기반 사용률 예측 모델 (v2)

resource_grouped.csv를 사용하여:
1. GCP 데이터 (AvgCPUUsage, AvgMemoryUsage 있음) → 학습
2. AWS 데이터 (사용률 없음) → 예측

RandomForest 회귀 모델 사용
"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# ============================================================
# 프로젝트 루트 설정
# ============================================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))

from pipeline_base import PipelineBase


class MLUsagePredictorV2(PipelineBase):
    """
    ML 기반 사용률 예측 클래스 (v2)
    
    데이터 흐름:
    1. resource_grouped.csv 로드 (GCP + AWS 통합)
    2. ProviderName으로 GCP/AWS 분리
    3. GCP 데이터로 학습 (AvgCPUUsage, AvgMemoryUsage)
    4. AWS 데이터에 적용하여 사용률 예측
    5. 과다 프로비저닝 탐지
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로 (config에서 모든 경로 읽음)
        """
        super().__init__(config_path)
        
        # ============================================================
        # 경로 설정 (config 기반)
        # ============================================================
        data_config = self.config['data']
        self.input_path = Path(data_config['resource_grouped_output'])  # 핵심 변경!
        self.model_output_dir = Path('results/transfer_learning/models')
        self.result_output_path = Path('results/transfer_learning/ml_predictions_v2.csv')
        
        # ============================================================
        # 임계값 설정 (config 기반)
        # ============================================================
        thresholds = self.config['thresholds']['over_provisioning']
        self.cpu_threshold = thresholds['cpu_threshold']      # 0.30
        self.memory_threshold = thresholds['memory_threshold']  # 0.30
        
        # ============================================================
        # 모델 관련 변수
        # ============================================================
        self.cpu_model = None
        self.memory_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # ============================================================
        # 데이터 변수
        # ============================================================
        self.df_all = None       # 전체 데이터
        self.df_gcp = None       # GCP 데이터 (학습용)
        self.df_aws = None       # AWS 데이터 (예측 대상)
        self.df_predictions = None  # 예측 결과
        
        # ============================================================
        # Feature 설정
        # ============================================================
        self.categorical_cols = ['ServiceName']
        self.numerical_cols = ['TotalHourlyCost', 'HourOfDay', 'DayOfWeek']
        self.feature_cols = []
    
    
    def load(self):
        """
        resource_grouped.csv 로드 및 GCP/AWS 분리
        
        Returns:
            self (메서드 체이닝)
        """
        self.print_step("데이터 로딩", f"{self.input_path}")
        
        if not self.input_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.input_path}")
            raise FileNotFoundError(f"{self.input_path}")
        
        # CSV 로드
        self.df_all = pd.read_csv(self.input_path)
        
        self.print_success("로드 완료")
        print(f"   📊 전체 레코드: {len(self.df_all):,}건")
        print(f"   📋 컬럼: {list(self.df_all.columns)}")
        
        # ============================================================
        # ProviderName으로 GCP/AWS 분리
        # ============================================================
        print(f"\n   🔀 Provider별 분리 중...")
        
        self.df_gcp = self.df_all[self.df_all['ProviderName'] == 'GCP'].copy()
        self.df_aws = self.df_all[self.df_all['ProviderName'] == 'AWS'].copy()
        
        print(f"   ☁️  GCP: {len(self.df_gcp):,}건")
        print(f"   ☁️  AWS: {len(self.df_aws):,}건")
        
        # ============================================================
        # GCP 데이터에 CPU/Memory 있는지 확인
        # ============================================================
        if 'AvgCPUUsage' not in self.df_gcp.columns:
            self.print_error("GCP 데이터에 AvgCPUUsage 컬럼이 없습니다!")
            raise ValueError("AvgCPUUsage 컬럼 필요")
        
        if 'AvgMemoryUsage' not in self.df_gcp.columns:
            self.print_error("GCP 데이터에 AvgMemoryUsage 컬럼이 없습니다!")
            raise ValueError("AvgMemoryUsage 컬럼 필요")
        
        self.print_success("GCP 데이터에 CPU/Memory 사용률 확인됨")
        
        return self
    
    
    def _extract_features(self, df):
        """
        Feature 추출
        
        Args:
            df: 원본 DataFrame
        
        Returns:
            DataFrame: Feature DataFrame
        """
        features = pd.DataFrame()
        
        # 1. ServiceName (그대로 사용)
        features['ServiceName'] = df['ServiceName'].fillna('Unknown')
        
        # 2. TotalHourlyCost
        features['TotalHourlyCost'] = pd.to_numeric(
            df['TotalHourlyCost'], errors='coerce'
        ).fillna(0)
        
        # 3. HourOfDay, DayOfWeek (HourlyTimestamp에서 추출)
        if 'HourlyTimestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['HourlyTimestamp'], errors='coerce')
                features['HourOfDay'] = timestamps.dt.hour.fillna(12)
                features['DayOfWeek'] = timestamps.dt.dayofweek.fillna(3)
            except:
                features['HourOfDay'] = 12
                features['DayOfWeek'] = 3
        else:
            features['HourOfDay'] = 12
            features['DayOfWeek'] = 3
        
        return features
    
    
    def _encode_features(self, features, fit=True):
        """
        Feature 인코딩 (LabelEncoder + StandardScaler)
        
        Args:
            features: Feature DataFrame
            fit: 인코더/스케일러 학습 여부
        
        Returns:
            numpy array: 인코딩된 Feature
        """
        df_encoded = features.copy()
        
        # ============================================================
        # 1. 카테고리 컬럼 인코딩 (LabelEncoder)
        # ============================================================
        for col in self.categorical_cols:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(
                        df_encoded[col].astype(str)
                    )
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # 학습 시 없던 카테고리는 첫 번째 클래스로 대체
                        df_encoded[col] = df_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ 
                            else 0  # Unknown → 0
                        )
        
        # ============================================================
        # 2. Feature 컬럼 선택
        # ============================================================
        feature_cols = self.categorical_cols + self.numerical_cols
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        self.feature_cols = feature_cols
        
        X = df_encoded[feature_cols].values
        
        # ============================================================
        # 3. 수치형 정규화 (StandardScaler)
        # ============================================================
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X
    
    
    def process(self):
        """
        GCP 데이터로 ML 모델 학습
        
        Returns:
            self (메서드 체이닝)
        """
        self.print_step("ML 모델 학습 (GCP 데이터)")
        
        # ============================================================
        # 1. GCP 데이터 정제
        # ============================================================
        print(f"\n   1️⃣  GCP 데이터 정제 중...")
        
        # 숫자형 변환
        self.df_gcp['AvgCPUUsage'] = pd.to_numeric(
            self.df_gcp['AvgCPUUsage'], errors='coerce'
        )
        self.df_gcp['AvgMemoryUsage'] = pd.to_numeric(
            self.df_gcp['AvgMemoryUsage'], errors='coerce'
        )
        
        # 결측치 및 이상치 제거
        df_clean = self.df_gcp.dropna(subset=['AvgCPUUsage', 'AvgMemoryUsage'])
        df_clean = df_clean[
            (df_clean['AvgCPUUsage'] > 0) & 
            (df_clean['AvgCPUUsage'] <= 1) &
            (df_clean['AvgMemoryUsage'] > 0) & 
            (df_clean['AvgMemoryUsage'] <= 1)
        ]
        
        print(f"      • 원본: {len(self.df_gcp):,}건")
        print(f"      • 정제 후: {len(df_clean):,}건")
        
        if len(df_clean) < 100:
            self.print_error("학습 데이터가 너무 적습니다 (최소 100건 필요)")
            return self
        
        # ============================================================
        # 2. Feature 추출 및 인코딩
        # ============================================================
        print(f"\n   2️⃣  Feature 추출 중...")
        
        features = self._extract_features(df_clean)
        X = self._encode_features(features, fit=True)
        
        y_cpu = df_clean['AvgCPUUsage'].values
        y_memory = df_clean['AvgMemoryUsage'].values
        
        print(f"      • Feature 수: {len(self.feature_cols)}")
        print(f"      • Feature 목록: {self.feature_cols}")
        
        # ============================================================
        # 3. Train/Test 분할
        # ============================================================
        print(f"\n   3️⃣  Train/Test 분할 중...")
        
        X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(
            X, y_cpu, test_size=0.2, random_state=42
        )
        _, _, y_mem_train, y_mem_test = train_test_split(
            X, y_memory, test_size=0.2, random_state=42
        )
        
        print(f"      • Train: {len(X_train):,}건")
        print(f"      • Test: {len(X_test):,}건")
        
        # ============================================================
        # 4. CPU 모델 학습
        # ============================================================
        print(f"\n   4️⃣  CPU 모델 학습 중...")
        
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
        
        print(f"      ✅ CPU MAE: {cpu_mae*100:.2f}%")
        print(f"      ✅ CPU R²: {cpu_r2:.4f}")
        
        # ============================================================
        # 5. Memory 모델 학습
        # ============================================================
        print(f"\n   5️⃣  Memory 모델 학습 중...")
        
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
        
        print(f"      ✅ Memory MAE: {mem_mae*100:.2f}%")
        print(f"      ✅ Memory R²: {mem_r2:.4f}")
        
        # ============================================================
        # 6. Feature 중요도 출력
        # ============================================================
        print(f"\n   📊 Feature 중요도 (CPU):")
        importances = self.cpu_model.feature_importances_
        for i, col in enumerate(self.feature_cols):
            print(f"      • {col}: {importances[i]*100:.1f}%")
        
        # ============================================================
        # 7. 모델 저장
        # ============================================================
        self._save_models()
        
        # 학습 결과 저장
        self.training_results = {
            'cpu_mae': float(cpu_mae),
            'cpu_r2': float(cpu_r2),
            'memory_mae': float(mem_mae),
            'memory_r2': float(mem_r2),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': {col: float(imp) for col, imp in zip(self.feature_cols, importances)}
        }
        
        return self
    
    
    def _save_models(self):
        """학습된 모델 저장"""
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.cpu_model, self.model_output_dir / 'cpu_model_v2.joblib')
        joblib.dump(self.memory_model, self.model_output_dir / 'memory_model_v2.joblib')
        joblib.dump(self.label_encoders, self.model_output_dir / 'label_encoders_v2.joblib')
        joblib.dump(self.scaler, self.model_output_dir / 'scaler_v2.joblib')
        joblib.dump(self.feature_cols, self.model_output_dir / 'feature_cols_v2.joblib')
        
        print(f"\n   💾 모델 저장: {self.model_output_dir}")
    
    
    def predict_aws(self):
        """
        AWS 데이터에 대해 사용률 예측
        
        Returns:
            self (메서드 체이닝)
        """
        self.print_step("AWS 데이터 예측")
        
        if self.cpu_model is None:
            self.print_error("먼저 모델을 학습하세요 (process)")
            return self
        
        if len(self.df_aws) == 0:
            self.print_warning("AWS 데이터가 없습니다")
            return self
        
        print(f"   📊 AWS 데이터: {len(self.df_aws):,}건")
        
        # ============================================================
        # 1. Feature 추출
        # ============================================================
        print(f"\n   🔧 Feature 추출 중...")
        features = self._extract_features(self.df_aws)
        print(f"   ✅ Feature 추출 완료: {len(features):,}건, {len(self.feature_cols)}개 컬럼")
        
        # ============================================================
        # 2. Feature 인코딩 (학습된 인코더 사용)
        # ============================================================
        X = self._encode_features(features, fit=False)
        
        # ============================================================
        # 3. 예측
        # ============================================================
        print(f"\n   🔮 예측 중...")
        cpu_predictions = self.cpu_model.predict(X)
        memory_predictions = self.memory_model.predict(X)
        
        # ============================================================
        # 4. 결과 저장
        # ============================================================
        self.df_predictions = self.df_aws.copy()
        self.df_predictions['PredictedCPU'] = cpu_predictions
        self.df_predictions['PredictedMemory'] = memory_predictions
        
        # ============================================================
        # 5. 과다 프로비저닝 판정
        # ============================================================
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
        
        # 예상 절감액 (과다 프로비저닝인 경우 60% 절감 가정)
        if 'TotalHourlyCost' in self.df_predictions.columns:
            self.df_predictions['TotalHourlyCost'] = pd.to_numeric(
                self.df_predictions['TotalHourlyCost'], errors='coerce'
            ).fillna(0)
            
            self.df_predictions['PotentialSavings'] = np.where(
                self.df_predictions['IsOverProvisioned'],
                self.df_predictions['TotalHourlyCost'] * 0.6,
                0
            )
        
        # ============================================================
        # 6. 결과 요약 출력
        # ============================================================
        self._print_prediction_summary()
        
        return self
    
    
    def _print_prediction_summary(self):
        """예측 결과 요약 출력"""
        print(f"\n{'='*100}")
        print("📊 AWS 사용률 예측 결과 (ML 기반)")
        print(f"{'='*100}")
        
        total = len(self.df_predictions)
        over_prov = self.df_predictions['IsOverProvisioned'].sum()
        over_prov_rate = over_prov / total * 100 if total > 0 else 0
        
        print(f"\n   🚨 과다 프로비저닝 탐지:")
        print(f"      • 전체: {total:,}건")
        print(f"      • 과다 프로비저닝: {over_prov:,}건 ({over_prov_rate:.1f}%)")
        print(f"      • 정상: {total - over_prov:,}건")
        
        # 예측값 분포
        print(f"\n   📊 예측 사용률 분포:")
        print(f"      • CPU 평균: {self.df_predictions['PredictedCPU'].mean()*100:.1f}%")
        print(f"      • CPU 중앙값: {self.df_predictions['PredictedCPU'].median()*100:.1f}%")
        print(f"      • CPU 최소: {self.df_predictions['PredictedCPU'].min()*100:.1f}%")
        print(f"      • CPU 최대: {self.df_predictions['PredictedCPU'].max()*100:.1f}%")
        print(f"      • Memory 평균: {self.df_predictions['PredictedMemory'].mean()*100:.1f}%")
        print(f"      • Memory 중앙값: {self.df_predictions['PredictedMemory'].median()*100:.1f}%")
        
        # 임계값 이하 비율
        below_cpu = (self.df_predictions['PredictedCPU'] < self.cpu_threshold).sum()
        below_mem = (self.df_predictions['PredictedMemory'] < self.memory_threshold).sum()
        
        print(f"\n   📉 임계값({self.cpu_threshold*100:.0f}%) 이하:")
        print(f"      • CPU < {self.cpu_threshold*100:.0f}%: {below_cpu:,}건 ({below_cpu/total*100:.1f}%)")
        print(f"      • Memory < {self.memory_threshold*100:.0f}%: {below_mem:,}건 ({below_mem/total*100:.1f}%)")
        
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
            self (메서드 체이닝)
        """
        if self.df_predictions is None:
            self.print_warning("저장할 예측 결과가 없습니다")
            return self
        
        self.print_step("예측 결과 저장", f"{self.result_output_path}")
        
        # 디렉토리 생성
        self.result_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CSV 저장
        self.df_predictions.to_csv(self.result_output_path, index=False)
        
        self.print_success("저장 완료")
        print(f"   📂 경로: {self.result_output_path}")
        print(f"   📊 레코드: {len(self.df_predictions):,}건")
        
        # 과다 프로비저닝만 별도 저장
        over_prov_path = self.result_output_path.parent / 'ml_overprovisioned_v2.csv'
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
            self (메서드 체이닝)
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


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*100)
    print("🚀 ML 기반 사용률 예측 v2 (resource_grouped.csv 사용)")
    print("="*100)
    
    predictor = MLUsagePredictorV2('config/focus_config.yaml')
    predictor.run()
    
    # 결과 조회
    df_predictions, training_results = predictor.get_results()
    
    print(f"\n✅ 완료!")
    if df_predictions is not None:
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