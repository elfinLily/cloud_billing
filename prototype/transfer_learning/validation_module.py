# -*- coding: utf-8 -*-
"""
Transfer Learning 검증 모듈

GCP 데이터를 활용하여 추정 정확도를 검증합니다.
실제 사용률 vs 추정 사용률 비교
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))

from pipeline_base import PipelineBase


class TransferLearningValidator(PipelineBase):
    """
    Transfer Learning 검증 클래스
    
    검증 방법:
    1. GCP 데이터를 Train/Test로 분할
    2. Train 데이터로 패턴 학습
    3. Test 데이터로 추정
    4. 실제 vs 추정 비교
    
    평가 지표:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² Score
    - MAPE (Mean Absolute Percentage Error)
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
        self.output_path = Path('results/transfer_learning/validation_results.json')
        
        # 데이터
        self.df_gcp = None
        self.df_train = None
        self.df_test = None
        self.train_patterns = None
        self.validation_results = None
        
        # 검증 설정
        self.test_ratio = 0.2  # 20% 테스트
        self.random_state = 42
    
    
    def load(self):
        """
        GCP 데이터 로드
        
        Returns:
            self
        """
        self.print_step("GCP 데이터 로딩", f"{self.gcp_data_path}")
        
        if not self.gcp_data_path.exists():
            self.print_error(f"파일을 찾을 수 없습니다: {self.gcp_data_path}")
            raise FileNotFoundError(f"{self.gcp_data_path}")
        
        # CSV 로드
        self.df_gcp = pd.read_csv(self.gcp_data_path)
        
        self.print_success("로드 완료")
        print(f"   📊 레코드: {len(self.df_gcp):,}건")
        
        return self
    
    
    def _find_usage_columns(self):
        """
        CPU/Memory 사용률 컬럼 찾기
        
        Returns:
            tuple: (cpu_col, memory_col)
        """
        # CPU 컬럼 찾기
        cpu_cols = [col for col in self.df_gcp.columns 
                   if 'cpu' in col.lower() and ('usage' in col.lower() or 'utilization' in col.lower())]
        
        # Memory 컬럼 찾기
        memory_cols = [col for col in self.df_gcp.columns 
                      if 'memory' in col.lower() and ('usage' in col.lower() or 'utilization' in col.lower())]
        
        cpu_col = cpu_cols[0] if cpu_cols else None
        memory_col = memory_cols[0] if memory_cols else None
        
        print(f"\n   🔍 발견된 컬럼:")
        print(f"      • CPU: {cpu_col}")
        print(f"      • Memory: {memory_col}")
        
        return cpu_col, memory_col
    
    
    def _find_service_column(self):
        """
        서비스명 컬럼 찾기
        
        Returns:
            str: 서비스명 컬럼
        """
        service_cols = [col for col in self.df_gcp.columns 
                       if 'service' in col.lower() and 'name' in col.lower()]
        
        service_col = service_cols[0] if service_cols else None
        print(f"      • Service: {service_col}")
        
        return service_col
    
    
    def split_data(self):
        """
        데이터를 Train/Test로 분할
        
        분할 기준: 서비스별 Stratified Split
        
        Returns:
            self
        """
        self.print_step("Train/Test 분할")
        
        # 컬럼 찾기
        self.cpu_col, self.memory_col = self._find_usage_columns()
        self.service_col = self._find_service_column()
        
        if not self.cpu_col or not self.memory_col or not self.service_col:
            self.print_error("필수 컬럼을 찾을 수 없습니다.")
            return self
        
        # 데이터 정제
        df_clean = self.df_gcp[
            self.df_gcp[self.cpu_col].notna() &
            self.df_gcp[self.memory_col].notna() &
            self.df_gcp[self.service_col].notna()
        ].copy()
        
        # 0-1 범위로 정규화
        if df_clean[self.cpu_col].max() > 1.5:
            df_clean[self.cpu_col] = df_clean[self.cpu_col] / 100.0
        if df_clean[self.memory_col].max() > 1.5:
            df_clean[self.memory_col] = df_clean[self.memory_col] / 100.0
        
        # 0 제거
        df_clean = df_clean[
            (df_clean[self.cpu_col] > 0) & 
            (df_clean[self.memory_col] > 0)
        ]
        
        print(f"\n   📊 정제된 데이터: {len(df_clean):,}건")
        
        # 서비스별 분할
        services = df_clean[self.service_col].unique()
        
        train_indices = []
        test_indices = []
        
        np.random.seed(self.random_state)
        
        for service in services:
            service_indices = df_clean[df_clean[self.service_col] == service].index.tolist()
            
            if len(service_indices) < 5:
                # 샘플이 적으면 모두 Train으로
                train_indices.extend(service_indices)
            else:
                # Shuffle
                np.random.shuffle(service_indices)
                
                # Split
                split_idx = int(len(service_indices) * (1 - self.test_ratio))
                train_indices.extend(service_indices[:split_idx])
                test_indices.extend(service_indices[split_idx:])
        
        self.df_train = df_clean.loc[train_indices].copy()
        self.df_test = df_clean.loc[test_indices].copy()
        
        self.print_success("분할 완료")
        print(f"   • Train: {len(self.df_train):,}건 ({len(self.df_train)/len(df_clean)*100:.1f}%)")
        print(f"   • Test: {len(self.df_test):,}건 ({len(self.df_test)/len(df_clean)*100:.1f}%)")
        print(f"   • Train 서비스: {self.df_train[self.service_col].nunique()}개")
        print(f"   • Test 서비스: {self.df_test[self.service_col].nunique()}개")
        
        return self
    
    
    def learn_train_patterns(self):
        """
        Train 데이터에서 패턴 학습
        
        Returns:
            self
        """
        self.print_step("Train 데이터 패턴 학습")
        
        if self.df_train is None:
            self.print_error("먼저 split_data()를 실행하세요.")
            return self
        
        # 서비스별 패턴 학습
        self.train_patterns = {}
        
        grouped = self.df_train.groupby(self.service_col)
        
        for service, group in grouped:
            pattern = {
                'service_name': service,
                'sample_count': len(group)
            }
            
            # CPU 통계
            cpu_data = group[self.cpu_col].dropna()
            if len(cpu_data) > 0:
                pattern['cpu'] = {
                    'mean': float(cpu_data.mean()),
                    'std': float(cpu_data.std()),
                    'median': float(cpu_data.median())
                }
            
            # Memory 통계
            mem_data = group[self.memory_col].dropna()
            if len(mem_data) > 0:
                pattern['memory'] = {
                    'mean': float(mem_data.mean()),
                    'std': float(mem_data.std()),
                    'median': float(mem_data.median())
                }
            
            self.train_patterns[service] = pattern
        
        self.print_success(f"패턴 학습 완료: {len(self.train_patterns)}개 서비스")
        
        return self
    
    
    def _get_global_average(self):
        """
        전체 평균 패턴 계산 (Fallback용)
        
        Returns:
            dict: 평균 패턴
        """
        cpu_means = [p['cpu']['mean'] for p in self.train_patterns.values() if 'cpu' in p]
        mem_means = [p['memory']['mean'] for p in self.train_patterns.values() if 'memory' in p]
        
        return {
            'cpu': {'mean': np.mean(cpu_means)},
            'memory': {'mean': np.mean(mem_means)}
        }
    
    
    def process(self):
        """
        Test 데이터로 검증 수행
        
        Returns:
            self
        """
        self.print_step("검증 수행")
        
        if self.df_test is None or self.train_patterns is None:
            self.print_error("먼저 split_data()와 learn_train_patterns()를 실행하세요.")
            return self
        
        # 추정값 계산
        estimated_cpu = []
        estimated_memory = []
        actual_cpu = []
        actual_memory = []
        match_methods = []
        
        global_avg = self._get_global_average()
        
        for idx, row in self.df_test.iterrows():
            service = row[self.service_col]
            
            # 실제값
            actual_cpu.append(row[self.cpu_col])
            actual_memory.append(row[self.memory_col])
            
            # 추정값
            if service in self.train_patterns:
                pattern = self.train_patterns[service]
                estimated_cpu.append(pattern['cpu']['mean'])
                estimated_memory.append(pattern['memory']['mean'])
                match_methods.append('exact_match')
            else:
                # Fallback: 전체 평균
                estimated_cpu.append(global_avg['cpu']['mean'])
                estimated_memory.append(global_avg['memory']['mean'])
                match_methods.append('global_average')
        
        # 결과 DataFrame
        df_results = pd.DataFrame({
            'actual_cpu': actual_cpu,
            'estimated_cpu': estimated_cpu,
            'actual_memory': actual_memory,
            'estimated_memory': estimated_memory,
            'match_method': match_methods
        })
        
        # 평가 지표 계산
        self.validation_results = self._calculate_metrics(df_results)
        self.validation_results['df_comparison'] = df_results
        
        # 결과 출력
        self._print_validation_results()
        
        self.result = self.validation_results
        
        return self
    
    
    def _calculate_metrics(self, df_results):
        """
        평가 지표 계산
        
        Args:
            df_results: 실제/추정 비교 DataFrame
        
        Returns:
            dict: 평가 지표
        """
        metrics = {}
        
        # 전체 메트릭
        for target in ['cpu', 'memory']:
            actual = df_results[f'actual_{target}'].values
            estimated = df_results[f'estimated_{target}'].values
            
            # MAE
            mae = mean_absolute_error(actual, estimated)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(actual, estimated))
            
            # R² Score
            r2 = r2_score(actual, estimated)
            
            # MAPE
            mape = np.mean(np.abs((actual - estimated) / actual)) * 100
            
            metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
        
        # 매칭 방법별 메트릭
        for method in ['exact_match', 'global_average']:
            df_method = df_results[df_results['match_method'] == method]
            
            if len(df_method) == 0:
                continue
            
            for target in ['cpu', 'memory']:
                actual = df_method[f'actual_{target}'].values
                estimated = df_method[f'estimated_{target}'].values
                
                mae = mean_absolute_error(actual, estimated)
                
                metrics[f'{target}_{method}'] = {
                    'mae': mae,
                    'count': len(df_method)
                }
        
        # 메타 정보
        metrics['total_samples'] = len(df_results)
        metrics['exact_match_count'] = (df_results['match_method'] == 'exact_match').sum()
        metrics['global_average_count'] = (df_results['match_method'] == 'global_average').sum()
        metrics['exact_match_ratio'] = metrics['exact_match_count'] / metrics['total_samples'] * 100
        
        return metrics
    
    def process_service_level(self):
        """
        서비스 수준에서 검증 수행 (개선된 방식)
        
        개별 레코드가 아닌 서비스별 평균을 비교
        
        Returns:
            self
        """
        self.print_step("서비스 수준 검증 수행 (개선된 방식)")
        
        if self.df_test is None or self.train_patterns is None:
            self.print_error("먼저 split_data()와 learn_train_patterns()를 실행하세요.")
            return self
        
        # Test 데이터의 서비스별 실제 평균 계산
        test_service_avg = self.df_test.groupby(self.service_col).agg({
            self.cpu_col: 'mean',
            self.memory_col: 'mean'
        }).reset_index()
        
        test_service_avg.columns = ['service', 'actual_cpu_avg', 'actual_memory_avg']
        
        # Train 패턴에서 추정값 가져오기
        estimated_cpu = []
        estimated_memory = []
        match_methods = []
        
        global_avg = self._get_global_average()
        
        for service in test_service_avg['service']:
            if service in self.train_patterns:
                pattern = self.train_patterns[service]
                estimated_cpu.append(pattern['cpu']['mean'])
                estimated_memory.append(pattern['memory']['mean'])
                match_methods.append('exact_match')
            else:
                estimated_cpu.append(global_avg['cpu']['mean'])
                estimated_memory.append(global_avg['memory']['mean'])
                match_methods.append('global_average')
        
        test_service_avg['estimated_cpu_avg'] = estimated_cpu
        test_service_avg['estimated_memory_avg'] = estimated_memory
        test_service_avg['match_method'] = match_methods
        
        # 서비스 수준 메트릭 계산
        service_metrics = self._calculate_service_level_metrics(test_service_avg)
        
        # 결과 저장
        if self.validation_results is None:
            self.validation_results = {}
        
        self.validation_results['service_level'] = service_metrics
        self.validation_results['df_service_comparison'] = test_service_avg
        
        # 결과 출력
        self._print_service_level_results(service_metrics, test_service_avg)
        
        return self
    
    
    def _calculate_service_level_metrics(self, df_service):
        """
        서비스 수준 메트릭 계산
        
        Args:
            df_service: 서비스별 비교 DataFrame
        
        Returns:
            dict: 서비스 수준 메트릭
        """
        metrics = {}
        
        for target in ['cpu', 'memory']:
            actual = df_service[f'actual_{target}_avg'].values
            estimated = df_service[f'estimated_{target}_avg'].values
            
            # MAE
            mae = mean_absolute_error(actual, estimated)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(actual, estimated))
            
            # R² Score
            r2 = r2_score(actual, estimated)
            
            # MAPE (0 제외)
            mask = actual > 0.01
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - estimated[mask]) / actual[mask])) * 100
            else:
                mape = 0
            
            metrics[target] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
        
        metrics['total_services'] = len(df_service)
        metrics['exact_match_count'] = (df_service['match_method'] == 'exact_match').sum()
        metrics['exact_match_ratio'] = metrics['exact_match_count'] / metrics['total_services'] * 100
        
        return metrics
    
    
    def _print_service_level_results(self, metrics, df_service):
        """서비스 수준 검증 결과 출력"""
        print(f"\n{'='*100}")
        print("📊 서비스 수준 Transfer Learning 검증 결과 (개선된 방식)")
        print(f"{'='*100}")
        
        print(f"\n   📌 서비스 정보:")
        print(f"      • 총 서비스: {metrics['total_services']}개")
        print(f"      • Exact Match: {metrics['exact_match_count']}개 ({metrics['exact_match_ratio']:.1f}%)")
        
        # CPU 메트릭
        print(f"\n   📌 CPU 사용률 추정 정확도 (서비스 평균):")
        cpu = metrics['cpu']
        print(f"      • MAE: {cpu['mae']*100:.2f}% (절대 오차)")
        print(f"      • RMSE: {cpu['rmse']*100:.2f}%")
        print(f"      • R² Score: {cpu['r2']:.4f}")
        print(f"      • MAPE: {cpu['mape']:.2f}%")
        
        # Memory 메트릭
        print(f"\n   📌 Memory 사용률 추정 정확도 (서비스 평균):")
        mem = metrics['memory']
        print(f"      • MAE: {mem['mae']*100:.2f}% (절대 오차)")
        print(f"      • RMSE: {mem['rmse']*100:.2f}%")
        print(f"      • R² Score: {mem['r2']:.4f}")
        print(f"      • MAPE: {mem['mape']:.2f}%")
        
        # 해석
        print(f"\n   📌 해석:")
        avg_mae = (cpu['mae'] + mem['mae']) / 2 * 100
        avg_r2 = (cpu['r2'] + mem['r2']) / 2
        
        if avg_r2 > 0.7:
            print(f"      🎉 매우 우수! R² > 0.7")
        elif avg_r2 > 0.5:
            print(f"      ✅ 양호! R² > 0.5")
        elif avg_r2 > 0.3:
            print(f"      ⚠️ 보통 수준 (R² > 0.3)")
        else:
            print(f"      ❌ 서비스 특성이 다름 (R² < 0.3)")
        
        if avg_mae < 5:
            print(f"      🎉 평균 오차 < 5% - 실무 적용 가능!")
        elif avg_mae < 10:
            print(f"      ✅ 평균 오차 5-10% - 참고용으로 활용")
        else:
            print(f"      ⚠️ 평균 오차 > 10% - 주의 필요")
        
        # 상위 5개 정확한 서비스
        df_service['cpu_error'] = abs(df_service['actual_cpu_avg'] - df_service['estimated_cpu_avg'])
        top_accurate = df_service.nsmallest(5, 'cpu_error')
        
        print(f"\n   📈 가장 정확한 서비스 Top 5:")
        for i, row in top_accurate.iterrows():
            service = row['service'][:30]
            error = row['cpu_error'] * 100
            print(f"      • {service:30s}: 오차 {error:.2f}%")
        
        print(f"\n{'='*100}")

    def _print_validation_results(self):
        """검증 결과 출력"""
        print(f"\n{'='*100}")
        print("📊 Transfer Learning 검증 결과")
        print(f"{'='*100}")
        
        metrics = self.validation_results
        
        # 샘플 정보
        print(f"\n   📌 샘플 정보:")
        print(f"      • 총 테스트 샘플: {metrics['total_samples']:,}건")
        print(f"      • Exact Match: {metrics['exact_match_count']:,}건 ({metrics['exact_match_ratio']:.1f}%)")
        print(f"      • Global Average: {metrics['global_average_count']:,}건")
        
        # CPU 메트릭
        print(f"\n   📌 CPU 사용률 추정 정확도:")
        cpu = metrics['cpu']
        print(f"      • MAE: {cpu['mae']*100:.2f}% (절대 오차)")
        print(f"      • RMSE: {cpu['rmse']*100:.2f}%")
        print(f"      • R² Score: {cpu['r2']:.4f}")
        print(f"      • MAPE: {cpu['mape']:.2f}%")
        
        # Memory 메트릭
        print(f"\n   📌 Memory 사용률 추정 정확도:")
        mem = metrics['memory']
        print(f"      • MAE: {mem['mae']*100:.2f}% (절대 오차)")
        print(f"      • RMSE: {mem['rmse']*100:.2f}%")
        print(f"      • R² Score: {mem['r2']:.4f}")
        print(f"      • MAPE: {mem['mape']:.2f}%")
        
        # 매칭 방법별 비교
        print(f"\n   📌 매칭 방법별 MAE 비교:")
        
        if 'cpu_exact_match' in metrics:
            print(f"      • Exact Match CPU: {metrics['cpu_exact_match']['mae']*100:.2f}%")
        if 'cpu_global_average' in metrics:
            print(f"      • Global Avg CPU: {metrics['cpu_global_average']['mae']*100:.2f}%")
        if 'memory_exact_match' in metrics:
            print(f"      • Exact Match Memory: {metrics['memory_exact_match']['mae']*100:.2f}%")
        if 'memory_global_average' in metrics:
            print(f"      • Global Avg Memory: {metrics['memory_global_average']['mae']*100:.2f}%")
        
        # 해석
        print(f"\n   📌 해석:")
        
        avg_mae = (cpu['mae'] + mem['mae']) / 2 * 100
        
        if avg_mae < 5:
            print(f"      🎉 매우 우수한 추정 정확도! (평균 오차 < 5%)")
        elif avg_mae < 10:
            print(f"      ✅ 양호한 추정 정확도 (평균 오차 5-10%)")
        elif avg_mae < 15:
            print(f"      ⚠️ 보통 수준의 추정 정확도 (평균 오차 10-15%)")
        else:
            print(f"      ❌ 개선 필요 (평균 오차 > 15%)")
        
        print(f"\n{'='*100}")
    
    
    def _convert_to_serializable(self, obj):
        """
        numpy/pandas 타입을 JSON 직렬화 가능한 타입으로 변환
        
        Args:
            obj: 변환할 객체
        
        Returns:
            JSON 직렬화 가능한 객체
        """
        # DataFrame은 스킵 (별도 CSV로 저장)
        if isinstance(obj, pd.DataFrame):
            return None
        
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # 기타 타입은 문자열로 변환
            try:
                return str(obj)
            except:
                return None
    
    
    def save(self):
        """
        검증 결과 저장
        
        Returns:
            self
        """
        if self.validation_results is None:
            self.print_warning("저장할 결과가 없습니다.")
            return self
        
        self.print_step("검증 결과 저장", f"{self.output_path}")
        
        # 디렉토리 생성
        self.ensure_dir(self.output_path.parent)
        
        # DataFrame 제외한 결과만 JSON 저장
        results_to_save = {k: v for k, v in self.validation_results.items() 
                         if k != 'df_comparison'}
        
        # numpy 타입 변환
        results_to_save = self._convert_to_serializable(results_to_save)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    
    def run(self):
        """
        전체 검증 프로세스 실행
        
        Returns:
            self
        """
        return (self.load()
                .split_data()
                .learn_train_patterns()
                .process()
                .process_service_level()
                .save())
    
    def get_results(self):
        """
        검증 결과 반환
        
        Returns:
            dict: 검증 결과
        """
        return self.validation_results
    
    
    def get_summary_for_paper(self):
        """
        논문용 요약 통계
        
        Returns:
            dict: 논문에 넣을 핵심 지표
        """
        if self.validation_results is None:
            return {}
        
        metrics = self.validation_results
        
        return {
            'test_samples': metrics['total_samples'],
            'exact_match_ratio': metrics['exact_match_ratio'],
            'cpu_mae_percent': metrics['cpu']['mae'] * 100,
            'cpu_rmse_percent': metrics['cpu']['rmse'] * 100,
            'cpu_r2': metrics['cpu']['r2'],
            'memory_mae_percent': metrics['memory']['mae'] * 100,
            'memory_rmse_percent': metrics['memory']['rmse'] * 100,
            'memory_r2': metrics['memory']['r2'],
            'avg_mae_percent': (metrics['cpu']['mae'] + metrics['memory']['mae']) / 2 * 100,
            'avg_r2': (metrics['cpu']['r2'] + metrics['memory']['r2']) / 2
        }


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 Transfer Learning 검증 시작")
    print("="*100)
    
    validator = TransferLearningValidator('config/focus_config.yaml')
    validator.run()
    
    # 논문용 요약
    summary = validator.get_summary_for_paper()
    
    print(f"\n📝 논문용 요약:")
    print(f"   • 테스트 샘플: {summary.get('test_samples', 'N/A'):,}건")
    print(f"   • Exact Match 비율: {summary.get('exact_match_ratio', 'N/A'):.1f}%")
    print(f"   • CPU MAE: {summary.get('cpu_mae_percent', 'N/A'):.2f}%")
    print(f"   • Memory MAE: {summary.get('memory_mae_percent', 'N/A'):.2f}%")
    print(f"   • 평균 R² Score: {summary.get('avg_r2', 'N/A'):.4f}")
    
    print("\n✅ 검증 완료!")