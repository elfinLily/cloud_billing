# -*- coding: utf-8 -*-
"""
사용률 추정 모델

GCP에서 학습한 패턴을 기반으로 AWS 리소스의 CPU/Memory 사용률을 추정합니다.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'data_processing'))
from pipeline_base import PipelineBase

class UsageEstimator(PipelineBase):
    """
    사용률 추정 클래스
    
    주요 기능:
    1. GCP 학습 패턴 로드
    2. AWS-GCP 서비스 매칭
    3. CPU/Memory 사용률 추정
    4. 불확실성 점수 계산
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        super().__init__(config_path)
        
        # 경로 설정
        self.pattern_path = Path('results/transfer_learning/gcp_learned_patterns.json')
        self.mapping_path = Path('config/service_mapping.yaml')
        
        # 결과 저장
        self.gcp_patterns = None
        self.service_mapping = None
    
    
    def load(self):
        """
        GCP 학습 패턴 로드
        
        Returns:
            self
        """
        self.print_step("GCP 패턴 로드", f"{self.pattern_path}")
        
        if not self.pattern_path.exists():
            self.print_error(f"패턴 파일을 찾을 수 없습니다: {self.pattern_path}")
            raise FileNotFoundError(f"{self.pattern_path}")
        
        # JSON 로드
        with open(self.pattern_path, 'r', encoding='utf-8') as f:
            self.gcp_patterns = json.load(f)
        
        self.print_success("패턴 로드 완료")
        print(f"   📊 서비스: {len(self.gcp_patterns)}개")
        
        return self
    
    
    def _load_service_mapping(self):
        """
        AWS-GCP 서비스 매칭 테이블 로드
        
        매칭 파일이 없으면 자동 생성
        
        Returns:
            self
        """
        print("\n🔗 서비스 매칭 테이블 로드 중...")
        
        if not self.mapping_path.exists():
            self.print_warning("매칭 파일이 없습니다. 자동 생성합니다.")
            self._create_default_mapping()
        
        # YAML 로드
        import yaml
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            self.service_mapping = yaml.safe_load(f)
        
        mapping_count = len(self.service_mapping.get('mappings', {}))
        
        self.print_success("매칭 테이블 로드 완료")
        print(f"   📊 매칭: {mapping_count}개")
        
        return self
    
    
    def _create_default_mapping(self):
        """
        기본 AWS-GCP 서비스 매칭 테이블 생성
        """
        default_mapping = {
            'mappings': {
                # Compute
                'Amazon Elastic Compute Cloud': 'Compute Engine',
                'AWS Lambda': 'Cloud Functions',
                'Amazon Elastic Container Service': 'Cloud Run',
                'Amazon Elastic Kubernetes Service': 'Kubernetes Engine',
                
                # Storage
                'Amazon Simple Storage Service': 'Cloud Storage',
                'Amazon Elastic Block Store': 'Persistent Disk',
                'Amazon Elastic File System': 'Cloud Filestore',
                
                # Database
                'Amazon Relational Database Service': 'Cloud SQL',
                'Amazon DynamoDB': 'Cloud Firestore',
                'Amazon ElastiCache': 'Cloud Memorystore',
                
                # Networking
                'Amazon Virtual Private Cloud': 'Virtual Private Cloud',
                'Elastic Load Balancing': 'Cloud Load Balancing',
                'Amazon CloudFront': 'Cloud CDN',
                
                # Analytics
                'Amazon Athena': 'BigQuery',
                'Amazon EMR': 'Cloud Dataproc',
                'Amazon Kinesis': 'Cloud Pub/Sub',
                
                # AI/ML
                'Amazon SageMaker': 'Vertex AI',
                'Amazon Rekognition': 'Cloud Vision API',
                'Amazon Comprehend': 'Cloud Natural Language',
                
                # Monitoring
                'AmazonCloudWatch': 'Cloud Monitoring',
                'AWS CloudTrail': 'Cloud Logging',
            },
            'fallback_strategy': 'use_global_average'
        }
        
        # 디렉토리 생성
        self.ensure_dir(self.mapping_path.parent)
        
        # YAML 저장
        import yaml
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_mapping, f, 
                     default_flow_style=False, 
                     allow_unicode=True,
                     sort_keys=False)
        
        print(f"   ✅ 기본 매칭 테이블 생성: {self.mapping_path}")
        
        self.service_mapping = default_mapping
    
    
    def estimate_single(self, aws_service_name, aws_cost=None):
        """
        단일 AWS 서비스의 사용률 추정
        
        Args:
            aws_service_name (str): AWS 서비스명
            aws_cost (float): AWS 비용 (선택)
        
        Returns:
            dict: 추정 결과
                - cpu_mean: CPU 평균 사용률
                - cpu_std: CPU 표준편차
                - memory_mean: Memory 평균 사용률
                - memory_std: Memory 표준편차
                - confidence: 신뢰도 (0-1)
                - matched_gcp_service: 매칭된 GCP 서비스
                - method: 추정 방법
        """
        # 1. AWS → GCP 서비스 매칭
        gcp_service = self._match_service(aws_service_name)
        
        # 2. GCP 패턴 조회
        if gcp_service and gcp_service in self.gcp_patterns:
            pattern = self.gcp_patterns[gcp_service]
            method = 'exact_match'
            confidence = 1.0
        else:
            # Fallback: 전체 평균 사용
            pattern = self._get_global_average()
            method = 'global_average'
            confidence = 0.3
            gcp_service = 'Global Average'
        
        # 3. 추정값 생성
        result = {
            'aws_service': aws_service_name,
            'matched_gcp_service': gcp_service,
            'method': method,
            'confidence': confidence
        }
        
        # CPU 추정
        if 'cpu' in pattern:
            result['cpu_mean'] = pattern['cpu']['mean']
            result['cpu_std'] = pattern['cpu']['std']
            result['cpu_median'] = pattern['cpu']['median']
            result['cpu_min'] = pattern['cpu']['min']
            result['cpu_max'] = pattern['cpu']['max']
        
        # Memory 추정
        if 'memory' in pattern:
            result['memory_mean'] = pattern['memory']['mean']
            result['memory_std'] = pattern['memory']['std']
            result['memory_median'] = pattern['memory']['median']
            result['memory_min'] = pattern['memory']['min']
            result['memory_max'] = pattern['memory']['max']
        
        return result
    
    
    def _match_service(self, aws_service_name):
        """
        AWS 서비스를 GCP 서비스로 매칭
        
        Args:
            aws_service_name (str): AWS 서비스명
        
        Returns:
            str: 매칭된 GCP 서비스명 (없으면 None)
        """
        if not self.service_mapping:
            return None
        
        mappings = self.service_mapping.get('mappings', {})
        
        # 정확히 매칭
        if aws_service_name in mappings:
            return mappings[aws_service_name]
        
        # 부분 매칭 (대소문자 무시)
        aws_lower = aws_service_name.lower()
        for aws_key, gcp_value in mappings.items():
            if aws_key.lower() in aws_lower or aws_lower in aws_key.lower():
                return gcp_value
        
        return None
    
    
    def _get_global_average(self):
        """
        전체 GCP 서비스의 평균 패턴 계산
        
        Returns:
            dict: 평균 패턴
        """
        if not self.gcp_patterns:
            return {}
        
        # CPU 평균
        cpu_means = []
        cpu_stds = []
        
        for pattern in self.gcp_patterns.values():
            if 'cpu' in pattern:
                cpu_means.append(pattern['cpu']['mean'])
                cpu_stds.append(pattern['cpu']['std'])
        
        # Memory 평균
        mem_means = []
        mem_stds = []
        
        for pattern in self.gcp_patterns.values():
            if 'memory' in pattern:
                mem_means.append(pattern['memory']['mean'])
                mem_stds.append(pattern['memory']['std'])
        
        result = {}
        
        if cpu_means:
            result['cpu'] = {
                'mean': np.mean(cpu_means),
                'std': np.mean(cpu_stds),
                'median': np.median(cpu_means),
                'min': np.min(cpu_means),
                'max': np.max(cpu_means)
            }
        
        if mem_means:
            result['memory'] = {
                'mean': np.mean(mem_means),
                'std': np.mean(mem_stds),
                'median': np.median(mem_means),
                'min': np.min(mem_means),
                'max': np.max(mem_means)
            }
        
        return result
    
    
    def estimate_batch(self, aws_services):
        """
        여러 AWS 서비스의 사용률 일괄 추정
        
        Args:
            aws_services (list): AWS 서비스명 리스트
        
        Returns:
            pd.DataFrame: 추정 결과
        """
        results = []
        
        print(f"\n{'='*100}")
        print(f"🔄 일괄 추정 시작: {len(aws_services)}개 서비스")
        print(f"{'='*100}")
        
        for i, service in enumerate(aws_services, 1):
            if i % 10 == 0 or i == len(aws_services):
                print(f"   진행: {i}/{len(aws_services)}...", end='\r')
            
            result = self.estimate_single(service)
            results.append(result)
        
        print()  # 줄바꿈
        
        df_results = pd.DataFrame(results)
        
        self.print_success(f"일괄 추정 완료: {len(results)}개")
        
        # 통계 출력
        self._print_batch_summary(df_results)
        
        return df_results
    
    
    def _print_batch_summary(self, df_results):
        """일괄 추정 결과 요약"""
        print(f"\n{'='*100}")
        print("📊 추정 결과 요약")
        print(f"{'='*100}")
        
        # 추정 방법별 통계
        method_counts = df_results['method'].value_counts()
        print(f"\n   📌 추정 방법:")
        for method, count in method_counts.items():
            pct = count / len(df_results) * 100
            print(f"      • {method:20s}: {count:4,}건 ({pct:5.1f}%)")
        
        # 신뢰도 통계
        print(f"\n   📌 신뢰도:")
        print(f"      • 평균: {df_results['confidence'].mean():.2f}")
        print(f"      • 중앙값: {df_results['confidence'].median():.2f}")
        print(f"      • 최소: {df_results['confidence'].min():.2f}")
        print(f"      • 최대: {df_results['confidence'].max():.2f}")
        
        # CPU 사용률 통계
        if 'cpu_mean' in df_results.columns:
            print(f"\n   📌 추정 CPU 사용률:")
            print(f"      • 평균: {df_results['cpu_mean'].mean()*100:.2f}%")
            print(f"      • 중앙값: {df_results['cpu_mean'].median()*100:.2f}%")
            print(f"      • 최소: {df_results['cpu_mean'].min()*100:.2f}%")
            print(f"      • 최대: {df_results['cpu_mean'].max()*100:.2f}%")
        
        # Memory 사용률 통계
        if 'memory_mean' in df_results.columns:
            print(f"\n   📌 추정 Memory 사용률:")
            print(f"      • 평균: {df_results['memory_mean'].mean()*100:.2f}%")
            print(f"      • 중앙값: {df_results['memory_mean'].median()*100:.2f}%")
            print(f"      • 최소: {df_results['memory_mean'].min()*100:.2f}%")
            print(f"      • 최대: {df_results['memory_mean'].max()*100:.2f}%")
        
        print(f"\n{'='*100}")
    
    
    def process(self):
        """
        더미 process (PipelineBase 호환)
        
        Returns:
            self
        """
        return self
    
    
    def save(self):
        """
        더미 save (PipelineBase 호환)
        
        Returns:
            self
        """
        return self
    
    
    def run(self):
        """
        초기화 실행
        
        Returns:
            self
        """
        return self.load()._load_service_mapping()


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    
    print("\n🚀 사용률 추정 모델 테스트")
    print("="*100)
    
    estimator = UsageEstimator('config/focus_config.yaml')
    estimator.run()
    
    # 단일 추정 테스트
    print("\n" + "="*100)
    print("🧪 단일 서비스 추정 테스트")
    print("="*100)
    
    test_services = [
        'Amazon Elastic Compute Cloud',
        'Amazon Simple Storage Service',
        'Amazon Relational Database Service',
        'AmazonCloudWatch'
    ]
    
    for service in test_services:
        result = estimator.estimate_single(service)
        print(f"\n📊 {service}")
        print(f"   → GCP: {result['matched_gcp_service']}")
        print(f"   → 방법: {result['method']}")
        print(f"   → 신뢰도: {result['confidence']:.2f}")
        if 'cpu_mean' in result:
            print(f"   → CPU: {result['cpu_mean']*100:.2f}%")
        if 'memory_mean' in result:
            print(f"   → Memory: {result['memory_mean']*100:.2f}%")
    
    print("\n" + "="*100)
    print("✅ 테스트 완료!")