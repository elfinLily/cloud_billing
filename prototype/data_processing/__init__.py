"""
FOCUS 데이터 처리 모듈

FOCUS 표준 데이터를 로드하고 전처리하며,
2가지 패턴(과다 프로비저닝, 미사용 리소스)을 탐지합니다.
"""

from .focus_loader import FocusDataLoader
from .focus_analyzer import FocusAnalyzer
from .focus_patterns import (
    OverProvisioningDetector,
    UnusedResourceDetector
)
from .gcp_to_focus import GCPToFocusConverter
from .extract_aws_from_focus import AWSExtractor
from .analyze_all_focus import AllFocusAnalyzer
__all__ = [
    # 기본 모듈
    'FocusDataLoader',
    'FocusAnalyzer',
    
    # 패턴 탐지기
    'OverProvisioningDetector',
    'UnusedResourceDetector',
    
    # 데이터 변환/추출
    'GCPToFocusConverter',
    'AWSExtractor',
    
    'AllFocusAnalyzer',
]

# 버전 정보
__version__ = '2.0.0'

# 모듈 설명
__doc__ = """
FOCUS 데이터 처리 모듈

주요 기능:
1. FocusDataLoader: FOCUS CSV 데이터 로드 및 전처리
2. FocusAnalyzer: 단일 파일 분석 (구버전)
3. OverProvisioningDetector: 과다 프로비저닝 패턴 탐지
4. UnusedResourceDetector: 미사용 리소스 패턴 탐지
5. GCPToFocusConverter: GCP Kaggle → FOCUS 변환
6. AWSExtractor: FOCUS 샘플에서 AWS 추출
7. AllFocusAnalyzer: processed 폴더 전체 분석

사용 예시:
    >>> from data_processing import FocusAnalyzer
    >>> analyzer = FocusAnalyzer('config/focus_config.yaml')
    >>> results = analyzer.run()
"""