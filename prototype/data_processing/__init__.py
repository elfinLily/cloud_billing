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

__all__ = [
    'FocusDataLoader',
    'FocusAnalyzer',
    'OverProvisioningDetector',
    'UnusedResourceDetector',
]

# 버전 정보
__version__ = '1.0.0'

# 모듈 설명
__doc__ = """
FOCUS 데이터 처리 모듈

주요 기능:
1. FocusDataLoader: FOCUS CSV 데이터 로드 및 전처리
2. FocusAnalyzer: 메인 분석 오케스트레이터
3. OverProvisioningDetector: 과다 프로비저닝 패턴 탐지
4. UnusedResourceDetector: 미사용 리소스 패턴 탐지

사용 예시:
    >>> from data_processing import FocusAnalyzer
    >>> analyzer = FocusAnalyzer('config/focus_config.yaml')
    >>> results = analyzer.run()
"""