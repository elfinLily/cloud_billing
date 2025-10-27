"""
이상 탐지 모듈

통계 기반 이상 탐지 알고리즘:
- Z-score 기반 탐지
- 퍼센타일 기반 탐지
- 연속 증가 패턴 탐지
"""

# 향후 구현될 탐지기들
# from .zscore_detector import ZScoreDetector
# from .percentile_detector import PercentileDetector
# from .trend_detector import TrendDetector

__all__ = [
    # 'ZScoreDetector',
    # 'PercentileDetector',
    # 'TrendDetector',
]

__version__ = '1.0.0'

# 모듈 설명
__doc__ = """
이상 탐지 모듈

Week 3-4에 구현 예정:
1. Z-score 기반 이상 탐지
2. 퍼센타일 기반 이상 탐지
3. 시계열 트렌드 분석
"""