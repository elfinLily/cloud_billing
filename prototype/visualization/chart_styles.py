# -*- coding: utf-8 -*-
"""
차트 스타일 설정
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import platform

# Colorblind-friendly 색상 팔레트
# 참고: https://personal.sron.nl/~pault/
COLORS = {
    'primary': '#0173B2',      # 파랑 (신뢰)
    'secondary': '#DE8F05',    # 주황 (경고)
    'success': '#029E73',      # 초록 (성공)
    'danger': '#CC78BC',       # 보라 (위험)
    'warning': '#ECE133',      # 노랑 (주의)
    'info': '#56B4E9',         # 하늘색 (정보)
    'gray': '#949494',         # 회색 (중립)
}

# 차트용 색상 리스트 (순서대로 사용)
COLOR_PALETTE = [
    '#0173B2', '#DE8F05', '#029E73', 
    '#CC78BC', '#ECE133', '#56B4E9',
    '#949494', '#CA9161'
]

def set_preview_style():
    """
    프리뷰용 다크 스타일
    개발 중 화면에서 보기 편한 스타일
    """
    plt.style.use('dark_background')
    
    # 폰트 설정
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    
    # 그리드
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#444444'
    
    # 축 색상
    plt.rcParams['axes.edgecolor'] = '#666666'
    plt.rcParams['axes.labelcolor'] = '#CCCCCC'
    plt.rcParams['xtick.color'] = '#CCCCCC'
    plt.rcParams['ytick.color'] = '#CCCCCC'
    
    # 배경
    plt.rcParams['figure.facecolor'] = '#1e1e1e'
    plt.rcParams['axes.facecolor'] = '#1e1e1e'
    
    print("✅ 다크모드 스타일 적용됨 (개발용)")


def set_paper_style():
    """
    논문용 화이트 스타일
    출판 품질 (300 DPI) 차트용
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 폰트 설정
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    
    # 그리드
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#CCCCCC'
    plt.rcParams['grid.linewidth'] = 0.5
    
    # 축
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0
    
    # 배경
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    print("✅ 화이트 스타일 적용됨 (논문용)")


def format_number(num):
    """
    숫자를 읽기 쉽게 포맷
    
    예시:
        1234 → "1,234"
        1234.56 → "1,234.56"
    """
    if isinstance(num, (int, float)):
        if num >= 1000:
            return f"{num:,.0f}"
        else:
            return f"{num:.2f}"
    return str(num)


def format_currency(amount):
    """
    금액을 USD 형식으로 포맷
    
    예시:
        1234.56 → "$1,234.56"
    """
    return f"${amount:,.2f}"