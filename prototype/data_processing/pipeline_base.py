"""
Pipeline Base Class
"""

import yaml
from pathlib import Path
from abc import ABC, abstractmethod


class PipelineBase(ABC):
    """
    베이스 클래스
    
    주요 기능:
    - Config 자동 로드
    - 메서드 체이닝 지원 (모든 메서드가 self 반환)
    - 공통 유틸리티 메서드 제공
    
    사용 예시:
        class MyAnalyzer(PipelineBase):
            def load(self):
                # ... 로직
                return self  # 체이닝을 위해 self 반환
            
            def process(self):
                # ... 로직
                return self
            
            def run(self):
                return self.load().process().save()
    """
    
    def __init__(self, config_path='config/focus_config.yaml'):
        """
        초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data = None  # 처리 중인 데이터
        self.result = None  # 최종 결과
    
    
    def _load_config(self):
        """
        Config 파일 로드
        
        Returns:
            dict: 설정 딕셔너리
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    
    @abstractmethod
    def load(self):
        """
        데이터 로드 (추상 메서드)
        
        서브클래스에서 반드시 구현해야 함
        반환: self (체이닝 지원)
        """
        pass
    
    
    @abstractmethod
    def process(self):
        """
        데이터 처리 (추상 메서드)
        
        서브클래스에서 반드시 구현해야 함
        반환: self (체이닝 지원)
        """
        pass
    
    
    @abstractmethod
    def save(self):
        """
        결과 저장 (추상 메서드)
        
        서브클래스에서 반드시 구현해야 함
        반환: self (체이닝 지원)
        """
        pass
    
    
    def run(self):
        """
        전체 파이프라인 실행 (기본 구현)
        
        서브클래스에서 오버라이드 가능
        기본 순서: load → process → save
        
        Returns:
            self: 체이닝 지원
        """
        return self.load().process().save()
    
    
    def get_data(self):
        """
        현재 데이터 반환
        
        Returns:
            처리 중인 데이터
        """
        return self.data
    
    
    def get_result(self):
        """
        최종 결과 반환
        
        Returns:
            최종 결과 데이터
        """
        return self.result
    
    
    def get_config(self, key_path):
        """
        Config 값 조회 (중첩 키 지원)
        
        Args:
            key_path (str): 점(.)으로 구분된 키 경로
                예: 'data.output_dir' → config['data']['output_dir']
        
        Returns:
            Config 값
        
        예시:
            output_dir = self.get_config('data.output_dir')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    
    def ensure_dir(self, path):
        """
        디렉토리 존재 확인 및 생성
        
        Args:
            path (str or Path): 디렉토리 경로
        
        Returns:
            Path: 생성된 디렉토리 경로
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    
    def print_step(self, step_name, message=""):
        """
        단계별 진행 상황 출력 (일관된 포맷)
        
        Args:
            step_name (str): 단계 이름
            message (str): 추가 메시지
        """
        print(f"\n{'='*100}")
        print(f"🔄 {step_name}")
        if message:
            print(f"   {message}")
        print(f"{'='*100}")
    
    
    def print_success(self, message):
        """
        성공 메시지 출력
        
        Args:
            message (str): 성공 메시지
        """
        print(f"✅ {message}")
    
    
    def print_error(self, message):
        """
        에러 메시지 출력
        
        Args:
            message (str): 에러 메시지
        """
        print(f"❌ {message}")
    
    
    def print_warning(self, message):
        """
        경고 메시지 출력
        
        Args:
            message (str): 경고 메시지
        """
        print(f"⚠️  {message}")
    
    
    def __repr__(self):
        """
        객체 표현 문자열
        """
        return f"{self.__class__.__name__}(config='{self.config_path}')"


# ==================== 사용 예시 ====================
if __name__ == "__main__":
    
    # 추상 클래스라 직접 인스턴스화 불가
    # 서브클래스에서 상속받아 사용
    
    print("""
    PipelineBase 사용 예시:
    
    class MyAnalyzer(PipelineBase):
        def load(self):
            self.data = load_csv()
            return self
        
        def process(self):
            self.data = transform(self.data)
            return self
        
        def save(self):
            save_csv(self.data)
            return self
        
        def run(self):
            return self.load().process().save()
    
    # 사용
    analyzer = MyAnalyzer('config.yaml')
    analyzer.run()  # 내부적으로 체이닝 실행
    """)