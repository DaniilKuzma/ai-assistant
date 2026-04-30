"""
Пакет для интеллектуального ассистента проверки орфографии и пунктуации.
"""

try:
    from .hybrid_corrector import HybridCorrector
except ImportError:
    from hybrid_corrector import HybridCorrector

__version__ = "2.5.0"
__author__ = "Student"

__all__ = ["HybridCorrector"]

