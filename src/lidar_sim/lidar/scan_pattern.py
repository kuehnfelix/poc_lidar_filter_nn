from abc import ABC, abstractmethod
from typing import Iterator, Tuple

class ScanPattern(ABC):
    """Abstract base class for LiDAR scan patterns."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Iterate over azimuth and elevation pairs.
        
        Yields:
            Tuple[float, float]: (azimuth, elevation) in radians
        """
        pass