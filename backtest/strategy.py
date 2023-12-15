# strategy.py

from .dataloader import DataLoader

class Strategy:
    '''
    Base class for strategies. Your strategy should also subclass this class.
    '''
    
    def signal(self, data: DataLoader) -> None:
        '''
        Implement the strategy and produce the signals.
        '''
        raise NotImplementedError('Implement the strategy.')