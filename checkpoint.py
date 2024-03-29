class Checkpoint:
    def __init__(self, higher_is_better = True):
        self.higher_is_better = higher_is_better
        self.best = None
        self.best_epoch = None
    
    def update(self, value, epoch):
        if self.best is None or (self.higher_is_better and value > self.best) or (not self.higher_is_better and value < self.best):
            self.best = value
            self.best_epoch = epoch
            return True
        return False