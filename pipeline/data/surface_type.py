from .semanticlabel import SemanticLabel

#Keep major (floor, wall, ceiling) even, "Like" versions odd. 
#Perhaps write class method
class SurfaceType(SemanticLabel):
    Floor=0
    OnFloor=1

    Wall=2
    OnWall=3

    Ceiling=4
    OnCeiling=5

    Other=6

    @property
    def is_major(self):
        return self.index % 2 == 0

    def is_pair(self, label):
        return self.complement == label

    @property
    def complement(self):
        if self.is_major:
            return None if self == SurfaceType.Other else SurfaceType(self.index + 1)
        else: 
            return SurfaceType(self.index - 1)