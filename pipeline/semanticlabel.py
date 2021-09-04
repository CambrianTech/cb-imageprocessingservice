from enum import IntEnum

class SemanticLabel(IntEnum):

    @property
    def index(self):
        """The index of the label."""
        return self.value - self.value_offset()

    @classmethod
    def values(cls):
        """Get list of all values."""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def max_value(cls):
        """The maximum value of the Enum."""
        return max(cls.values())

    @classmethod
    def max_index(cls):
        """The maximum value of the Enum."""
        return cls.max_value() - cls.value_offset()

    @classmethod
    def value_offset(cls):
        return 0