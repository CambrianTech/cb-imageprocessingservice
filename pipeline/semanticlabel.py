from enum import IntEnum

class SemanticLabel(IntEnum):

	@property
	def index(self):
		"""The index of the Enum member."""
		return self.value - 1