

class EarlyStopping:
	"""A simple implementation of the EarlyStopping algorithm.

	Args:
		mode: options are 'min' and 'max'.
		patience: number of epochs to wait before early stopping.
		threshold: minimum delta between the latest score and the best score so far.
	"""

	def __init__(self, mode: str = 'min', patience: int = 10, threshold: float = 0):
		self.mode = mode
		self.patience = patience
		self.threshold = threshold

		self.best_score = None
		self.early_stop = False
		self.counter = 0

	def __call__(self, score: float):
		if self.best_score is None:
			self.best_score = score

		elif ((self.mode == 'min' and score >= self.best_score - self.threshold)
		      or (self.mode == 'max' and score <= self.best_score + self.threshold)):
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True

		else:
			self.best_score = score
			self.counter = 0

		return self.early_stop
