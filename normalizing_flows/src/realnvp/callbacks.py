import os

import torch
import numpy as np
import matplotlib.pyplot as plt


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


class ModelCheckpoint:
	"""A callback to save and load model checkpoints.

	Args:
		save_dir: Directory to save model checkpoints.
		filename: Base filename for saved checkpoints. You can optionally provide a 
			format string to include the epoch and score.
		save_best_only: If True, only save when the model achieves the best score.
		mode: One of 'min' or 'max' to determine if a lower or higher score is better.
	"""
	
	def __init__(self, save_dir: str, filename: str = 'model_{epoch:03d}_{score:.3f}.pt', save_best_only: bool = True, mode: str = 'min'):
		self.save_dir = save_dir
		self.filename = filename
		self.save_best_only = save_best_only
		self.mode = mode
		self.best_score = float('inf') if mode == 'min' else float('-inf')
		
	def save(self, model, score: float = None, epoch: int = None) -> None:
		"""Save a model checkpoint.
		
		Args:
			model: The model to save
			score: Optional score associated with this checkpoint
			epoch: Optional epoch number
		"""
		os.makedirs(self.save_dir, exist_ok=True)
		
		if self.save_best_only:
			# Check if the score is better than the best score, if not, return and don't save
			if ((self.mode == 'min' and score >= self.best_score) or 
				(self.mode == 'max' and score <= self.best_score)):
				return
			self.best_score = score
			
		# Create filename with optional epoch and/or score
		filename = self.filename.format(epoch=epoch, score=score)
		
		# Save the model
		save_path = os.path.join(self.save_dir, filename)
		torch.save(model.state_dict(), save_path)

	@staticmethod
	def load(model, checkpoint_path: str) -> None:
		"""Load a model checkpoint into the provided model.
		
		Args:
			model: The model to load the weights into
			checkpoint_path: Path to the checkpoint file
		"""
		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
			
		model.load_state_dict(torch.load(checkpoint_path))


class SampleGeneration:
	"""A callback to generate and save sample images from the model at the end of each epoch.
	Uses fixed latent vectors z to track model evolution over time.
	
	Args:
		save_dir: Directory to save generated samples
		n_samples: Number of samples to generate
	"""

	def __init__(self, save_dir: str, n_samples: int = 25):
		self.save_dir = save_dir
		self.n_samples = n_samples

		os.makedirs(save_dir, exist_ok=True)
		self.fixed_z = None

	def _initialize_fixed_z(self, model):
		"""Initialize fixed z values from the model's base distribution."""
		device = next(model.parameters()).device
		self.fixed_z = model.base_dist.sample((self.n_samples,)).to(device)

	def _save_image_grid(self, samples: torch.Tensor, epoch: int) -> None:
		"""Save samples as a grid image.
		
		Args:
			samples: Tensor of shape (n_samples, 3, height, width)
			epoch: Current epoch number
		"""
		# Convert to numpy and clip
		samples = samples.cpu().numpy()
		samples = np.clip(samples, 0, 1)

		# Create grid
		n_rows = int(np.sqrt(self.n_samples))
		n_cols = int(np.ceil(self.n_samples / n_rows))

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
		axes = axes.flatten()

		for i, (ax, img) in enumerate(zip(axes, samples)):
			if i < self.n_samples:
				ax.imshow(np.transpose(img, (1, 2, 0)))
			ax.axis('off')

		# Remove empty subplots
		for i in range(self.n_samples, len(axes)):
			axes[i].axis('off')

		plt.tight_layout()
		save_path = os.path.join(self.save_dir, f'samples_epoch_{epoch:03d}.png')
		plt.savefig(save_path)
		plt.close()

	def generate_and_plot_images(self, model, epoch: int) -> None:
		"""Generate and save samples at the end of an epoch using fixed z values.
		
		Args:
			model: The model to generate samples from
			epoch: Current epoch number
		"""
		# Initialize fixed z values if not already done
		if self.fixed_z is None:
			self._initialize_fixed_z(model)

		# Generate samples using fixed z values
		samples = model.inverse(self.fixed_z)

		# Save samples as grid image
		self._save_image_grid(samples, epoch)
