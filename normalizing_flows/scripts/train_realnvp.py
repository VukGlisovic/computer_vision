import os
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from normalizing_flows.src.utils import load_config
from normalizing_flows.src.realnvp.dataset import CelebADataset, ChunkedDataset
from normalizing_flows.src.realnvp.model.realnvp_flow import RealNVP
from normalizing_flows.src.realnvp.metrics import bits_per_dim
from normalizing_flows.src.realnvp.callbacks import EarlyStopping, ModelCheckpoint, SampleGeneration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_celeba_dataset():
	# Define image transformations
	transform = transforms.Compose([
		transforms.CenterCrop(size=128),
		transforms.Resize((32, 32)),
		transforms.ToTensor(),
	])

	# Load CelebA dataset
	ds = CelebADataset(
		root='../data',
		split='train',
		# if you have trouble downloading the images, download them manually and move the zip file to ../data/celeba/
		download=True,
		transform=transform
	)
	ds = ChunkedDataset(ds, n_chunks=6)

	# Create dataloader
	dl = DataLoader(
		ds,
		batch_size=32,
		num_workers=4,
		collate_fn=ds.dataset.collate_fn_skip_errors
	)

	# Print dataset information
	print(f"Train dataset size: {len(ds)}")
	print(f"Number of batches: {len(dl)}")
	return ds, dl


def create_model():
	model = RealNVP(
		in_channels=3,  # RGB images
		size=32,
		hidden_channels=32,
		n_hidden_layers=1,
		final_size=4
	)
	model = model.to(device)
	return model


def train(config):
	# Training configuration
	tr_config = config['training']
	n_epochs = tr_config['n_epochs']
	lr = tr_config['lr']

	# Init dataloader, model and optimizer
	ds_train, dl_train = create_celeba_dataset()
	model = create_model()
	model = model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	# Init callbacks
	cb_config = config['callbacks']
	reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cb_config['reduce_lr_on_plateau'])
	early_stopping = EarlyStopping(**cb_config['early_stopping'])
	model_checkpoint = ModelCheckpoint(**cb_config['model_checkpoint'])
	sample_generation = SampleGeneration(**cb_config['sample_generation'])

	for ep in range(n_epochs):
		# Reset nll loss and bps metric to zero
		nll_loss_sum = 0
		bpd_sum = 0

		for i, (x, _) in tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Epoch {ep:03d}"):
			# Get batch
			x = x.to(device)

			# Perform forward pass through model and run backpropagation
			optimizer.zero_grad()
			nll_loss = -model.log_prob(x)  # shape (bs,)
			nll_loss.mean().backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

			# Keep track of metrics
			nll_loss = nll_loss.detach().cpu()  # shape (bs,)
			nll_loss_sum += nll_loss.mean().item()
			bpd_sum += bits_per_dim(nll_loss, x.shape)

		# Calculate average metrics and log
		nll_loss_avg = nll_loss_sum / len(dl_train)
		bpd_avg = bpd_sum / len(dl_train)
		print(f"Epoch {ep + 1}/{n_epochs}, nll_loss: {nll_loss_avg:.4f}, bpd: {bpd_avg}, lr: {reduce_lr_on_plateau.get_last_lr()[0]}")

		# Execute callbacks
		model_checkpoint.save(model, score=bpd_avg, epoch=ep)
		sample_generation.generate_and_plot_images(model, epoch=ep)
		if early_stopping(bpd_avg):
			print(f'EarlyStopping activated. Ending training now.')
			break
		reduce_lr_on_plateau.step(bpd_avg)

		# Go to the next chunk of the training dataset; this will update the dataset in dl_train by reference
		ds_train.advance_chunk()

	best_path = os.path.join('checkpoints', os.listdir('checkpoints')[-1])
	print(f"Loading best model from checkpoint: {best_path}.")
	model_checkpoint.load(model, best_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_path', default='config_realnvp.yaml', type=str, help='Path to yaml file.')
	known_args, _ = parser.parse_known_args()

	train(
		load_config(known_args.config_path)
	)
