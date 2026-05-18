import torch

from src.dnachunker_minimal.data import SyntheticDNADataset, mask_dna_batch
from src.dnachunker_minimal.model import MinimalDNAChunker, mlm_loss


def test_forward_shapes_and_metrics():
    model = MinimalDNAChunker(d_model=16, transformer_heads=4, transformer_layers=1)
    input_ids = torch.randint(1, 5, (2, 32))

    logits, metrics = model(input_ids)

    assert logits.shape == (2, 32, 7)
    assert metrics["avg_chunk_length"].item() > 0
    assert 0 <= metrics["boundary_density"].item() <= 1


def test_one_training_batch():
    dataset = SyntheticDNADataset(num_sequences=4, seq_len=32, seed=7)
    batch = mask_dna_batch([dataset[0], dataset[1]])
    model = MinimalDNAChunker(d_model=16, transformer_heads=4, transformer_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logits, _ = model(batch["input_ids"])
    loss = mlm_loss(logits, batch["labels"])
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
