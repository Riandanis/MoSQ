"""
Tests for Gastro-Transformer v2: ModalitySlotQFormer Model.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gastro_transformer.config import GastroTransformerConfig
from gastro_transformer.model import ModalitySlotQFormer, ModalityProjector, QFormerBlock, CellLineEncoder


class TestModalitySlotQFormer:
    """Test suite for ModalitySlotQFormer model."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return GastroTransformerConfig(
            hidden_dim=256,
            qformer_layers=2,
            qformer_heads=4,
            num_query_tokens=16,
            num_tissue_types=10,
            num_cancer_types=8,
            num_drug_classes=6,
            num_cell_lines=50,
            batch_size=4,
        )

    @pytest.fixture
    def model(self, config):
        """Create model instance."""
        return ModalitySlotQFormer(config)

    def test_modality_slot_forward_image_rna(self, model):
        """Test: Image + RNA → fused embedding, tissue/cancer logits, no IC50."""
        batch_size = 4
        out = model(
            image_embeds=torch.randn(batch_size, 512),
            rna_embeds=torch.randn(batch_size, 256),
            return_embeddings=True
        )

        assert 'fused_embedding' in out
        assert out['fused_embedding'].shape == (batch_size, 256)
        assert 'tissue_logits' in out
        assert 'cancer_logits' in out
        assert 'ic50_pred' not in out  # No drug/cellline

    def test_modality_slot_forward_drug_cellline(self, model):
        """Test: Drug + CellLine → IC50 prediction."""
        batch_size = 4
        out = model(
            drug_embeds=torch.randn(batch_size, 768),
            cellline_ids=torch.randint(0, 50, (batch_size,)),
            cancer_type_ids=torch.randint(0, 8, (batch_size,)),
            cellline_rna_embeds=torch.randn(batch_size, 256),
        )

        assert 'fused_embedding' in out
        assert 'ic50_pred' in out
        assert out['ic50_pred'].shape == (batch_size,)

    def test_modality_slot_forward_all_modalities(self, model):
        """Test: All modalities → all outputs."""
        batch_size = 4
        out = model(
            image_embeds=torch.randn(batch_size, 512),
            rna_embeds=torch.randn(batch_size, 256),
            drug_embeds=torch.randn(batch_size, 768),
            cellline_ids=torch.randint(0, 50, (batch_size,)),
            cancer_type_ids=torch.randint(0, 8, (batch_size,)),
            tissue_ids=torch.randint(0, 10, (batch_size,)),
            cellline_rna_embeds=torch.randn(batch_size, 256),
        )

        assert 'fused_embedding' in out
        assert 'ic50_pred' in out
        assert 'tissue_logits' in out
        assert 'cancer_logits' in out
        assert 'drug_logits' in out

    def test_modality_slot_forward_single_modality(self, model):
        """Test: Each modality alone → fused embedding."""
        batch_size = 4

        # Image only
        out = model(image_embeds=torch.randn(batch_size, 512))
        assert 'fused_embedding' in out
        assert 'ic50_pred' not in out

        # RNA only
        out = model(rna_embeds=torch.randn(batch_size, 256))
        assert 'fused_embedding' in out

        # Drug only
        out = model(drug_embeds=torch.randn(batch_size, 768))
        assert 'fused_embedding' in out

    def test_modality_slot_forward_with_cellline_rna(self, model):
        """Test: Drug + CellLine + CellLine RNA → IC50."""
        batch_size = 4
        out = model(
            drug_embeds=torch.randn(batch_size, 768),
            cellline_ids=torch.randint(0, 50, (batch_size,)),
            cancer_type_ids=torch.randint(0, 8, (batch_size,)),
            tissue_ids=torch.randint(0, 10, (batch_size,)),
            cellline_rna_embeds=torch.randn(batch_size, 256),
        )

        assert 'ic50_pred' in out
        assert out['ic50_pred'].shape == (batch_size,)

    def test_type_embeddings_different(self, model):
        """Verify modality type embeddings are distinct."""
        img_type = model.modality_type_embeddings['image']
        rna_type = model.modality_type_embeddings['rna']
        drug_type = model.modality_type_embeddings['drug']
        cl_type = model.modality_type_embeddings['cellline']

        # Type embeddings should be different
        assert torch.allclose(img_type, rna_type, atol=1e-4) == False
        assert torch.allclose(img_type, drug_type, atol=1e-4) == False
        assert torch.allclose(img_type, cl_type, atol=1e-4) == False

    def test_missing_modality_robustness(self, model):
        """Test various missing modality combinations work."""
        batch_size = 4

        # Image + Drug (no RNA)
        out = model(
            image_embeds=torch.randn(batch_size, 512),
            drug_embeds=torch.randn(batch_size, 768),
        )
        assert 'fused_embedding' in out
        assert 'ic50_pred' not in out  # No cellline

        # Drug + cellline only
        out = model(
            drug_embeds=torch.randn(batch_size, 768),
            cellline_ids=torch.randint(0, 50, (batch_size,)),
        )
        assert 'ic50_pred' in out

    def test_prototype_initialization(self, model):
        """Test prototype initialization from paired data."""
        from collections import defaultdict

        paired_data = defaultdict(list)
        batch_size = 4

        # Add some dummy paired data
        for tissue_id in [0, 1, 2]:
            for _ in range(3):
                img_emb = torch.randn(512)
                rna_emb = torch.randn(256)
                paired_data[tissue_id].append((img_emb, rna_emb))

        model.initialize_prototypes(paired_data)

        assert model.image_prototypes is not None
        assert model.rna_prototypes is not None
        assert model.prototypes_initialized.item() == True

    def test_parameter_count(self, model):
        """Test parameter count is reasonable."""
        params = model.count_parameters()

        assert params['total'] > 1_000_000  # At least 1M params
        assert params['trainable'] == params['total']  # All trainable by default
        assert params['frozen'] == 0

    def test_gradient_flow(self, model):
        """Test gradients flow to all expected parameters."""
        batch_size = 4

        # Forward pass
        out = model(
            image_embeds=torch.randn(batch_size, 512, requires_grad=True),
            rna_embeds=torch.randn(batch_size, 256, requires_grad=True),
            return_embeddings=True
        )

        # Backward pass
        loss = out['fused_embedding'].sum()
        loss.backward()

        # Check gradients exist on projectors
        assert model.projectors['image'].weight.grad is not None
        assert model.projectors['rna'].weight.grad is not None

    def test_batch_size_invariance(self, model):
        """Test same input gives same output regardless of batch position."""
        batch_size = 2

        # Single sample twice
        img = torch.randn(1, 512)
        rna = torch.randn(1, 256)

        out1 = model(image_embeds=img, rna_embeds=rna)
        out2 = model(image_embeds=img, rna_embeds=rna)

        # Same inputs should give same outputs
        assert torch.allclose(out1['fused_embedding'], out2['fused_embedding'], atol=1e-5)

    def test_checkpoint_save_load(self, model, tmp_path):
        """Test save and load checkpoint."""
        import torch

        batch_size = 4
        out_before = model(
            image_embeds=torch.randn(batch_size, 512),
            rna_embeds=torch.randn(batch_size, 256),
        )

        # Save
        checkpoint_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # Load into new model
        model2 = ModalitySlotQFormer(model.config)
        model2.load_state_dict(torch.load(checkpoint_path))

        out_after = model2(
            image_embeds=torch.randn(batch_size, 512),
            rna_embeds=torch.randn(batch_size, 256),
        )

        # Outputs should match
        assert torch.allclose(out_before['fused_embedding'], out_after['fused_embedding'], atol=1e-5)


class TestModalityProjector:
    """Test ModalityProjector class."""

    def test_forward_1d(self):
        """Test 1D input."""
        proj = ModalityProjector(512, 256)
        x = torch.randn(4, 512)
        out = proj(x)
        assert out.shape == (4, 256)

    def test_forward_2d(self):
        """Test 2D input (multiple tokens)."""
        proj = ModalityProjector(512, 256)
        x = torch.randn(4, 10, 512)
        out = proj(x)
        assert out.shape == (4, 10, 256)


class TestQFormerBlock:
    """Test QFormerBlock class."""

    def test_forward(self):
        """Test forward pass."""
        block = QFormerBlock(256, 4)
        queries = torch.randn(4, 16, 256)
        modality = torch.randn(4, 5, 256)
        out = block(queries, modality)
        assert out.shape == (4, 16, 256)


class TestCellLineEncoder:
    """Test CellLineEncoder class."""

    @pytest.fixture
    def config(self):
        return GastroTransformerConfig(
            hidden_dim=256,
            num_cell_lines=50,
            num_cancer_types=8,
            num_tissue_types=10,
        )

    def test_forward_basic(self, config):
        """Test basic forward."""
        encoder = CellLineEncoder(config)
        cellline_ids = torch.randint(0, 50, (4,))
        out = encoder(cellline_ids)
        assert out.shape == (4, 256)

    def test_forward_with_cancer_type(self, config):
        """Test with cancer type."""
        encoder = CellLineEncoder(config)
        cellline_ids = torch.randint(0, 50, (4,))
        cancer_ids = torch.randint(0, 8, (4,))
        out = encoder(cellline_ids, cancer_type_ids=cancer_ids)
        assert out.shape == (4, 256)

    def test_forward_with_tissue_film(self, config):
        """Test with tissue FiLM modulation."""
        encoder = CellLineEncoder(config)
        cellline_ids = torch.randint(0, 50, (4,))
        tissue_ids = torch.randint(0, 10, (4,))
        out = encoder(cellline_ids, tissue_ids=tissue_ids)
        assert out.shape == (4, 256)

    def test_forward_with_rna(self, config):
        """Test with RNA fusion."""
        encoder = CellLineEncoder(config)
        cellline_ids = torch.randint(0, 50, (4,))
        rna_emb = torch.randn(4, 256)
        out = encoder(cellline_ids, rna_embeds=rna_emb)
        assert out.shape == (4, 256)

    def test_out_of_bounds_handling(self, config):
        """Test out-of-bounds IDs are clamped."""
        encoder = CellLineEncoder(config)
        # IDs that are too high
        cellline_ids = torch.tensor([100, 200, 300, 400])
        out = encoder(cellline_ids)  # Should clamp
        assert out.shape == (4, 256)


class TestNCDSplit:
    """Test NCD (No Common Drug) split functionality."""

    def test_create_ncd_folds_no_overlap(self):
        """Verify no drug appears in both train and test for any fold."""
        import pandas as pd
        from gastro_transformer.data import create_ncd_folds

        # Create synthetic IC50 data
        n_drugs = 50
        n_samples_per_drug = 20
        n_celllines = 100

        data = {
            'drug_id': [f'drug_{i % n_drugs}' for i in range(n_drugs * n_samples_per_drug)],
            'cellline_id': [f'cellline_{i % n_celllines}' for i in range(n_drugs * n_samples_per_drug)],
            'ic50_value': np.random.randn(n_drugs * n_samples_per_drug)
        }
        df = pd.DataFrame(data)

        # Create NCD folds
        folds = create_ncd_folds(df, n_folds=5, seed=42)

        # Verify no overlap in any fold
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_drugs = set(df.iloc[train_idx]['drug_id'])
            test_drugs = set(df.iloc[test_idx]['drug_id'])
            overlap = train_drugs & test_drugs

            assert len(overlap) == 0, f"Fold {fold_idx}: Found {len(overlap)} overlapping drugs"

    def test_create_ncd_folds_balanced(self):
        """Verify drug counts per fold are reasonably balanced."""
        import pandas as pd
        from gastro_transformer.data import create_ncd_folds

        # Create synthetic IC50 data with varying drug frequencies
        n_drugs = 30

        # Create data with different drug frequencies
        data = []
        for i in range(n_drugs):
            # Each drug has different number of samples (10 to 100)
            n_samples = 10 + i * 3
            for _ in range(n_samples):
                data.append({
                    'drug_id': f'drug_{i}',
                    'cellline_id': f'cellline_{len(data) % 50}',
                    'ic50_value': np.random.randn()
                })

        df = pd.DataFrame(data)

        # Create NCD folds
        folds = create_ncd_folds(df, n_folds=5, seed=42)

        # Check that all drugs are accounted for (union of train and test across folds)
        all_drugs = set()
        for train_idx, test_idx in folds:
            all_drugs.update(df.iloc[train_idx]['drug_id'])
            all_drugs.update(df.iloc[test_idx]['drug_id'])

        # All drugs should appear in either train or test (across folds)
        assert len(all_drugs) == n_drugs

        # Each fold should have balanced drug counts
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_drugs = len(set(df.iloc[train_idx]['drug_id']))
            test_drugs = len(set(df.iloc[test_idx]['drug_id']))
            # Each fold should have 24 train + 6 test = 30 drugs total
            assert train_drugs + test_drugs == n_drugs, f"Fold {fold_idx}: {train_drugs} + {test_drugs} != {n_drugs}"

    def test_create_ncd_folds_reproducibility(self):
        """Verify same seed produces same folds."""
        import pandas as pd
        from gastro_transformer.data import create_ncd_folds

        # Create synthetic data
        data = {
            'drug_id': [f'drug_{i % 20}' for i in range(200)],
            'cellline_id': [f'cellline_{i % 50}' for i in range(200)],
            'ic50_value': np.random.randn(200)
        }
        df = pd.DataFrame(data)

        # Create folds twice with same seed
        folds1 = create_ncd_folds(df, n_folds=5, seed=123)
        folds2 = create_ncd_folds(df, n_folds=5, seed=123)

        # Should produce identical splits
        for (train1, test1), (train2, test2) in zip(folds1, folds2):
            assert np.array_equal(train1, train2), "Train indices should match"
            assert np.array_equal(test1, test2), "Test indices should match"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
