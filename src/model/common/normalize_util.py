import torch

from model.common.normalizer import SingleFieldLinearNormalizer,SequentialNormalizer, LinearNormalizer
import numpy as np
from legacy_constants import DepthNormalizationType, IMAGENET_DEPTH_MEAN, IMAGENET_DEPTH_STD

def get_range_normalizer_from_stat(stat, output_max = 1, output_min = -1, range_eps = 1e-7):
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def get_image_range_normalizer(device = None):
    """Get normalizer for image range [0,1] -> [-1,1]"""
    # Create tensors directly instead of numpy arrays

    scale = torch.tensor([2.], dtype=torch.float32)
    offset = torch.tensor([-1.], dtype=torch.float32)
    stat = {
        'min': torch.tensor([0.], dtype=torch.float32),
        'max': torch.tensor([1.], dtype=torch.float32),
        'mean': torch.tensor([0.5], dtype=torch.float32),
        'std': torch.tensor([np.sqrt(1 / 12)], dtype=torch.float32)
    }

    if device is not None:
        scale = scale.to(device)
        offset = offset.to(device)
        stat = {k: v.to(device) for k, v in stat.items()}

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat)


def get_depth_image_normalizer(input_min, input_max, input_mean, input_std, device = None, norm_type: str = DepthNormalizationType.ZERO_TO_ONE.value):
    if norm_type in [DepthNormalizationType.ZERO_TO_ONE.value, DepthNormalizationType.IMAGENET.value]:
        output_max, output_min = 1.0, 0.0
    elif norm_type == DepthNormalizationType.MINUS_ONE_TO_ONE.value:
        output_max, output_min = 1.0, -1.0
    else:
        raise ValueError(f"Unsupported depth normalization type: {norm_type}")

    # Stage 1: linear scaling to [output_min, output_max]
    scale = (output_max - output_min) / (input_max - input_min)
    offset = output_min - scale * input_min
    scale = torch.tensor([scale], dtype=torch.float32, device=device)
    offset = torch.tensor([offset], dtype=torch.float32, device=device)
    stat = {
        'min': torch.tensor([input_min], dtype=torch.float32, device=device),
        'max': torch.tensor([input_max], dtype=torch.float32, device=device),
        'mean': torch.tensor([input_mean], dtype=torch.float32, device=device),
        'std': torch.tensor([input_std], dtype=torch.float32, device=device),
    }
    if device is not None:
        scale = scale.to(device)
        offset = offset.to(device)
        stat = {k: v.to(device) for k, v in stat.items()}

    stage1 = SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)

    if norm_type == DepthNormalizationType.IMAGENET.value:
        # Stage 2: standardize with IMAGENET mean/std
        # Cf. https://github.com/VCIP-RGBD/RGBD-Pretrain/blob/main/data/constants.py#L3
        input_range = input_max - input_min

        # Calculate the input mean and std for the second normalizer
        scaled_mean = output_min + (input_mean - input_min) / input_range * (output_max - output_min)
        scaled_std = input_std / input_range * (output_max - output_min)

        scale2 = torch.tensor([1.0 / IMAGENET_DEPTH_STD], dtype=torch.float32, device=device)
        offset2 = torch.tensor([- IMAGENET_DEPTH_MEAN / IMAGENET_DEPTH_STD], dtype=torch.float32, device=device)
        stat2 = {
            'min': torch.tensor([output_min], dtype=torch.float32, device=device),
            'max': torch.tensor([output_max], dtype=torch.float32, device=device),
            'mean': torch.tensor([scaled_mean], dtype=torch.float32, device=device),
            'std': torch.tensor([scaled_std], dtype=torch.float32, device=device),
        }
        if device is not None:
            scale2 = scale2.to(device)
            offset2 = offset2.to(device)
            stat2 = {k: v.to(device) for k, v in stat2.items()}

        stage2 = SingleFieldLinearNormalizer.create_manual(scale=scale2, offset=offset2, input_stats_dict=stat2)

        return SequentialNormalizer(normalizers=[stage1, stage2])

    return stage1


def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_zero_to_one_normalizer(device=None):
    """Get identity normalizer for image range [0,1] -> [0,1]"""
    scale = torch.tensor([1.], dtype=torch.float32)
    offset = torch.tensor([0.], dtype=torch.float32)
    stat = {
        'min': torch.tensor([0.], dtype=torch.float32),
        'max': torch.tensor([1.], dtype=torch.float32),
        'mean': torch.tensor([0.5], dtype=torch.float32),
        'std': torch.tensor([np.sqrt(1 / 12)], dtype=torch.float32)
    }

    if device is not None:
        scale = scale.to(device)
        offset = offset.to(device)
        stat = {k: v.to(device) for k, v in stat.items()}

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )

def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat


if __name__ == "__main__":
    torch.manual_seed(0)

    torch.manual_seed(42)

    # Test data setup
    dmin, dmax = 0.0, 10.0
    dmean, dstd = 5.0, 2.0

    # Create dummy depth data with known statistics
    data = torch.linspace(dmin, dmax, steps=101).unsqueeze(-1)

    print("=" * 60)
    print("DEPTH NORMALIZER TESTS")
    print("=" * 60)

    # Test 1: ZERO_TO_ONE normalization
    print("\n--- Test 1: ZERO_TO_ONE Normalization ---")
    n_zero_to_one = get_depth_image_normalizer(
        dmin, dmax, dmean, dstd,
        norm_type='zero_to_one'
    )

    normalized = n_zero_to_one.normalize(data)
    unnormalized = n_zero_to_one.unnormalize(normalized)

    # Assertions for ZERO_TO_ONE
    assert isinstance(n_zero_to_one, SingleFieldLinearNormalizer), \
        f"Expected SingleFieldLinearNormalizer, got {type(n_zero_to_one)}"
    assert torch.allclose(normalized.min(), torch.tensor([0.0]), atol=1e-6), \
        f"Min should be 0, got {normalized.min()}"
    assert torch.allclose(normalized.max(), torch.tensor([1.0]), atol=1e-6), \
        f"Max should be 1, got {normalized.max()}"
    assert torch.allclose(data, unnormalized, atol=1e-6), \
        f"Round-trip failed for ZERO_TO_ONE"

    input_stats = n_zero_to_one.get_input_stats()
    output_stats = n_zero_to_one.get_output_stats()
    assert torch.allclose(input_stats['min'], torch.tensor([dmin])), \
        f"Input min mismatch: {input_stats['min']} != {dmin}"
    assert torch.allclose(input_stats['max'], torch.tensor([dmax])), \
        f"Input max mismatch: {input_stats['max']} != {dmax}"
    assert torch.allclose(output_stats['min'], torch.tensor([0.0])), \
        f"Output min should be 0, got {output_stats['min']}"
    assert torch.allclose(output_stats['max'], torch.tensor([1.0])), \
        f"Output max should be 1, got {output_stats['max']}"

    print(f"✓ ZERO_TO_ONE: min={normalized.min():.4f}, max={normalized.max():.4f}")
    print(f"✓ Round-trip successful")

    # Test 2: MINUS_ONE_TO_ONE normalization
    print("\n--- Test 2: MINUS_ONE_TO_ONE Normalization ---")
    n_minus_one = get_depth_image_normalizer(
        dmin, dmax, dmean, dstd,
        norm_type='minus_one_to_one'
    )

    normalized = n_minus_one.normalize(data)
    unnormalized = n_minus_one.unnormalize(normalized)

    assert isinstance(n_minus_one, SingleFieldLinearNormalizer), \
        f"Expected SingleFieldLinearNormalizer, got {type(n_minus_one)}"
    assert torch.allclose(normalized.min(), torch.tensor([-1.0]), atol=1e-6), \
        f"Min should be -1, got {normalized.min()}"
    assert torch.allclose(normalized.max(), torch.tensor([1.0]), atol=1e-6), \
        f"Max should be 1, got {normalized.max()}"
    assert torch.allclose(data, unnormalized, atol=1e-6), \
        f"Round-trip failed for MINUS_ONE_TO_ONE"

    output_stats = n_minus_one.get_output_stats()
    assert torch.allclose(output_stats['min'], torch.tensor([-1.0])), \
        f"Output min should be -1, got {output_stats['min']}"
    assert torch.allclose(output_stats['max'], torch.tensor([1.0])), \
        f"Output max should be 1, got {output_stats['max']}"

    print(f"✓ MINUS_ONE_TO_ONE: min={normalized.min():.4f}, max={normalized.max():.4f}")
    print(f"✓ Round-trip successful")

    # Test 3: IMAGENET normalization (Sequential)
    print("\n--- Test 3: IMAGENET Normalization (Sequential) ---")
    n_imagenet = get_depth_image_normalizer(
        dmin, dmax, dmean, dstd,
        norm_type='imagenet'
    )

    normalized = n_imagenet.normalize(data)
    unnormalized = n_imagenet.unnormalize(normalized)

    assert isinstance(n_imagenet, SequentialNormalizer), \
        f"Expected SequentialNormalizer, got {type(n_imagenet)}"
    assert hasattr(n_imagenet, 'normalizers') and n_imagenet.normalizers is not None, \
        "SequentialNormalizer should have normalizers"
    assert len(n_imagenet.normalizers) == 2, \
        f"Should have 2 normalizers, got {len(n_imagenet.normalizers)}"
    assert torch.allclose(data, unnormalized, atol=1e-6), \
        f"Round-trip failed for IMAGENET"

    # Check IMAGENET stats
    IMAGENET_MEAN = 0.48
    IMAGENET_STD = 0.28

    # Expected values after IMAGENET normalization
    expected_min = (0.0 - IMAGENET_MEAN) / IMAGENET_STD  # -1.714...
    expected_max = (1.0 - IMAGENET_MEAN) / IMAGENET_STD  # 1.857...
    scaled_mean = (dmean - dmin) / (dmax - dmin)  # 0.5 for our data
    expected_mean = (scaled_mean - IMAGENET_MEAN) / IMAGENET_STD
    scaled_std = dstd / (dmax - dmin)  # 0.2 for our data
    expected_std = scaled_std / IMAGENET_STD

    input_stats = n_imagenet.get_input_stats()
    output_stats = n_imagenet.get_output_stats()

    assert torch.allclose(input_stats['min'], torch.tensor([dmin])), \
        f"Input min mismatch: {input_stats['min']} != {dmin}"
    assert torch.allclose(input_stats['max'], torch.tensor([dmax])), \
        f"Input max mismatch: {input_stats['max']} != {dmax}"
    assert torch.allclose(output_stats['min'], torch.tensor([expected_min]), atol=1e-4), \
        f"Output min should be {expected_min:.4f}, got {output_stats['min'].item():.4f}"
    assert torch.allclose(output_stats['max'], torch.tensor([expected_max]), atol=1e-4), \
        f"Output max should be {expected_max:.4f}, got {output_stats['max'].item():.4f}"
    assert torch.allclose(output_stats['mean'], torch.tensor([expected_mean]), atol=1e-4), \
        f"Output mean should be {expected_mean:.4f}, got {output_stats['mean'].item():.4f}"
    assert torch.allclose(output_stats['std'], torch.tensor([expected_std]), atol=1e-4), \
        f"Output std should be {expected_std:.4f}, got {output_stats['std'].item():.4f}"

    print(f"✓ IMAGENET Sequential: {len(n_imagenet.normalizers)} stages")
    print(f"✓ Output stats: min={output_stats['min'].item():.4f}, max={output_stats['max'].item():.4f}")
    print(f"✓ Output stats: mean={output_stats['mean'].item():.4f}, std={output_stats['std'].item():.4f}")
    print(f"✓ Round-trip successful")

    # Test 4: LinearNormalizer integration
    print("\n--- Test 4: LinearNormalizer Integration ---")
    normalizer = LinearNormalizer()

    # Test storing ZERO_TO_ONE
    normalizer['depth_zero'] = get_depth_image_normalizer(
        dmin, dmax, dmean, dstd, norm_type='zero_to_one'
    )
    retrieved_zero = normalizer['depth_zero']
    assert isinstance(retrieved_zero, SingleFieldLinearNormalizer), \
        f"Retrieved ZERO_TO_ONE should be SingleFieldLinearNormalizer, got {type(retrieved_zero)}"

    # Test storing IMAGENET (Sequential)
    normalizer['depth_imagenet'] = get_depth_image_normalizer(
        dmin, dmax, dmean, dstd, norm_type='imagenet'
    )
    retrieved_imagenet = normalizer['depth_imagenet']
    assert isinstance(retrieved_imagenet, SequentialNormalizer), \
        f"Retrieved IMAGENET should be SequentialNormalizer, got {type(retrieved_imagenet)}"

    # Test that retrieved normalizers work correctly
    normalized_zero = retrieved_zero.normalize(data)
    assert torch.allclose(normalized_zero.min(), torch.tensor([0.0]), atol=1e-6), \
        "Retrieved ZERO_TO_ONE normalizer not working"

    normalized_imagenet = retrieved_imagenet.normalize(data)
    unnormalized_imagenet = retrieved_imagenet.unnormalize(normalized_imagenet)
    assert torch.allclose(data, unnormalized_imagenet, atol=1e-6), \
        "Retrieved IMAGENET normalizer round-trip failed"

    print(f"✓ LinearNormalizer stores and retrieves SingleFieldLinearNormalizer")
    print(f"✓ LinearNormalizer stores and retrieves SequentialNormalizer")
    print(f"✓ Retrieved normalizers function correctly")

    # Test 5: State dict save/load
    print("\n--- Test 5: State Dict Save/Load ---")
    state_dict = normalizer.state_dict()

    # Create new normalizer and load state
    new_normalizer = LinearNormalizer()
    new_normalizer.load_state_dict(state_dict)

    # Test that loaded normalizers work
    loaded_zero = new_normalizer['depth_zero']
    loaded_imagenet = new_normalizer['depth_imagenet']

    assert isinstance(loaded_zero, SingleFieldLinearNormalizer), \
        "Loaded ZERO_TO_ONE type incorrect"
    assert isinstance(loaded_imagenet, SequentialNormalizer), \
        "Loaded IMAGENET type incorrect"

    # Test functionality after loading
    test_data = torch.randn(10, 1) * 2 + 5  # Random data in roughly [3, 7] range
    test_data = torch.clamp(test_data, dmin, dmax)  # Ensure within bounds

    # Original normalizers
    orig_zero_out = normalizer['depth_zero'].normalize(test_data)
    orig_imagenet_out = normalizer['depth_imagenet'].normalize(test_data)

    # Loaded normalizers
    loaded_zero_out = loaded_zero.normalize(test_data)
    loaded_imagenet_out = loaded_imagenet.normalize(test_data)

    assert torch.allclose(orig_zero_out, loaded_zero_out, atol=1e-6), \
        "Loaded ZERO_TO_ONE produces different output"
    assert torch.allclose(orig_imagenet_out, loaded_imagenet_out, atol=1e-6), \
        "Loaded IMAGENET produces different output"

    print(f"✓ State dict save/load preserves normalizer types")
    print(f"✓ Loaded normalizers produce identical outputs")

    # Test 6: Device handling
    if torch.cuda.is_available():
        print("\n--- Test 6: Device Handling (CUDA) ---")
        device = 'cuda:0'

        n_cuda = get_depth_image_normalizer(
            dmin, dmax, dmean, dstd,
            device=device,
            norm_type='imagenet'
        )

        data_cuda = data.to(device)
        normalized_cuda = n_cuda.normalize(data_cuda)

        assert normalized_cuda.device.type == 'cuda', \
            f"Output should be on CUDA, got {normalized_cuda.device}"
        assert n_cuda.params_dict['scale'].device.type == 'cuda', \
            f"Parameters should be on CUDA"

        print(f"✓ CUDA device handling works correctly")

    print("\n" + "=" * 60)
    print("ALL DEPTH NORMALIZER TESTS PASSED ✅")
    print("=" * 60)


