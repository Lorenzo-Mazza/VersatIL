from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn
from pytorch_utils import dict_apply
from model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']


    @torch.no_grad()
    def fit(self,
            data: Union[Dict, torch.Tensor, np.ndarray],
            last_n_dims = 1,
            dtype = torch.float32,
            mode = 'limits',
            output_max = 1.,
            output_min = -1.,
            range_eps = 1e-4,
            fit_offset = True,
            device=None):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] = _fit(value,
                                             last_n_dims=last_n_dims,
                                             dtype=dtype,
                                             mode=mode,
                                             output_max=output_max,
                                             output_min=output_min,
                                             range_eps=range_eps,
                                             fit_offset=fit_offset,
                                             device=device)
        else:
            self.params_dict['_default'] = _fit(data,
                                                last_n_dims=last_n_dims,
                                                dtype=dtype,
                                                mode=mode,
                                                output_max=output_max,
                                                output_min=output_min,
                                                range_eps=range_eps,
                                                fit_offset=fit_offset,
                                                device=device            )


    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


    def __getitem__(self, key: str):
        params = self.params_dict[key]
        # Check if this is a sequential normalizer by looking for the metadata
        if '_is_sequential' in params and params['_is_sequential'].item() == 1.0:
            # Reconstruct SequentialNormalizer from stored params
            num_normalizers = int(params['_num_normalizers'].item())
            normalizers = []

            for i in range(num_normalizers):
                # Reconstruct each normalizer's params
                scale = params[f'_norm_{i}_scale']
                offset = params[f'_norm_{i}_offset']
                stats = {}
                for k in ['min', 'max', 'mean', 'std']:
                    if f'_norm_{i}_stat_{k}' in params:
                        stats[k] = params[f'_norm_{i}_stat_{k}']

                norm = SingleFieldLinearNormalizer.create_manual(
                    scale=scale,
                    offset=offset,
                    input_stats_dict=stats
                )
                normalizers.append(norm)

            return SequentialNormalizer(normalizers=normalizers)
        else:
            return SingleFieldLinearNormalizer(params)


    def __setitem__(self, key: str, value):
        if isinstance(value, SequentialNormalizer):
            # Extract normalizers list
            normalizer_list = list(value.normalizers) if hasattr(value.normalizers, '__iter__') else []

            # Create enhanced params_dict with metadata for reconstruction
            params_dict = nn.ParameterDict()

            # Store the effective scale/offset from SequentialNormalizer
            params_dict['scale'] = value.params_dict['scale'].clone().detach()
            params_dict['offset'] = value.params_dict['offset'].clone().detach()

            # Store input stats from first normalizer
            input_stats = normalizer_list[0].params_dict['input_stats']
            params_dict['input_stats'] = nn.ParameterDict({
                k: v.clone().detach() for k, v in input_stats.items()
            })

            # Store metadata
            params_dict['_is_sequential'] = nn.Parameter(torch.tensor([1.0]))
            params_dict['_num_normalizers'] = nn.Parameter(torch.tensor([float(len(normalizer_list))]))

            # Store individual normalizer params for reconstruction
            for i, n in enumerate(normalizer_list):
                params_dict[f'_norm_{i}_scale'] = n.params_dict['scale'].clone().detach()
                params_dict[f'_norm_{i}_offset'] = n.params_dict['offset'].clone().detach()
                for k, v in n.params_dict['input_stats'].items():
                    params_dict[f'_norm_{i}_stat_{k}'] = v.clone().detach()

            # Ensure no gradients
            for p in params_dict.parameters():
                p.requires_grad_(False)

            self.params_dict[key] = params_dict
        else:
            # Regular SingleFieldLinearNormalizer
            self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward = True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)


    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=True)


    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> Union[Dict, torch.Tensor]:
        return self._normalize_impl(x, forward=False)


    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']

        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key = '_default'):
        input_stats = self.params_dict['input_stats']
        scale = self.params_dict['scale']
        offset = self.params_dict['offset']
        dict_to_return = {}
        if 'min' in input_stats:
            dict_to_return['min'] = self.normalize(input_stats['min'])
        if 'max' in input_stats:
            dict_to_return['max'] = self.normalize(input_stats['max'])
        if 'mean' in input_stats:
            dict_to_return['mean'] = self.normalize(input_stats['mean'])
        if 'std' in input_stats:
            dict_to_return['std'] = torch.abs(scale) * input_stats['std']
        return dict_to_return



class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']


    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray],
            last_n_dims = 1,
            dtype = torch.float32,
            mode = 'limits',
            output_max = 1.,
            output_min = -1.,
            range_eps = 1e-10,
            fit_offset = True,
            device='cpu'):
        self.params_dict = _fit(data,
                                last_n_dims=last_n_dims,
                                dtype=dtype,
                                mode=mode,
                                output_max=output_max,
                                output_min=output_min,
                                range_eps=range_eps,
                                fit_offset=fit_offset,
                                device=device)



    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj


    @classmethod
    def create_manual(cls,
                      scale: Union[torch.Tensor, np.ndarray],
                      offset: Union[torch.Tensor, np.ndarray],
                      input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                      ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x


        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)


    @classmethod
    def create_identity(cls, dtype = torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)


    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)


    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)


    def get_input_stats(self):
        return self.params_dict['input_stats']


    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)


    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)


def _fit(data: Union[torch.Tensor, np.ndarray],
         last_n_dims = 1,
         dtype = torch.float32,
         mode = 'limits',
         output_max = 1.,
         output_min = -1.,
         range_eps = 1e-10,
         fit_offset = True,
         device = None):  # Add device parameter

    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)
    if device is not None:
        data = data.to(device)  # Move data to device before computing stats

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)
    if mode == 'gaussian':
        input_std = torch.clamp(input_std, min=1e-2)  # avoid too small std

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)

    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward = True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x



class SequentialNormalizer(SingleFieldLinearNormalizer):
    """
    A normalizer that applies a sequence of normalizers. The normalizers must be instances of SingleFieldLinearNormalizer
     and be already fitted.
    """
    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = nn.ModuleList(normalizers)
        if len(self.normalizers) == 0:
            raise ValueError("At least one normalizer required")
        # Compute effective scale and offset
        eff_scale = self.normalizers[0].params_dict['scale'].clone().detach()
        eff_offset = self.normalizers[0].params_dict['offset'].clone().detach()
        for n in self.normalizers[1:]:
            s = n.params_dict['scale'].clone().detach()
            o = n.params_dict['offset'].clone().detach()
            eff_offset = o + s * eff_offset
            eff_scale = s * eff_scale

        input_stats = self.normalizers[0].params_dict['input_stats']
        self.params_dict = nn.ParameterDict({
            'scale': nn.Parameter(eff_scale),
            'offset': nn.Parameter(eff_offset),
            'input_stats': nn.ParameterDict({k: nn.Parameter(v.clone().detach()) for k, v in input_stats.items()})
        })
        for p in self.params_dict.parameters():
            p.requires_grad_(False)


    def fit(self, *args, **kwargs):
        raise NotImplementedError("SequentialNormalizer does not support fitting")

    def normalize(self, x):
        """ Apply all normalizers in sequence."""
        for n in self.normalizers:
            x = n.normalize(x)
        return x

    def unnormalize(self, x):
        """ Apply all normalizers in reverse sequence."""
        for n in reversed(self.normalizers):
            x = n.unnormalize(x)
        return x

    def get_input_stats(self):
        """ Get input stats of the first normalizer."""
        return self.normalizers[0].get_input_stats()

    def get_output_stats(self):
        """ Get output stats of the last normalizer."""
        stats = self.get_input_stats()
        for n in self.normalizers:
            scale = n.params_dict['scale']
            offset = n.params_dict['offset']
            stats = {
                'min': scale * stats['min'] + offset,
                'max': scale * stats['max'] + offset,
                'mean': scale * stats['mean'] + offset,
                'std': torch.abs(scale) * stats['std']
            }
        return stats





def test():
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100, 10, 9, 2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    # dict
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        'action': torch.zeros((1000, 128, 2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)


    data = torch.zeros((100, 10, 9, 2)).uniform_()*10

    output1_max=1.
    output1_min=0.
    input_min = data.min()
    input_max = data.max()
    input_mean = data.mean()
    input_std = data.std()
    input_stat1 = {
        'min': torch.tensor([input_min], dtype=torch.float32, device='cpu'),
        'max': torch.tensor([input_max], dtype=torch.float32, device='cpu'),
        'mean': torch.tensor([input_mean], dtype=torch.float32, device='cpu'),
        'std': torch.tensor([input_std], dtype=torch.float32, device='cpu'),
    }
    scale1 = (output1_max - output1_min) / (input_max - input_min)
    offset1 = output1_min - scale1 * input_min
    scale1 = torch.tensor([scale1], dtype=torch.float32, device='cpu')
    offset1 = torch.tensor([offset1], dtype=torch.float32, device='cpu')

    imagenet_mean= 0.456
    imagenet_std=0.21
    input_mean2 = input_mean * scale1 + offset1
    input_std2 = input_std * scale1
    input_stat2 = {
        'min': torch.tensor([output1_min], dtype=torch.float32, device='cpu'),
        'max': torch.tensor([output1_max], dtype=torch.float32, device='cpu'),
        'mean': torch.tensor([input_mean2], dtype=torch.float32, device='cpu'),
        'std': torch.tensor([input_std2], dtype=torch.float32, device='cpu'),
    }
    scale2 = torch.tensor([1.0 / imagenet_std], dtype=torch.float32, device='cpu')
    offset2 = torch.tensor([-imagenet_mean / imagenet_std], dtype=torch.float32, device='cpu')
    normalizer1 = SingleFieldLinearNormalizer.create_manual(scale=scale1, offset=offset1, input_stats_dict=input_stat1)
    normalizer2 = SingleFieldLinearNormalizer.create_manual(scale=scale2, offset=offset2, input_stats_dict=input_stat2)
    normalizer = SequentialNormalizer(normalizers=[normalizer1, normalizer2])
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    expected_mean = (input_mean2 - imagenet_mean) / imagenet_std
    expected_std = input_std2 / imagenet_std
    assert torch.allclose(datan.mean().detach(), torch.tensor(expected_mean))
    assert torch.allclose(datan.std().detach(), torch.tensor(expected_std))
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-6)
    datan1 = normalizer1.normalize(data)
    datan2 = normalizer2.normalize(datan1)
    assert torch.allclose(datan2, datan)