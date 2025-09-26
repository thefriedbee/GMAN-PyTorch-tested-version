"""
Test the model sub-modules/components by testing output shapes...
"""
import argparse

import torch
import torch.nn as nn
from . import model as my_model


def test_conv2d_():
    """Test conv2d_ function."""
    # the traffic case: [BS, T, N, D]
    input_tensor = torch.randn(8, 3, 7, 10)
    model = my_model.conv2d_(
        input_dims=10, 
        output_dims=16, 
        kernel_size=1, 
        padding='SAME', 
        use_bias=True, 
        activation=nn.ReLU(), 
        bn_decay=0.9
    )
    output = model(input_tensor)
    assert output.shape == (8, 3, 7, 16)


def test_fc():
    # the traffic case: [BS, T, N, D]
    input_tensor = torch.randn(10, 3, 7, 5)
    # although called conv2d_, kernel sizes and stride sizes are both 1
    # it means no communication between the spatial and temporal dimensions
    # so it is actually a fully connected layer within the channel dimension
    model = my_model.FC(
        input_dims=5, 
        units=8, 
        activations=nn.ReLU(), 
        bn_decay=0.9, 
        use_bias=True
    )
    output = model(input_tensor)
    assert output.shape == (10, 3, 7, 8)
    # after ReLU, all values should be non-negative
    assert torch.all(output >= 0)


def test_st_embedding():
    # [N, D]
    SE = torch.randn(10, 5)
    # [BS, T, 2]
    TE = torch.randn(8, 24, 2)
    model = my_model.STEmbedding(D_se=5, D_out=5, bn_decay=0.9)
    output = model(SE, TE)
    # [BS, T, N, D]
    assert output.shape == (8, 24, 10, 5)


def test_spatial_attention():
    # [BS, T, N, D]
    X = torch.randn(3, 24, 10, 32)
    # [BS, T, N, D]
    STE = torch.randn(3, 24, 10, 32)
    model = my_model.spatialAttention(K=4, d=8, bn_decay=0.9)
    output = model(X, STE)
    assert output.shape == (3, 24, 10, 32)


def test_temporal_attention():
    # [BS, T, N, D]
    X = torch.randn(3, 24, 10, 32)
    # [BS, T, N, D]
    STE = torch.randn(3, 24, 10, 32)
    model = my_model.temporalAttention(K=4, d=8, bn_decay=0.9)
    output = model(X, STE)
    assert output.shape == (3, 24, 10, 32)


def test_gated_fusion():
    # [BS, T, N, D]
    HS = torch.randn(3, 24, 10, 32)
    HT = torch.randn(3, 24, 10, 32)
    model = my_model.gatedFusion(D=32, bn_decay=0.9)
    output = model(HS, HT)
    assert output.shape == (3, 24, 10, 32)


def test_st_att_block():
    # [BS, T, N, D]
    X = torch.randn(3, 24, 10, 32)
    STE = torch.randn(3, 24, 10, 32)
    model = my_model.STAttBlock(K=4, d=8, bn_decay=0.9)
    output = model(X, STE)
    assert output.shape == (3, 24, 10, 32)


def test_transform_attention():
    # [BS, T_his, N, D]
    X = torch.randn(3, 48, 10, 32)
    STE_his = torch.randn(3, 48, 10, 32)
    STE_pred = torch.randn(3, 24, 10, 32)
    model = my_model.transformAttention(K=4, d=8, bn_decay=0.9)
    output = model(X, STE_his, STE_pred)
    assert output.shape == (3, 24, 10, 32)


def test_gman():
    X = torch.randn(3, 48, 10)
    TE = torch.randn(3, 48+24, 2)
    SE = torch.randn(10, 32)
    args = argparse.Namespace()
    args.L = 3
    args.K = 4
    args.d = 8
    args.num_his = 48
    args.num_pred = 24
    model = my_model.GMAN(SE=SE, args=args, bn_decay=0.9)
    output = model(X, TE)
    assert output.shape == (3, 24, 10)
