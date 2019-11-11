from provenance.hashing import hash
from conftest import artifact_record
import provenance.utils as u
import provenance.repos as r
import provenance.core as pc
import provenance as p
import conftest as c

import toolz as t
import pandas as pd
import numpy as np

import cloudpickle as pickle
import os
import random
import shutil
import tempfile
from copy import copy, deepcopy

import pytest
pytest.importorskip("torch")


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


@p.provenance(returns_composite=True)
def random_data(N=64, D_in=1000, H=100, D_out=10):
    """
    N is batch size
    D_in is input dimension
    H is hidden dimension
    D_out is output dimension
    """

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    return {
        'X_train': x,
        'Y_train': y,
        'X_test': x,
        'Y_test': y
    }


@p.provenance(returns_composite=True)
def fit_model(model,
              x,
              y,
              batch_size=32,
              epochs=500,):

    # Construct our loss function and an Optimizer. The call to
    # model.parameters() in the SGD constructor will contain the learnable
    # parameters of the two nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    losses = []
    for t in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {'model': model, 'losses': losses}


@p.provenance()
def basic_model(D_in=1000, H=100, D_out=10):
    return TwoLayerNet(D_in=D_in, H=H, D_out=D_out)


def test_integration_pytorch_test(dbdiskrepo):

    data = random_data()
    model = basic_model()
    model2 = model
    assert model2.artifact.id == compiled_model.artifact.id
    assert hash(compiled_model2) == hash(compiled_model)

    fitted_model = fit_model(compiled_model, data['X_train'], data['Y_train'])

    assert model.artifact.id != compiled_model.artifact.id
    assert compiled_model.artifact.id != fitted_model.artifact.id
    assert fitted_model.artifact.value_id == p.hash(
        fitted_model.artifact.value)

    model2 = basic_model()
    assert model2.artifact.id == model.artifact.id
    assert hash(model2) == hash(model)

    compiled_model3 = compile_model(model2)
    assert compiled_model3.artifact.id == compiled_model.artifact.id

    fitted_model2 = fit_model(compiled_model3, data['X_train'],
                              data['Y_train'])

    assert fitted_model2.artifact.id == fitted_model.artifact.id

    # now see if a model that is modified in place ends up with the same hash
    model3 = basic_model()
    assert hash(model3) == hash(model)
    model3.compile(**DEFAULT_COMPILE_OPTS)
    assert hash(model3) == hash(compiled_model)
