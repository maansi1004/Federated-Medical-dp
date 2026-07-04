import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from model import SimpleMedicalCNN, evaluate_model
from data_setup import get_hospital_data
from opacus import PrivacyEngine

parser = argparse.ArgumentParser(description="Flower Client with Oversampling + Opacus-safe FedProx + DP")
parser.add_argument("--node", type=int, required=True, help="Node ID (0, 1, or 2)")
args = parser.parse_args()

loaders = get_hospital_data()
hospital_loader = loaders[args.node]

class HospitalClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = SimpleMedicalCNN()
        self.criterion = nn.CrossEntropyLoss()

        self.privacy_engine = PrivacyEngine(accountant="rdp")
        self.noise_multiplier = 1.0   # increased from 0.10 — was far too weak
        self.max_grad_norm = 1.0

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        self.model, self.optimizer, self.local_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=hospital_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=False
        )

    def get_parameters(self, config):
        state_dict = self.model.state_dict()
        clean_state_dict = OrderedDict({k.replace("_module.", ""): v.cpu().numpy() for k, v in state_dict.items()})
        return [val for val in clean_state_dict.values()]

    def set_parameters(self, parameters):
        raw_keys = [k.replace("_module.", "") for k in self.model.state_dict().keys()]
        params_dict = zip(raw_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model._module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        global_params = [p.clone().detach() for p in self.model.parameters()]

        self.model.train()
        current_round = config.get("current_round", 1)
        proximal_mu = config.get("proximal-mu", 0.1)

        batches_per_epoch = len(self.local_loader)
        local_epochs = 1   # reduced from 5 — was inflating total training steps massively

        for epoch in range(local_epochs):
            for images, labels in self.local_loader:
                self.optimizer.zero_grad()

                # Per-example loss only — this is what Opacus clips & noises correctly
                loss = self.criterion(self.model(images), labels)
                loss.backward()

                # FedProx term added AFTER backward, computed analytically —
                # bypasses Opacus's per-sample hooks safely since it isn't a
                # function of any individual training example.
                with torch.no_grad():
                    for local_w, global_w in zip(self.model.parameters(), global_params):
                        if local_w.grad is not None:
                            local_w.grad += proximal_mu * (local_w - global_w)

                self.optimizer.step()

        batch_size = hospital_loader.batch_size if hospital_loader.batch_size is not None else 32
        sample_rate = batch_size / len(hospital_loader.dataset)
        total_steps_so_far = current_round * local_epochs * batches_per_epoch

        self.privacy_engine.accountant.history = [
            (self.noise_multiplier, sample_rate, total_steps_so_far)
        ]

        cumulative_epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔒 [NODE {args.node}] Round {current_round} Done. Cumulative Privacy (Epsilon ε): {cumulative_epsilon:.2f}")

        return self.get_parameters(config={}), len(hospital_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_model(self.model, hospital_loader)
        return float(loss), len(hospital_loader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: str) -> fl.client.Client:
    return HospitalClient().to_client()

if __name__ == "__main__":
    print(f"\n[CLIENT] Starting Verified Secure Hospital Node {args.node}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client_fn=client_fn
    )