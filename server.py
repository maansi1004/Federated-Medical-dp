import flwr as fl
import torch
from collections import OrderedDict
from model import SimpleMedicalCNN
# Change this line at the top of client.py, server.py, and train_baseline.py:
# from model import PrivateTransferMedicalCNN as SimpleMedicalCNN
from typing import List, Tuple, Dict

# --- Custom Aggregation Strategy preventing Node Volume Bias with FedProx Configs ---
class EqualWeightFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], failures: List[BaseException]):
        if not results:
            return None, {}
        
        # Read updates from all clients
        weights_results = [fl.common.parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        
        # Calculate a pure unweighted arithmetic mean across all reporting hospitals
        num_clients = len(weights_results)
        averaged_weights = [
            sum(layer_updates) / num_clients 
            for layer_updates in zip(*weights_results)
        ]
        
        aggregated_parameters = fl.common.ndarrays_to_parameters(averaged_weights)
        
        # Save checkpoints on final execution round (Updated to match target round)
        if server_round == 10 and aggregated_parameters is not None:
            print("\n[SERVER] Final Round Complete! Saving Global Model Weights...")
            model = SimpleMedicalCNN()
            params_dict = zip(model.state_dict().keys(), averaged_weights)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            torch.save(state_dict, "federated_model.pth")
            print("[SERVER] Success! 'federated_model.pth' saved.")
            
        return aggregated_parameters, {}

def aggregate_evaluate_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    total_examples = sum([num_examples for num_examples, _ in metrics])
    weighted_accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    return {"global_accuracy": sum(weighted_accuracies) / total_examples}

def fit_config(server_round: int):
    return {
        "current_round": server_round,
        # --- PHASE 3 FIX: Broadcast leash stiffness down to the local nodes ---
        "proximal-mu": 0.1  
    }

if __name__ == "__main__":
    print("\n[SERVER] Launching Equal-Weight Centralized Aggregation Server with FedProx Hooks...")
    strategy = EqualWeightFedAvg(
        min_fit_clients=3,         
        min_available_clients=3,   
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
        on_fit_config_fn=fit_config
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10), # Extended for stable DP convergence
        strategy=strategy,
    )