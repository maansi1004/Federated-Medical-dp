import subprocess
import time
import sys

def launch_federated_system():
    print("[SYSTEM] Starting Central Server...")
    # 1. Start the server in the background
    server_process = subprocess.Popen([sys.executable, "server.py"])
    
    # Give the server 3 seconds to boot up and start listening on port 8080
    time.sleep(8)
    
    client_processes = []
    print("[SYSTEM] Spawning 3 Hospital Client Nodes in the background...")
    
    # 2. Launch Client Node 0, 1, and 2 automatically
    for node_id in range(3):
        print(f" -> Launching Hospital Node {node_id}")
        p = subprocess.Popen([sys.executable, "client.py", "--node", str(node_id)])
        client_processes.append(p)
        time.sleep(1) # Small pause to avoid CPU spikes
        
    print("\n[SYSTEM] Network running! Watch your logs below...\n")

    # 3. Keep the main script alive until all clients finish training (after 3 rounds)
    for p in client_processes:
        p.wait()
        
    # 4. Clean up and shut down the server when finished
    server_process.terminate()
    print("\n[SYSTEM] Federated simulation complete. Server terminated smoothly.")

if __name__ == "__main__":
    launch_federated_system()