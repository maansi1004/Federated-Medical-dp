import subprocess
import time
import sys

def run_system():
    print("🚀 Initializing Federated Cluster...")
    
    # 1. Start Server in the background
    server_process = subprocess.Popen([sys.executable, "server.py"])
    
    # Wait for server framework to bind to port 8080
    time.sleep(3)
    
    # 2. Launch all 3 nodes concurrently
    client_processes = []
    for node_id in [0, 1, 2]:
        print(f"📦 Launching Hospital Node {node_id}...")
        p = subprocess.Popen([sys.executable, "client.py", "--node", str(node_id)])
        client_processes.append(p)
        time.sleep(1) # Stagger client initialization slightly
        
    print("✨ System fully active. Processing local rounds...")
    
    try:
        # Keep this orchestrator script alive while training runs
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down cluster gracefully...")
        server_process.terminate()
        for p in client_processes:
            p.terminate()

if __name__ == "__main__":
    run_system()