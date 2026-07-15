import subprocess
import sys

def main():
    frontend_cmd = [sys.executable, "-m", "http.server", "8000", "--directory", "kernel/static"]
    backend_cmd = [sys.executable, "-m", "kernel.server"]

    processes = []
    
    try:
        print("Starting frontend on http://localhost:8000")
        processes.append(subprocess.Popen(frontend_cmd))
        
        print("Starting backend on ws://localhost:8765")
        processes.append(subprocess.Popen(backend_cmd))

        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()

if __name__ == "__main__":
    main()
