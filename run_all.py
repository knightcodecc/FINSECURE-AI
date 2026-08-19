import subprocess
import sys
import webbrowser
import time
import os

def main():
    print("=== FinSecure AI Startup Tool ===")
    
    # Determine the directory where the running executable or script is located
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        exe_dir = os.path.dirname(os.path.abspath(__file__))
        
    venv_python = os.path.join(exe_dir, ".venv311", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        # Fallback to system python.exe instead of sys.executable to avoid recursive self-execution
        venv_python = "python"
        
    print(f"Using python interpreter: {venv_python}")
    print(f"Project directory: {exe_dir}")
    
    print("\n=== [1/3] Running Automated Tests ===")
    pytest_res = subprocess.run([venv_python, "-m", "pytest"], capture_output=False, cwd=exe_dir)
    if pytest_res.returncode != 0:
        print("\nWarning: Some tests did not pass. Proceeding to run server...")
    else:
        print("\nAll tests passed successfully!")
        
    print("\n=== [2/3] Starting FastAPI Server ===")
    server_process = None
    try:
        # Start FastAPI server via uvicorn
        server_process = subprocess.Popen([
            venv_python, "-m", 
            "uvicorn", "api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000"
        ], cwd=exe_dir)
        
        # Wait for uvicorn to initialize
        time.sleep(2)
        
        print("\n=== [3/3] Launching Dashboards in Web Browser ===")
        realtime_dash = os.path.join(exe_dir, "frontend", "realtime.html")
        analyst_dash = os.path.join(exe_dir, "frontend", "index.html")
        
        if os.path.exists(realtime_dash):
            print(f"Launching Real-Time Scanner: {realtime_dash}")
            webbrowser.open(f"file:///{realtime_dash.replace(os.sep, '/')}")
        else:
            print(f"Error: {realtime_dash} not found.")
            
        if os.path.exists(analyst_dash):
            print(f"Launching Analyst Dashboard: {analyst_dash}")
            webbrowser.open(f"file:///{analyst_dash.replace(os.sep, '/')}")
        else:
            print(f"Error: {analyst_dash} not found.")
            
        print("\nFinSecure AI server is active. Press Ctrl+C to terminate.")
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        if server_process:
            server_process.terminate()
            server_process.wait()
        print("Server shutdown complete.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if server_process:
            server_process.kill()

if __name__ == "__main__":
    main()
