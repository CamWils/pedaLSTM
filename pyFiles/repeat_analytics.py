import subprocess
import sys

def run_script(num_times, num_epochs):
    command = ['python3', 'pedaLSTMlite_auto_analytics.py', f'--num_epochs={num_epochs}']
    
    for i in range(num_times):
        print(f"Running iteration {i + 1}/{num_times} with num_epochs={num_epochs}...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error during iteration {i + 1}:")
            print(result.stderr)
        else:
            print(f"Iteration {i + 1} completed successfully.")
            #print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_multiple_times.py <num_times> <num_epochs>")
        sys.exit(1)
    
    try:
        num_times = int(sys.argv[1])
        num_epochs = int(sys.argv[2])
    except ValueError:
        print("Both parameters <num_times> and <num_epochs> should be integers.")
        sys.exit(1)
    
    run_script(num_times, num_epochs)
