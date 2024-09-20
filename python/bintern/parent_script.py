import subprocess
import sys

def run_script_n_times(script_name, n):
    # clean both output files:


    for i in range(n):
        print(f"Running {script_name}, Iteration {i+1}/{n}")
        try:
            # Run the script as a subprocess
            subprocess.run(['python', script_name], check=True)



        except subprocess.CalledProcessError as e:
            print(f"Error while running {script_name}: {e}")
            sys.exit(1)  # Exit if an error occurs
        print(f"Finished iteration {i+1}\n")


def wipe_file(file_path):
    with open(file_path, 'w') as file:
        pass  # Opening in 'w' mode wipes the file




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please enter 3 input arguments: $ python parent_script.py <script_name> <N>")
        sys.exit(1)
    
    script_name = sys.argv[1]
    n = int(sys.argv[2])

    wipe_file('python_results.txt')
    wipe_file('init_for_cuda.txt')
    run_script_n_times(script_name, n)