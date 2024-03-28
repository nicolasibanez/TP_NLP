#!/usr/bin/python

import os
import sys
import subprocess
import tempfile

def get_available_nodes():
    """Returns a list of available nodes within the specified ranges."""
    print("Checking for available nodes...")
    try:
        # Execute squeue and get the current jobs
        result = subprocess.run(['squeue', '--format=%N'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully retrieved node information.")
            # Parse the output to get a list of nodes currently in use
            used_nodes = result.stdout.split()
            print(f"Currently used nodes: {used_nodes}")
            # all_nodes = [f'sh0{i}' for i in range(1, 10)] #+ [f'sh2{i}' for i in range(0, 3)]
            all_nodes = [f'sh0{i}' for i in [1, 2, 3, 4, 6]] + [f'sh2{i}' for i in [0, 1, 2]]
            # Determine available nodes by removing used nodes from the all_nodes list
            available_nodes = [node for node in all_nodes if node not in used_nodes]
            print(f"Available nodes: {available_nodes}")
            return available_nodes
        else:
            print("Failed to get node information")
            return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def makejob(commit_id, configpath, nruns, available_nodes = None):
    if available_nodes:
        node = available_nodes[0]  # Select the first available node for simplicity
        print(f"Selected node for job: {node}")
        pre_job_script = f"""#!/bin/bash

#SBATCH --job-name=nlp
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}
#SBATCH --nodelist={node}"""
    else:
        pre_job_script = f"""#!/bin/bash
#SBATCH --job-name=nlp
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}
#SBATCH --nodelist=sh03"""
     

    return pre_job_script + f"""
current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

source ../TP2/venvtp1/bin/activate

echo "Training"
python3 main.py --mode train

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# # Ensure all the modified files have been staged and commited
# # This is to guarantee that the commit id is a reliable certificate
# # of the version of the code you want to evaluate
# result = int(
#     subprocess.run(
#         "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
#         shell=True,
#         stdout=subprocess.PIPE,
#     ).stdout.decode()
# )
# if result > 0:
#     print(f"We found {result} modifications either not staged or not commited")
#     raise RuntimeError(
#         "You must stage and commit every modification before submission "
#     )

# commit_id = subprocess.check_output(
#     "git log --pretty=format:'%H' -n 1", shell=True
# ).decode()

# print(f"I will be using the commit id {commit_id}")
commit_id = None

# Ensure the log directory exists
os.system("mkdir -p logslurms")

# if len(sys.argv) not in [2, 3]:
#     print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
#     sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
available_nodes = get_available_nodes()
# submit_job(makejob(commit_id, tmp_configfilepath, nruns, available_nodes=None))
submit_job(makejob(commit_id, tmp_configfilepath, nruns, available_nodes=available_nodes))
