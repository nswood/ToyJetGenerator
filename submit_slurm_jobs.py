# submit_slurm_jobs.py
import os
import subprocess

# nprongs = [1, 2]
# nparts = [32]
# num_samples = 100

nprongs = [1, 2, 3, 4]
nparts = [32]
num_samples = 5000
output_folder = "Toy_resonances_datasets"
job_template = """#!/bin/bash
#SBATCH --partition=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --output=slurm_monitoring/%x-%j.out

source ~/.bashrc
source /n/holystore01/LABS/iaifi_lab/Users/nswood/mambaforge/etc/profile.d/conda.sh
conda activate flat-samples

python pre_process_gromov_data.py --nprong {nprong} --npts {npt} --num_samples {num_samples} --output_folder {output_folder} --file_id {file_id}
"""

for nprong in nprongs:
    for npt in nparts:
        for batch in range(10):  # 10 batches of 5000 samples each to make 50,000 samples
            file_id = f"{batch}"
            job_script = job_template.format(
                nprong=nprong,
                npt=npt,
                num_samples=num_samples,
                output_folder=output_folder,
                file_id=file_id
            )
            job_file = f"slurm/job_nprong{nprong}_npt{npt}_batch{batch}.slurm"
            with open(job_file, "w") as f:
                f.write(job_script)
            subprocess.run(["sbatch", job_file])
            os.remove(job_file)