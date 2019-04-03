
#!/bin/sh -l

#SBATCH -p regular
#SBATCH -N 1
#SBATCH -t 03:00:00

module load python/3.6-anaconda
srun -n 1 python clementbatch.py