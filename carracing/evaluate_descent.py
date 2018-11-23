from mpi4py import MPI
import sys
import subprocess

mpi_warn_on_fork = 0

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

filelist = ["log/carracing.cma.16.64.best.json", "log/fitnessai.json"]
for i in range(size):
  if rank == i:
  	command = 'xvfb-run -a -s ' + '"-screen 0 1400x900x24 +extension RANDR"' + '-- python3 model.py render ' + filelist[i]
  	subprocess.call(command, shell = True)