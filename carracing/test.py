from mpi4py import MPI
import sys
import subprocess

mpi_warn_on_fork = 0

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

for i in range(size):
  if rank == i:
    command = "python hello.py %d %d %s" % (rank, size, name)
    subprocess.call(command, shell = True)