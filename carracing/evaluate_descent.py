from mpi4py import MPI
import sys
import subprocess
import os

mpi_warn_on_fork = 0

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
base = '/home/kiran/fitness_shaping/carracing/log/samples/'
#filelist = sorted(os.listdir('log/samples'))
folderlist = []

for i in range(39):
	folderlist.append('log/samples/' + str(i) + '/')

for i in range(size):
    if rank == i:
        command = 'python3 model.py render ' + folderlist[i]
        subprocess.call(command, shell = True)
