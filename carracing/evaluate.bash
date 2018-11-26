for i in `seq 1 1`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- mpirun -np 64 python evaluate_descent.py > evaluate.txt &
  # on macbook for debugging:
  #python extract.py &
  sleep 1.0
done
