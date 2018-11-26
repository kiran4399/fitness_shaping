from pulp import *


# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])
# Output= 
# Status: Optimal

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)
# Output=
# Medicine_1_units = 3.0
# Medicine_2_units = 4.0

# The optimised objective function value is printed to the screen
print("Total Health that can be restored = ", value(prob.objective))
# Output= 
# Total Health that can be restored =  155.0