import pulp as pp
import numpy as np
import itertools

#####################
#  Problem Data:    #
#####################

full_matrix_size = 10
submatrix_size = 6

A = np.random.random((full_matrix_size, full_matrix_size)).round(2)

inds = range(full_matrix_size)
product_inds = list(itertools.product(inds,inds))

#####################
#  Variables:       #
#####################

# x[(i,j)] = 1 if the (i,j)th element of the data matrix is in the submatrix, 0 otherwise.
x = pp.LpVariable.dicts('x', product_inds, cat='Continuous', lowBound=0, upBound=1)

# y[i] = 1 if i is in the selected index set, 0 otherwise.
y = pp.LpVariable.dicts('y', inds, cat='Binary')

prob = pp.LpProblem("submatrix_problem", pp.LpMaximize)

#####################
#  Constraints:     #
#####################

# The following constraints express the required submatrix shape:
for (i,j) in product_inds:
    # x[(i,j)] must be 1 if y[i] and y[j] are both in the selected index set.
    prob += pp.LpConstraint(e=x[(i,j)] - y[i] - y[j], sense=1, rhs=-1,
                            name="true_if_both_%s_%s" % (i,j))

    # x[(i,j)] must be 0 if y[i] is not in the selected index set.
    prob += pp.LpConstraint(e=x[(i,j)] - y[i], sense=-1, rhs=0,
                            name="false_if_not_row_%s_%s" % (i,j))

    # x[(i,j)] must be 0 if y[j] is not in the selected index set.
    prob += pp.LpConstraint(e=x[(i,j)] - y[j], sense=-1, rhs=0,
                            name="false_if_not_col_%s_%s" % (i,j))

# The number of selected indices must be what we require:    
prob += pp.LpConstraint(e=pp.LpAffineExpression([(y[i],1) for i in inds]), sense=0,
                        rhs=submatrix_size, name="submatrix_size")

#####################
#  Objective:       #
#####################

prob += pp.LpAffineExpression([(x[pair], A[pair]) for pair in product_inds])

print(prob)

########################
#  Create the problem: #
########################

prob.writeLP("max_sum_submatrix.lp")
prob.solve()

########################## 
#  Display the solution: #
##########################
print("The following indices were selected:")
print([v.name for v in prob.variables() if v.name[0]=='y' and  v.varValue==1])
print("Objective value is " + str(pp.value(prob.objective)))