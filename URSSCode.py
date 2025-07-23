import numpy as np
import itertools
import galois

import time
start = time.time()
np.set_printoptions(threshold=np.inf)

# Program begins ---------------------------------

# d(F, n, k)
charF = 2 # prime-order finite field only
n = 4
k = 2
dim_W = 3 # all 1-dimensional subspaces are isotropic for an alternating bilinear form - this should be at least 2

relevant_ents = ((n*(n - 1)) // 2)
space_size = (charF**n) - 1
GF = galois.GF(charF)

forms = [np.zeros((n, n), dtype = int) for i in range(k)]
mat_ents = np.zeros(( relevant_ents , 2 ), dtype = int)                                             # entries for a single form
all_ents = np.zeros(((relevant_ents*k) , 2), dtype = int)                                           # enumeration of [form, entry]
vectors = [np.zeros(n, dtype = int) for i in range(space_size)]                                     # all nonzero vectors in the vector space F^n
W_bases_L = list(itertools.combinations(range(space_size), dim_W)); W_bases = np.array(W_bases_L)   # index combinations in 'vectors'
pair_comps_L = list(itertools.combinations(range(dim_W), 2)); pair_comps = np.array(pair_comps_L)   # pairs to be 'dotted' to check if each subspace is isotropic. All dimension-1 subspaces are isotropic

a = 0 # Fill mat_ents
for i in range(0, n - 1):
    for j in range(i + 1, n):
        mat_ents[a, 0] = i
        mat_ents[a, 1] = j
        a += 1

a = 0 # Fill all_ents --- [depth, 0] - forms enumerated,   [depth, 1] - matrix entries enumerated
for i in range(0, k):
    for j in range(0, relevant_ents):
        all_ents[a, 0] = k - i - 1
        all_ents[a, 1] = relevant_ents - j - 1
        a += 1

a = -1 # Fill vectors
test_vector = np.zeros(n, dtype = int)
def generate_vectors(depth):
    global a
    if depth == -1:
        if a >= 0:
            for i in range(n):
                vectors[a][i] = test_vector[i]
            a += 1
        else:
            a += 1
        return
    for element in range(0, charF):
        test_vector[depth] = element
        generate_vectors(depth - 1)
        test_vector[depth] = 0
    return
generate_vectors(n - 1)

# ensure that each set from W_bases is linearly independent
for i in range(len(W_bases)):
    columns = [vectors[W_bases[i][j]] for j in range(dim_W)]
    M = GF(columns); rref = M.row_reduce()
    rank = sum(any(x != 0 for x in row) for row in rref)
    if rank < dim_W: # discard (set to zeroes) sets with linear dependencies
        W_bases[i] = np.zeros(dim_W)
    
W_checked = 0
forms_checked = 0
iso_dim_W = 0
def search(depth):
    if depth == -1:                                      # i.e. we have a completed k-tuple of forms ready for testing
        global forms_checked; forms_checked += 1
        for basis in W_bases:                            # check each subspace to see if it is isotropic for all the forms:
            if np.array_equal(basis, np.zeros(dim_W)):   # ignore linearly dependent set (replaced with zeroes earlier)
                continue
            x = 0
            global W_checked; W_checked += 1             # count up the checked subspaces
            for form in forms:                           # check if the subspace is isotropic for each form in the k-tuple:
                for pair in pair_comps:                  # do this by comparing all distinct pairs of basis vectors
                    dot = (vectors[basis[pair[0]]] @ form @ vectors[basis[pair[1]]]) % charF
                    if dot != 0:
                        x += 1                           # W is isotropic if and only if x stays at 0, i.e. alpha(v, w) = 0 for all distinct pairs of W-basis elements
            if x == 0:                                   # m(alpha) is a maximum, so if we have a single isotropic subspace of dim dim_W, we have success for the k-tuple
                global iso_dim_W; iso_dim_W += 1
                return
        #print(forms)                                     # display any counterexamples
        return
    for element in range(0, charF):
        forms[ all_ents[depth, 0] ][ mat_ents[all_ents[depth, 1], 0] , mat_ents[all_ents[depth, 1], 1] ] = element
        forms[ all_ents[depth, 0] ][ mat_ents[all_ents[depth, 1], 1] , mat_ents[all_ents[depth, 1], 0] ] = -element # Alternating
        search(depth - 1)
        forms[ all_ents[depth, 0] ][ mat_ents[all_ents[depth, 1], 0] , mat_ents[all_ents[depth, 1], 1] ] = 0
        forms[ all_ents[depth, 0] ][ mat_ents[all_ents[depth, 1], 1] , mat_ents[all_ents[depth, 1], 0] ] = 0
    return

#print(vectors)
#print(W_bases)
#print(pair_comps)

search((relevant_ents * k) - 1)

print("Dimension of vector space:", n)
print("Collections of", k , "alternating bilinear forms checked:",forms_checked)
print(f"Of which have a {dim_W}-dimensional isotropic subspace:" , iso_dim_W)
print("Total subspace checks completed:" , W_checked)

# Program ends ---------------------------------

end = time.time()
print("Runtime:", end - start, "seconds")