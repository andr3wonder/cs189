import numpy as np

np.random.seed(10)

## Q6.1
# gen a random matrix of 5x5
A = np.random.rand(5, 5)
np_trace = np.trace(A) # with np
einsum_trace = np.einsum("ii", A) # with einsum (i = j, which is diag terms)

print("Trace Difference Norm:", np.linalg.norm(np_trace - einsum_trace))


## Q6.1.2
B = np.random.rand(5, 5)
C = np.random.rand(5, 5)
np_product = np.dot(B, C)
einsum_product = np.einsum("ij,jk->ik", B, C)
print("Product Difference Norm:", np.linalg.norm(np_product - einsum_product))


## Q6.1.3
batch_size = 3 # i
D = np.random.rand(batch_size, 4, 5) # ijk
E = np.random.rand(batch_size, 5, 6) # ikl 
np_batch_product = np.matmul(D, E)
einsum_batch_product = np.einsum("ijk,ikl->ijl", D, E)
print(
    "Batch Product Difference Norm:",
    np.linalg.norm(np_batch_product - einsum_batch_product),
)
