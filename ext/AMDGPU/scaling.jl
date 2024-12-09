KP.scaling_csr!(A::rocSPARSE.ROCSparseMatrixCSR, b::ROCVector) = scaling_csr!(A, b, ROCBackend())
