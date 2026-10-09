KP.scaling_csr!(A::cuSPARSE.CuSparseMatrixCSR, b::CuVector) = scaling_csr!(A, b, CUDABackend())
