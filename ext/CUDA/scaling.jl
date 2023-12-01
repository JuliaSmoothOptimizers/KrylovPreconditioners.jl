scaling_csr(A::CUSPARSE.CuSparseMatrixCSR, b::CuVector) = scaling_csr!(A, b, CUDABackend())
