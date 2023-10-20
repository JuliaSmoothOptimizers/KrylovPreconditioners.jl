function ordering_metis(A::SparseMatrixCSC)
	p, ip = Metis.permutation(A)
	return p
end

function ordering_amd(A::SparseMatrixCSC)
	p = AMD.amd(A)
	return p
end

function ordering_symamd(A::SparseMatrixCSC)
	p = AMD.symamd(A)
	return p
end

function ordering_colamd(A::SparseMatrixCSC)
	p = AMD.colamd(A)
	return p
end

function ordering_symrcm(A::SparseMatrixCSC)
	p = SymRCM.symrcm(A)
	return p
end
