@kernel function scaling_csr_kernel!(rowPtr, nzVal, b)
  m = @index(Global, Linear)
  max = 0.0
  @inbounds for i = rowPtr[m]:(rowPtr[m + 1] - 1)
    absnzVal = abs(nzVal[i])
    # This works somehow better in ExaPF. Was initially a bug I thought
    # absnzVal = nzVal[i]
    if absnzVal > max
      max = absnzVal
    end
  end
  if max < 1.0
    b[m] /= max
    @inbounds for i = rowPtr[m]:(rowPtr[m + 1] - 1)
        nzVal[i] /= max
    end
  end
end


function scaling_csr!(A, b, backend::KA.Backend)
  scaling_csr_kernel!(backend)(A.rowPtr, A.nzVal, b; ndrange=length(b))
  synchronize(backend)
end
