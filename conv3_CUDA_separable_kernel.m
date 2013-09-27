
function A = conv3_CUDA_separable_kernel(data, f_rows, f_cols, f_slices, k)

dim = size(data);

if mod(dim(1), k.rows.kernel.ThreadBlockSize(1)*k.rows.computations_per_thread)
    error(['Number of rows must be a multiple of ',int2str(k.rows.kernel.ThreadBlockSize(1)*k.rows.computations_per_thread)]);
end
if mod(dim(2), k.cols.kernel.ThreadBlockSize(2)*k.cols.computations_per_thread)
    error(['Number of columns must be a multiple of ',int2str(k.cols.kernel.ThreadBlockSize(2)*k.cols.computations_per_thread)]);
end
if mod(dim(3), k.slices.kernel.ThreadBlockSize(3)*k.slices.computations_per_thread)
    error(['Number of slices must be a multiple of ',int2str(k.slices.kernel.ThreadBlockSize(3)*k.slices.computations_per_thread)]);
end

k.rows.kernel.GridSize = [dim(1)/(k.rows.kernel.ThreadBlockSize(1)*k.rows.computations_per_thread) ...
                          dim(2)*dim(3)/(k.rows.kernel.ThreadBlockSize(2)*k.rows.kernel.ThreadBlockSize(3))];

k.cols.kernel.GridSize = [dim(1)*dim(3)/(k.cols.kernel.ThreadBlockSize(1)*k.cols.kernel.ThreadBlockSize(3)) ...
                          dim(2)/(k.cols.kernel.ThreadBlockSize(2)*k.cols.computations_per_thread)];

k.slices.kernel.GridSize = [dim(1)*dim(2)/(k.slices.kernel.ThreadBlockSize(1)*k.slices.kernel.ThreadBlockSize(2)) ...
                            dim(3)/(k.slices.kernel.ThreadBlockSize(3)*k.slices.computations_per_thread)];

A = parallel.gpu.GPUArray.zeros(dim, 'single');
B = parallel.gpu.GPUArray.zeros(dim, 'single');

if verLessThan('distcomp', '6.0')
    dim = gpuArray(int32(dim));
    
    A = feval(k.rows.kernel, A, data, dim, f_rows);
    B = feval(k.cols.kernel, B, A, dim, f_cols);
    A = feval(k.slices.kernel, A, B, dim, f_slices);
else
    setConstantMemory(k.rows.kernel, 'dim', int32(dim));
    setConstantMemory(k.rows.kernel, 'd_Kernel', f_rows);
    setConstantMemory(k.cols.kernel, 'dim', int32(dim));
    setConstantMemory(k.cols.kernel, 'd_Kernel', f_cols);
    setConstantMemory(k.slices.kernel, 'dim', int32(dim));
    setConstantMemory(k.slices.kernel, 'd_Kernel', f_slices);

    A = feval(k.rows.kernel, A, data);
    B = feval(k.cols.kernel, B, A);
    A = feval(k.slices.kernel, A, B);
end
