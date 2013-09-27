
%% Step to the parent directory
S = pwd;
cd ..

%% Create data (box shape)
dim = [256 256 256];
data = zeros(dim, 'single');
data(50:100, 100:150, 150:200) = 1;

%% Create filter kernel (box filter)
kernel_radius = 15;
H = ones(kernel_radius*2 + 1, 1);
H = H/sum(H);

%% Filter on CPU
tic;
result_CPU = imfilter(data, H, 'replicate', 'conv');
result_CPU = imfilter(result_CPU, H', 'replicate', 'conv');
result_CPU = imfilter(result_CPU, reshape(H, [1 1 length(H)]), 'replicate', 'conv');
T_CPU = toc;
disp(['CPU time: ',num2str(T_CPU),' s']);

%% Create CUDA kernel
k = conv3_CUDA_separable_kernel_make(kernel_radius);

%% Filter on GPU
% Upload data to GPU
data = gpuArray(data);
H_GPU = gpuArray(single(H));

tic;
result_GPU = conv3_CUDA_separable_kernel(data, H_GPU, H_GPU, H_GPU, k);
result_GPU = gather(result_GPU);
T_GPU = toc;
disp(['GPU time: ',num2str(T_GPU),' s']);

%% Compute difference (relative L2 norm)
disp(['Speedup: ',num2str(T_CPU/T_GPU)]);
A = (result_GPU - result_CPU).^2;
B = result_CPU.^2;
sqrt(sum(A(:))/sum(B(:)))

%% Step back...
cd(S);
