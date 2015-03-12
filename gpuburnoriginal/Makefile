CUDAPATH=/opt/nvidia/cudatoolkit6.5/default

# Have this point to an old enough gcc (for nvcc)
GCCPATH=/usr

NVCC=${CUDAPATH}/bin/nvcc
CCPATH=${GCCPATH}/bin

drv:
	PATH=.:${CCPATH}:${PATH} ${NVCC} -arch=compute_13 -ptx compare.cu -o compare.ptx
	g++ -O3 -I${CUDAPATH}/include -c gpu_burn-drv.cpp
	g++ -o gpu_burn gpu_burn-drv.o -O3 -lcuda -L${CUDAPATH}/lib64 -L/opt/cray/nvidia/default/lib64 -Wl,-rpath=${CUDAPATH}/lib64 -Wl,-rpath=${CUDAPATH}/lib -lcublas -lcudart -o gpu_burn
