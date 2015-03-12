NAME = CUDALucas
VERSION = 2.05.1
OptLevel = 1

OUT = $(NAME)

# In CUFLAGSS, make an entry: --generate-code arch=compute_XY,code=sm_XY
# for any compute capability you want to support. Possibilities are
# For cuda 4.2, XY = 13, 20, or 30
# For cuda 5.0, XY = 13, 20, 30, or 35
# For cuda 5.5, XY = 13, 20, 30, or 35
# For cuda 6.0, XY = 13, 20, 30, 35, or 50
# For cuda 6.5, XY = 13, 20, 30, 35, or 50
# For cuda 7.0, XY = 20, 30, 35, 50, or 52

$(NAME): CUDALucas.o parse.o
	$(CC) $^ $(LDOPTS) -o $(OUT)

CUDALucas.o: CUDALucas.cu parse.h cuda_safecalls.h
	$(NVCC) $(NVOPTS) -c $<

parse.o: parse.c
	$(CC) $(OPTS) -c $<

clean:
	rm -f *.o
	rm -f $(NAME) debug_$(NAME) test_$(NAME)

debug: OPTS += -DEBUG -g
debug: NVOPTS += -DEBUG -g
debug: OptLevel = 0
debug: OUT = debug_$(NAME)
debug: $(NAME)

test: OPTS += -DTEST
test: NVOPTS += -DTEST
test: OUT = test_$(NAME)
test: $(NAME)

help:
	@echo "\n\"make\"           builds CUDALucas"
	@echo "\"make clean\"     removes object files"
	@echo "\"make cleaner\"   removes executables"
	@echo "\"make cleanest\"  does both clean and cleaner"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make test\"      creates an experimental build"
	@echo "\"make help\"      prints this message\n"
