NAME = CUDAPm1
VERSION = 0.20 Alpha
OptLevel = 3
OUT = $(NAME)

# The nVidia CUDA Toolkit will provide both nvcc and the CUDA libraries. If you
# follow their defaults, the necessary files will be installed in your PATH and
# LDPATH. Otherwise, you'll need to manually insert their paths here.

CFLAGS = $(OPTS) -Wall

LDFLAGS = $(CFLAGS) -fPIC $(LIBDIR) $(LIB)

$(NAME): CUDAPm1.o parse.o
	$(CC) $^ $(LDFLAGS) -o $(OUT)

CUDAPm1.o: CUDAPm1.cu parse.h cuda_safecalls.h
	$(NVCC) $(NVCC_OPTS) $(NVOPTS) -c $<

parse.o: parse.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o
	rm -f $(NAME) debug_$(NAME) test_$(NAME)

debug: CFLAGS += -DEBUG -g
debug: NVOPTS += -DEBUG -g
debug: OptLevel = 0
debug: OUT = debug_$(NAME)
debug: $(NAME)

test: CFLAGS += -DTEST
test: NVOPTS += -DTEST
test: OUT = test_$(NAME)
test: $(NAME)

help:
	@echo "\n\"make\"           builds CUDAPm1"
	@echo "\"make clean\"     removes object files"
	@echo "\"make cleaner\"   removes executables"
	@echo "\"make cleanest\"  does both clean and cleaner"
	@echo "\"make debug\"     creates a debug build"
	@echo "\"make test\"      creates an experimental build"
	@echo "\"make help\"      prints this message\n"
