F_FILES := $(wildcard src/*.f)
OBJ_FILES := $(addprefix obj/,$(notdir $(F_FILES:.f=.o)))
F_FLAGS := -O3
FORTRAN := gfortran
ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib
BLASLIB   = bin/blas.a

all: $(BLASLIB)

$(BLASLIB): $(OBJ_FILES)
	mkdir bin
	$(ARCH) $(ARCHFLAGS) $@ $(OBJ_FILES)
	$(RANLIB) $@

clean:
	rm -r bin
	rm -r obj
	
obj/%.o: src/%.f
	$(FORTRAN) $(F_FLAGS) -c $< -o $@