F_FILES := $(wildcard src/*.f)
OBJ_FILES := $(addprefix obj/,$(notdir $(F_FILES:.f=.o)))
F_FLAGS := -O3
FORTRAN := gfortran
ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib
BLASLIB   = bin/libblas.a

all: $(BLASLIB)

$(BLASLIB): dirs $(OBJ_FILES)
	$(ARCH) $(ARCHFLAGS) $@ $(OBJ_FILES)
	$(RANLIB) $@
	
dirs:
	mkdir -p bin
	mkdir -p obj

clean:
	rm -r bin
	rm -r obj
	
obj/%.o: src/%.f
	$(FORTRAN) $(F_FLAGS) -c $< -o $@