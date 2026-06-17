include Makefile.include
#

SRCS := $(wildcard *.cc)
OBJS := $(patsubst %.cc,%.o,$(SRCS))
BINS := $(patsubst %.cc,%,$(SRCS))
SCRIPTS :=

# Edit the following as appropriate (retain the order!)

# For ROOT:
CPPFLAGS += $(ROOT_CPPFLAGS)   # see defn in ../Makefile.include
LDFLAGS  += $(ROOT_LDFLAGS)    # see defn in ../Makefile.include

# For FASTJET:
CPPFLAGS += $(FASTJET_CPPFLAGS)     # see defn in ../Makefile.include
LDFLAGS  += $(FASTJET_LDFLAGS)      # see defn in ../Makefile.include

CPP:= g++
F77:= gfortran
LD:= g++

default: $(BINS) $(SCRIPTS)
#	@echo 
#	@echo	
	@echo "Good! Everything was built successfully."

%: %.o
	 $(CPP) -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 -Wunused-variable -ldl -o $@ $< $(LDFLAGS)

%.o: %.cpp %.cc
	 $(CPP) -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 -c $< $(CPPFLAGS)

$(OBJS) : 

clean:
	rm -f *.o $(BINS) 
