# BUILD_DIR=/home/pan251/accel-sim-framework/gpu-simulator/build/release


# SRCS = $(shell ls *.cc)

# graphics:$(OBJS)
DEBUG?=0


include /home/pan251/accel-sim-framework/gpu-simulator/gpgpu-sim/src/graphics/Makefile.include
CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include
ifneq ($(DEBUG),1)
	BUILD_DIR = /home/pan251/accel-sim-framework/gpu-simulator/gpgpu-sim/build/gcc-7.5.0/cuda-11000/release/graphics
else
	BUILD_DIR = /home/pan251/accel-sim-framework/gpu-simulator/gpgpu-sim/build/gcc-7.5.0/cuda-11000/debug/graphics
	CXXFLAGS += -DDEBUG
endif

CUDA_VERSION_STRING:=$(shell $(CUDA_INSTALL_PATH)/bin/nvcc --version | awk '/release/ {print $$5;}' | sed 's/,//')
CUDART_VERSION:=$(shell echo $(CUDA_VERSION_STRING) | sed 's/\./ /' | awk '{printf("%02u%02u", 10*int($$1), 10*$$2);}')
CXXFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

OPTFLAGS += -g3 -fPIC

CPP = g++ $(SNOW)
OEXT = o

# $(BUILD_DIR)/graphics.Makefile.makedepend: depend

# SRCS = $(shell ls *.cc)
SRCS = gpgpusim_to_graphics_calls.cc mesa_calls.cc mesa_gpgpusim.cc
OBJS = $(SRCS:%.cc=$(BUILD_DIR)/%.$(OEXT))

$(BUILD_DIR)/libgraphics.a:	$(OBJS)
	ar rcs  $(BUILD_DIR)/libgpgpusim.a $(OBJS) $(BUILD_DIR)/*.o

$(BUILD_DIR)/Makefile.makedepend: depend

depend:
	touch $(BUILD_DIR)/graphics.Makefile.makedepend
	makedepend -f$(BUILD_DIR)/graphics.Makefile.makedepend -p$(BUILD_DIR)/ $(SRCS) 2> /dev/null

$(BUILD_DIR)/%.$(OEXT): %.cc
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) -o $(BUILD_DIR)/$*.$(OEXT) -c $*.cc

# include $(BUILD_DIR)/graphics.Makefile.makedepend