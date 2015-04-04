# Default pod makefile distributed with pods version: 12.11.14

default_target: all

# get a list of subdirs to build by reading tobuild.txt
SUBDIRS:=$(shell grep -v "^\#" tobuild.txt)

# Default to a less-verbose build.  If you want all the gory compiler output,
# run "make VERBOSE=1"
$(VERBOSE).SILENT:

# Figure out where to build the software.
#   Use BUILD_PREFIX if it was passed in.
#   If not, search up to four parent directories for a 'build' directory.
#   Otherwise, use ./build.
ifeq "$(BUILD_PREFIX)" ""
BUILD_PREFIX:=$(shell for pfx in ./ .. ../.. ../../.. ../../../..; do d=`pwd`/$$pfx/build;\
               if [ -d $$d ]; then echo $$d; exit 0; fi; done; echo `pwd`/build)
endif
# create the build directory if needed, and normalize its path name
BUILD_PREFIX:=$(shell mkdir -p $(BUILD_PREFIX) && cd $(BUILD_PREFIX) && echo `pwd`)

# Default to a release build.  If you want to enable debugging flags, run
# "make BUILD_TYPE=Debug"
ifeq "$(BUILD_TYPE)" ""
BUILD_TYPE="Release"
endif

all:
	@[ -d $(BUILD_PREFIX) ] || mkdir -p $(BUILD_PREFIX) || exit 1
	@for subdir in $(SUBDIRS); do \
    echo "\n-------------------------------------------"; \
    echo "-- $$subdir"; \
    echo "-------------------------------------------"; \
    $(MAKE) -C $$subdir all || exit 2; \
  done 
	@$(MAKE) -C pod-build all install
	@# Place additional commands here if you have any

clean:
	@for subdir in $(SUBDIRS); do \
    echo "\n-------------------------------------------"; \
    echo "-- $$subdir"; \
    echo "-------------------------------------------"; \
    $(MAKE) -C $$subdir clean; \
  done
	@# Place additional commands here if you have any

pod-build/Makefile:
	$(MAKE) configure

.PHONY: configure
configure:
	@echo "\nBUILD_PREFIX: $(BUILD_PREFIX)\n\n"

	# create the temporary build directory if needed
	@mkdir -p pod-build

	# run CMake to generate and configure the build scripts
	@cd pod-build && cmake -DCMAKE_INSTALL_PREFIX=$(BUILD_PREFIX) \
		   -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) ..


checkout:
	git clone /data/vision/fisher/code/jstraub/gitrep/research/jsCore.git/

update:
	cd jsCore; git pull; cd -

# other (custom) targets are passed through to the cmake-generated Makefile
%::
	$(MAKE) -C pod-build $@
