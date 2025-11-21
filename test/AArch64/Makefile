# AArch64/Makefile - build and run test_isa_extension_support.cpp

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
SRCS := test_isa_extension_support.cpp
TARGET := test_isa_extension_support
OBJ := $(SRCS:.cpp=.o)

.PHONY: all run clean

# default: build and run
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(TARGET)
	@echo "Running $(TARGET)..."
	./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)