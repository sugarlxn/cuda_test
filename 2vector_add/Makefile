# CUDA 编译器
NVCC = /usr/local/cuda-12.4/bin/nvcc

# 编译器标志
# -O2 是优化级别
# -arch=native 会让 nvcc 检测并使用你当前 GPU 的最佳架构
# 如果编译失败，可以尝试替换为具体的架构，例如：-gencode=arch=compute_75,code=sm_75 (适用于 Turing 架构)
NVCCFLAGS = -O2 -arch=native

# 链接器标志 (如果需要链接其他库，在这里添加，例如 -L/path/to/lib -lmylib)
LDFLAGS =

# 目标可执行文件名
# 放置可执行文件的目录
BINDIR = build

# 目标可执行文件路径
TARGET = $(BINDIR)/main

# 源文件
SRCS = main.cu

# 头文件
INCLUDES = -I../include

# 默认目标
all: $(TARGET)

# 编译和链接规则（在链接前确保 build 目录存在）
$(TARGET): $(SRCS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(INCLUDES)

# 创建输出目录
$(BINDIR):
	mkdir -p $@

# 清理规则，用于删除生成的文件
clean:
	rm -rf $(BINDIR)

# 将 all 和 clean 声明为伪目标
.PHONY: all clean
