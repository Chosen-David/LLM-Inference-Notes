#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yswang/tvm_learn/cuda/LLM-Inference-Notes"
cd "$ROOT"

[ -d .git ] || git init
mkdir -p third-party

add_sm () {
  local url="$1"
  local path="$2"
  local branch="${3:-}"     # 可选分支
  local depth="${4:-1}"     # 默认浅克隆 depth=1

  # 如果已经是子模块，跳过
  if git config -f .gitmodules --get-regexp "^submodule\.$path\.url" >/dev/null 2>&1; then
    echo "[SKIP] submodule already registered: $path"
    return 0
  fi

  # 如果同名目录已存在且不是子模块，先删掉，避免 git submodule add 报错
  if [ -d "$path" ] && [ ! -f "$path/.git" ]; then
    echo "[CLEAN] remove existing non-submodule dir: $path"
    rm -rf "$path"
  fi

  # 构造 add 参数
  args=(submodule add --depth "${depth}")
  [ -n "$branch" ] && args+=(-b "$branch")
  args+=("$url" "$path")

  echo "[ADD] git ${args[*]}"
  git "${args[@]}"

  # 可选：立刻 checkout 指定分支（保持一致）
  if [ -n "$branch" ]; then
    (cd "$path" && git checkout "$branch")
  fi
}

# 1) PyTorch（含 torch.compile 源码）
add_sm git@github.com:pytorch/pytorch.git third-party/pytorch main 1

# 2) vLLM
add_sm git@github.com:vllm-project/vllm.git third-party/vllm main 1

# 3) SGLang
add_sm git@github.com:sgl-project/sglang.git third-party/sglang main 1

# 4) TensorRT-LLM
add_sm git@github.com:NVIDIA/TensorRT-LLM.git third-party/tensorrt-llm main 1

git add .gitmodules third-party || true
git commit -m "Add submodules: pytorch, vllm, sglang, tensorrt-llm under third-party" || true

# 初始化/更新（递归）以确保可用
git submodule update --init --recursive
