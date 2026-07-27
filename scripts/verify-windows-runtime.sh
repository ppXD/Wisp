#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <runtime-directory|Wisp_*_x64-setup.exe> [--require-vulkan]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
input=$1
require_vulkan=false
if [[ $# -eq 2 ]]; then
  [[ $2 == "--require-vulkan" ]] || usage
  require_vulkan=true
fi
extract_dir=

cleanup() {
  if [[ -n "$extract_dir" && -d "$extract_dir" ]]; then
    rm -rf "$extract_dir"
  fi
}
trap cleanup EXIT

if [[ -f "$input" ]]; then
  command -v 7z >/dev/null 2>&1 || {
    echo "7z is required to inspect the NSIS installer" >&2
    exit 1
  }
  extract_dir=$(mktemp -d "${TMPDIR:-/tmp}/wisp-windows-package.XXXXXX")
  7z x -y "-o$extract_dir" "$input" >/dev/null
  runtime_dir=$extract_dir
elif [[ -d "$input" ]]; then
  runtime_dir=$input
else
  echo "Runtime directory or installer not found: $input" >&2
  exit 1
fi

files=(
  "onnxruntime.dll:1000000"
  "onnxruntime_providers_shared.dll:1000"
  "sherpa-onnx-c-api.dll:1000000"
)

if [[ $require_vulkan == true ]]; then
  files+=(
    "vulkan-1.dll:100000"
    "VulkanRT-License.txt:1000"
  )
fi

if [[ -n "$extract_dir" ]]; then
  files+=(
    "Wisp.exe:1000000"
    "resources/silero_vad.onnx:100000"
  )
fi

for expected in "${files[@]}"; do
  name=${expected%%:*}
  minimum=${expected##*:}
  file="$runtime_dir/$name"
  if [[ ! -f "$file" ]]; then
    echo "Missing Windows package file: $name" >&2
    exit 1
  fi
  size=$(wc -c < "$file" | tr -d '[:space:]')
  if (( size < minimum )); then
    echo "Windows package file is truncated: $name is $size bytes (minimum $minimum)" >&2
    exit 1
  fi
  echo "Verified $name ($size bytes)"
done
