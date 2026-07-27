#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <destination-directory>" >&2
  exit 2
fi

# Official LunarG runtime components matching the Vulkan SDK used by the Windows workflows.
# The component archive is intended for redistribution and includes the runtime license.
version=1.4.309.0
archive_name="VulkanRT-${version}-Components.zip"
archive_sha256=7d969f4d7b44e387667d3148f61559497c22d50cbe3d50adc9e5409afbce2df1
archive_url="https://sdk.lunarg.com/sdk/download/${version}/windows/${archive_name}"
destination=$1
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/wisp-vulkan-runtime.XXXXXX")
archive="$work_dir/$archive_name"

cleanup() {
  if [[ -d "$work_dir" ]]; then
    rm -rf "$work_dir"
  fi
}
trap cleanup EXIT

command -v curl >/dev/null 2>&1 || {
  echo "curl is required to download the Vulkan runtime" >&2
  exit 1
}
command -v 7z >/dev/null 2>&1 || {
  echo "7z is required to extract the Vulkan runtime" >&2
  exit 1
}

curl --fail --location --retry 3 --output "$archive" "$archive_url"
if command -v shasum >/dev/null 2>&1; then
  actual_sha256=$(shasum -a 256 "$archive" | cut -d ' ' -f 1)
else
  actual_sha256=$(sha256sum "$archive" | cut -d ' ' -f 1)
fi
[[ $actual_sha256 == "$archive_sha256" ]] || {
  echo "Vulkan runtime checksum mismatch: expected $archive_sha256, got $actual_sha256" >&2
  exit 1
}

mkdir -p "$destination"
7z e -y "-o$destination" "$archive" "*/x64/vulkan-1.dll" "*/VulkanRT-License.txt" >/dev/null

for file in vulkan-1.dll VulkanRT-License.txt; do
  [[ -s "$destination/$file" ]] || {
    echo "Failed to stage $file from the Vulkan runtime archive" >&2
    exit 1
  }
done

echo "Staged LunarG Vulkan runtime $version in $destination"
