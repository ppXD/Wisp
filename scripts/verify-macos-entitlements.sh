#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/Wisp.app-or.dmg" >&2
    exit 2
fi

artifact_path="$1"
mount_point=""

cleanup() {
    if [[ -n "$mount_point" ]]; then
        hdiutil detach "$mount_point" >/dev/null
        rmdir "$mount_point"
    fi
}

trap cleanup EXIT

if [[ -d "$artifact_path" ]]; then
    app_path="$artifact_path"
elif [[ -f "$artifact_path" && "$artifact_path" == *.dmg ]]; then
    mount_point="$(mktemp -d "${TMPDIR:-/tmp}/wisp-dmg.XXXXXX")"
    hdiutil attach -readonly -nobrowse -noautoopen -mountpoint "$mount_point" "$artifact_path" >/dev/null
    app_path="$mount_point/Wisp.app"
else
    echo "App bundle or DMG not found: $artifact_path" >&2
    exit 2
fi

if [[ ! -d "$app_path" ]]; then
    echo "Wisp.app not found in artifact: $artifact_path" >&2
    exit 2
fi

entitlements="$(codesign -d --entitlements :- "$app_path" 2>/dev/null)"

if ! microphone_access="$(printf '%s' "$entitlements" | plutil -extract 'com\.apple\.security\.device\.audio-input' raw - 2>/dev/null)" || [[ "$microphone_access" != "true" ]]; then
    echo "Missing com.apple.security.device.audio-input entitlement: $app_path" >&2
    exit 1
fi

if ! library_validation="$(printf '%s' "$entitlements" | plutil -extract 'com\.apple\.security\.cs\.disable-library-validation' raw - 2>/dev/null)" || [[ "$library_validation" != "true" ]]; then
    echo "Missing com.apple.security.cs.disable-library-validation entitlement: $app_path" >&2
    exit 1
fi

echo "Verified macOS audio-input and library-validation entitlements: $artifact_path"
