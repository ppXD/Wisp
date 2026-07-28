import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = resolve(
  root,
  "app/src-tauri/msvc-runtime-required.txt",
);
const configPath = resolve(root, "app/src-tauri/tauri.windows.conf.json");

const runtimeNames = readFileSync(manifestPath, "utf8")
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith("#"))
  .map((line) => line.split(":", 1)[0]);
const duplicate = runtimeNames.find(
  (name, index) => runtimeNames.indexOf(name) !== index,
);
if (duplicate) {
  throw new Error(`duplicate MSVC runtime manifest entry: ${duplicate}`);
}

const config = JSON.parse(readFileSync(configPath, "utf8"));
const resources = config.bundle?.resources ?? {};
const explicitRuntimeNames = Object.entries(resources)
  .filter(
    ([source, target]) =>
      source.startsWith("windows-runtime/") &&
      /^(concrt|msvcp|vcamp|vccorlib|vcomp|vcruntime).*\.dll$/i.test(target),
  )
  .map(([, target]) => target)
  .sort();

const runtimeGlob = "windows-runtime/msvc/*.dll";
if (resources[runtimeGlob] !== "") {
  throw new Error(`missing Windows bundle glob mapping: "${runtimeGlob}": ""`);
}

if (explicitRuntimeNames.length > 0) {
  throw new Error(
    `MSVC runtime config must use the toolset-neutral glob, not explicit files: ${explicitRuntimeNames.join(", ")}`,
  );
}

console.log(
  `Verified wildcard MSVC runtime bundling and ${runtimeNames.length} required ABI files`,
);
