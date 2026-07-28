import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = resolve(root, "app/src-tauri/msvc-runtime-dlls.txt");
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
const configuredRuntimeNames = Object.entries(resources)
  .filter(
    ([source, target]) =>
      source.startsWith("windows-runtime/") &&
      /^(concrt|msvcp|vcamp|vccorlib|vcomp|vcruntime).*\.dll$/i.test(target),
  )
  .map(([, target]) => target)
  .sort();

for (const name of runtimeNames) {
  const source = `windows-runtime/${name}`;
  if (resources[source] !== name) {
    throw new Error(`missing Windows bundle mapping: "${source}": "${name}"`);
  }
}

const expected = [...runtimeNames].sort();
if (JSON.stringify(configuredRuntimeNames) !== JSON.stringify(expected)) {
  throw new Error(
    `MSVC runtime config drift: expected ${expected.join(", ")}, got ${configuredRuntimeNames.join(", ")}`,
  );
}

console.log(`Verified ${runtimeNames.length} MSVC runtime bundle mappings`);
