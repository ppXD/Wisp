<script lang="ts">
  // A generic advanced-settings panel: given an engine's ParamSpecs and a bindable values map, it
  // renders a control per spec (slider / toggle / dropdown / text) with help, and a reset. It knows
  // nothing about any specific engine — a new knob is backend-only, rendered here automatically.
  import type { ParamSpec, ParamValue } from "$lib/cloud.svelte";
  import { defaultParamValues } from "$lib/cloud.svelte";

  let {
    specs,
    values = $bindable(),
  }: {
    specs: ParamSpec[];
    values: Record<string, ParamValue>;
  } = $props();

  // Basic controls first, then advanced ones.
  const ordered = $derived([...specs].sort((a, b) => Number(a.advanced) - Number(b.advanced)));
  const isDefault = $derived(specs.every((s) => values[s.key] === s.default));

  function set(key: string, v: ParamValue) {
    values = { ...values, [key]: v };
  }
  function setNum(key: string, raw: string) {
    const n = Number(raw);
    if (!Number.isNaN(n)) set(key, n);
  }
  function reset() {
    values = defaultParamValues(specs);
  }
</script>

{#if specs.length}
  <div class="params">
    <div class="params-head">
      <span class="params-title">Parameters</span>
      <button class="reset" disabled={isDefault} onclick={reset}>Reset to defaults</button>
    </div>

    {#each ordered as s (s.key)}
      <div class="row">
        <div class="row-head">
          <span class="label">{s.label}</span>
          {#if s.kind === "float" || s.kind === "int"}
            <span class="num">{values[s.key]}</span>
          {/if}
        </div>

        {#if s.kind === "float" || s.kind === "int"}
          <input
            class="range"
            type="range"
            min={s.min}
            max={s.max}
            step={s.kind === "int" ? 1 : s.step}
            value={Number(values[s.key])}
            oninput={(e) => setNum(s.key, e.currentTarget.value)}
          />
        {:else if s.kind === "bool"}
          <button
            class="toggle"
            class:on={values[s.key] === true}
            role="switch"
            aria-checked={values[s.key] === true}
            aria-label={s.label}
            onclick={() => set(s.key, !values[s.key])}><span class="knob"></span></button
          >
        {:else if s.kind === "enum"}
          <select
            class="select"
            value={values[s.key]}
            onchange={(e) => set(s.key, e.currentTarget.value)}
          >
            {#each s.options as opt (opt)}<option value={opt}>{opt}</option>{/each}
          </select>
        {:else}
          <input
            class="text"
            type="text"
            value={values[s.key]}
            oninput={(e) => set(s.key, e.currentTarget.value)}
          />
        {/if}

        {#if s.help}<span class="help">{s.help}</span>{/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .params {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .params-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .params-title {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .reset {
    font-family: inherit;
    font-size: 12px;
    color: var(--accent);
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 0;
  }

  .reset:disabled {
    color: var(--muted);
    opacity: 0.6;
    cursor: default;
  }

  .row {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }

  .row-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }

  .label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
  }

  .num {
    font-family: var(--mono, monospace);
    font-size: 12.5px;
    color: var(--muted);
  }

  .range {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .select,
  .text {
    width: 100%;
    box-sizing: border-box;
    font-family: inherit;
    font-size: 13px;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 6px 9px;
  }

  .select:focus,
  .text:focus {
    outline: none;
    border-color: var(--accent);
  }

  .toggle {
    align-self: flex-start;
    width: 38px;
    height: 22px;
    border-radius: 999px;
    border: none;
    background: var(--border-strong);
    cursor: pointer;
    padding: 0;
    transition: background 0.15s;
  }

  .toggle.on {
    background: var(--accent);
  }

  .knob {
    display: block;
    width: 18px;
    height: 18px;
    margin: 2px;
    border-radius: 50%;
    background: white;
    transition: transform 0.15s;
  }

  .toggle.on .knob {
    transform: translateX(16px);
  }

  .help {
    font-size: 11.5px;
    line-height: 1.4;
    color: var(--muted);
  }
</style>
