//! Generic, vendor-agnostic engine parameters — the schema behind an "advanced settings" UI.
//!
//! An engine declares the knobs it exposes as a list of [`ParamSpec`] (how to render each control,
//! what it means, and a smart default). A UI renders controls from those specs, the user's choices
//! ride in a [`ParamValues`] map, and the engine reads them back with a default fallback. Because
//! both sides speak this one schema, a new engine — or a new parameter — adds a spec and needs no UI
//! change: the same panel renders local-model knobs, cloud-realtime VAD settings, anything.

use std::collections::BTreeMap;

/// The control type and bounds for one tunable parameter — tells a UI how to render it.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    /// A continuous value within `[min, max]`, adjusted in `step` increments (a slider).
    Float { min: f64, max: f64, step: f64 },
    /// A whole number within `[min, max]` (a stepper or slider).
    Int { min: i64, max: i64 },
    /// An on/off toggle.
    Bool,
    /// One of a fixed set of option values (a dropdown).
    Enum(Vec<String>),
    /// Free text (a text field).
    Text,
}

/// A tunable value carried by a [`ParamValues`]; its variant matches its [`ParamSpec`]'s `kind`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Text(String),
}

/// Declares one tunable parameter an engine exposes: its wire `key`, a human `label` and `help`, how
/// to render it ([`ParamKind`]), its smart `default`, and whether it's an `advanced` knob (hidden
/// behind an "advanced" disclosure by default).
#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    pub key: String,
    pub label: String,
    pub help: String,
    pub kind: ParamKind,
    pub default: ParamValue,
    pub advanced: bool,
}

impl ParamSpec {
    /// A float slider parameter (advanced by default; call [`ParamSpec::basic`] to surface it).
    pub fn float(
        key: &str,
        label: &str,
        help: &str,
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    ) -> Self {
        Self::new(
            key,
            label,
            help,
            ParamKind::Float { min, max, step },
            ParamValue::Float(default),
        )
    }

    /// An integer parameter within `[min, max]`.
    pub fn int(key: &str, label: &str, help: &str, min: i64, max: i64, default: i64) -> Self {
        Self::new(
            key,
            label,
            help,
            ParamKind::Int { min, max },
            ParamValue::Int(default),
        )
    }

    /// A dropdown over `options`, defaulting to `default` (which should be one of them).
    pub fn enumerated(key: &str, label: &str, help: &str, options: &[&str], default: &str) -> Self {
        let options = options.iter().map(|o| (*o).to_owned()).collect();
        Self::new(
            key,
            label,
            help,
            ParamKind::Enum(options),
            ParamValue::Text(default.to_owned()),
        )
    }

    /// An on/off toggle.
    pub fn boolean(key: &str, label: &str, help: &str, default: bool) -> Self {
        Self::new(key, label, help, ParamKind::Bool, ParamValue::Bool(default))
    }

    /// A free-text field.
    pub fn text(key: &str, label: &str, help: &str, default: &str) -> Self {
        Self::new(
            key,
            label,
            help,
            ParamKind::Text,
            ParamValue::Text(default.to_owned()),
        )
    }

    /// Marks this spec as a basic (always-shown) control rather than an advanced one.
    pub fn basic(mut self) -> Self {
        self.advanced = false;
        self
    }

    fn new(key: &str, label: &str, help: &str, kind: ParamKind, default: ParamValue) -> Self {
        Self {
            key: key.to_owned(),
            label: label.to_owned(),
            help: help.to_owned(),
            kind,
            default,
            advanced: true,
        }
    }
}

/// User-set (or defaulted) parameter values, keyed by [`ParamSpec::key`].
///
/// An engine reads values back with a default fallback (`float`/`int`/`bool`/`text`), so a missing or
/// wrong-typed key never panics — it just uses the default. Build one pre-filled with an engine's
/// smart defaults via [`ParamValues::from_specs`], then overlay the user's overrides.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParamValues {
    values: BTreeMap<String, ParamValue>,
}

impl ParamValues {
    /// An empty set (every lookup returns the caller's default).
    pub fn new() -> Self {
        Self::default()
    }

    /// A set pre-filled with each spec's smart default.
    pub fn from_specs(specs: &[ParamSpec]) -> Self {
        let values = specs
            .iter()
            .map(|s| (s.key.clone(), s.default.clone()))
            .collect();
        Self { values }
    }

    /// Sets `key` to `value`, replacing any previous value.
    pub fn set(&mut self, key: &str, value: ParamValue) {
        self.values.insert(key.to_owned(), value);
    }

    /// `key` as a float, or `default` if absent. An `Int` value is accepted and widened.
    pub fn float(&self, key: &str, default: f64) -> f64 {
        match self.values.get(key) {
            Some(ParamValue::Float(v)) => *v,
            Some(ParamValue::Int(v)) => *v as f64,
            _ => default,
        }
    }

    /// `key` as an integer, or `default` if absent. A `Float` value is accepted and truncated.
    pub fn int(&self, key: &str, default: i64) -> i64 {
        match self.values.get(key) {
            Some(ParamValue::Int(v)) => *v,
            Some(ParamValue::Float(v)) => *v as i64,
            _ => default,
        }
    }

    /// `key` as a bool, or `default` if absent.
    pub fn bool(&self, key: &str, default: bool) -> bool {
        match self.values.get(key) {
            Some(ParamValue::Bool(v)) => *v,
            _ => default,
        }
    }

    /// `key` as text, or `default` if absent.
    pub fn text(&self, key: &str, default: &str) -> String {
        match self.values.get(key) {
            Some(ParamValue::Text(v)) => v.clone(),
            _ => default.to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_specs_pre_fills_each_default() {
        let specs = vec![
            ParamSpec::float("threshold", "Threshold", "", 0.0, 1.0, 0.05, 0.5),
            ParamSpec::int("silence_ms", "Silence", "", 100, 2000, 500),
            ParamSpec::enumerated("mode", "Mode", "", &["off", "near"], "near"),
        ];
        let values = ParamValues::from_specs(&specs);

        assert_eq!(values.float("threshold", 0.0), 0.5);
        assert_eq!(values.int("silence_ms", 0), 500);
        assert_eq!(values.text("mode", ""), "near");
    }

    #[test]
    fn lookups_fall_back_to_the_default_when_absent_or_mistyped() {
        let mut values = ParamValues::new();
        values.set("threshold", ParamValue::Float(0.8));

        assert_eq!(values.float("threshold", 0.5), 0.8, "present value wins");
        assert_eq!(values.float("missing", 0.5), 0.5, "absent → default");
        assert_eq!(values.int("threshold", 7), 0, "float read as int truncates");
        assert!(
            values.bool("threshold", true),
            "wrong type → default (true)"
        );
        assert!(
            !values.bool("threshold", false),
            "wrong type → default (false)"
        );
    }

    #[test]
    fn constructors_default_to_advanced_unless_marked_basic() {
        assert!(ParamSpec::boolean("x", "X", "", false).advanced);
        assert!(!ParamSpec::text("y", "Y", "", "").basic().advanced);
    }
}
