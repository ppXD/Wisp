//! Seed catalog of cloud transcription providers offered in the picker.
//!
//! Cloud is opt-in: each provider needs the user's own API key. Providers that share the OpenAI API
//! shape (here OpenAI and Groq) differ only by `base_url` and their model list, so one adapter
//! serves both — the point of the vendor-agnostic [`CloudProvider`] design.

use wisp_core::cloud::{CloudAuth, CloudModel, CloudProtocol, CloudProvider, StreamingProtocol};

/// Built-in cloud providers. Each is opt-in and requires the user's API key.
pub fn cloud_catalog() -> Vec<CloudProvider> {
    vec![openai(), speechmatics(), groq(), google(), qwen()]
}

/// A multilingual cloud model (empty `languages` = auto-detect).
fn model(id: &str, name: &str, streaming: bool, batch: bool, description: &str) -> CloudModel {
    CloudModel {
        id: id.to_owned(),
        display_name: name.to_owned(),
        streaming,
        batch,
        languages: vec![],
        description: description.to_owned(),
        recommended: false,
        diarizes: false,
    }
}

fn openai() -> CloudProvider {
    CloudProvider {
        id: "openai".to_owned(),
        display_name: "OpenAI".to_owned(),
        protocol: CloudProtocol::OpenAi,
        base_url: "https://api.openai.com/v1".to_owned(),
        keys_url: "https://platform.openai.com/api-keys".to_owned(),
        auth: CloudAuth::bearer(),
        streaming: Some(StreamingProtocol::OpenAiRealtime),
        models: vec![
            model(
                "gpt-4o-transcribe",
                "GPT-4o Transcribe",
                true,
                true,
                "Highest-accuracy multilingual transcription; live streaming and files.",
            )
            .recommended(),
            model(
                "gpt-4o-mini-transcribe",
                "GPT-4o mini Transcribe",
                true,
                true,
                "Faster and cheaper than GPT-4o Transcribe, still multilingual; streaming and files.",
            ),
            model(
                "gpt-4o-transcribe-diarize",
                "GPT-4o Transcribe (diarized)",
                false,
                true,
                "GPT-4o transcription with speaker labels — File only (realtime can't diarise).",
            )
            .diarizes(),
            model(
                "whisper-1",
                "Whisper",
                false,
                true,
                "OpenAI's hosted Whisper — files only, broad language coverage.",
            ),
            model(
                "gpt-realtime",
                "GPT Realtime",
                true,
                false,
                "Realtime model with text output — steer it with Instructions to transcribe or \
                 translate. More flexible (and pricier) than the dedicated transcribers.",
            ),
            model(
                "gpt-realtime-mini",
                "GPT Realtime mini",
                true,
                false,
                "Cost-efficient realtime model — transcribe or translate via Instructions.",
            ),
            model(
                "gpt-realtime-2",
                "GPT Realtime 2",
                true,
                false,
                "Reasoning realtime model — transcribe or translate via Instructions.",
            ),
            model(
                "gpt-realtime-translate",
                "GPT Realtime Translate",
                true,
                false,
                "Dedicated live speech translation — pick a target language, get streaming \
                 translated text.",
            ),
        ],
    }
}

fn speechmatics() -> CloudProvider {
    CloudProvider {
        id: "speechmatics".to_owned(),
        display_name: "Speechmatics".to_owned(),
        // Realtime-only here; the batch protocol field is an unused placeholder (no File models).
        protocol: CloudProtocol::OpenAi,
        base_url: "wss://eu.rt.speechmatics.com/v2".to_owned(),
        keys_url: "https://portal.speechmatics.com/manage-access/".to_owned(),
        auth: CloudAuth::bearer(),
        streaming: Some(StreamingProtocol::Speechmatics),
        // The "model" id is the Speechmatics operating point, sent in `StartRecognition`.
        models: vec![
            model(
                "enhanced",
                "Speechmatics (enhanced)",
                true,
                false,
                "Low-latency multilingual realtime — strong Cantonese/Chinese. Most accurate operating \
                 point.",
            )
            .recommended(),
            model(
                "standard",
                "Speechmatics (standard)",
                true,
                false,
                "Faster, lighter realtime operating point.",
            ),
        ],
    }
}

fn groq() -> CloudProvider {
    CloudProvider {
        id: "groq".to_owned(),
        display_name: "Groq".to_owned(),
        protocol: CloudProtocol::OpenAi, // OpenAI-compatible endpoint, different base_url
        base_url: "https://api.groq.com/openai/v1".to_owned(),
        keys_url: "https://console.groq.com/keys".to_owned(),
        auth: CloudAuth::bearer(),
        streaming: None, // Groq Whisper is batch-only (no realtime WS)
        models: vec![
            model(
                "whisper-large-v3",
                "Whisper large-v3",
                false,
                true,
                "Whisper large-v3 on Groq's fast inference — files, multilingual.",
            ),
            model(
                "whisper-large-v3-turbo",
                "Whisper large-v3 turbo",
                false,
                true,
                "Faster large-v3 turbo on Groq — files, multilingual.",
            ),
            model(
                "distil-whisper-large-v3-en",
                "Distil-Whisper large-v3 (English)",
                false,
                true,
                "Distilled large-v3 — English-only, ~6× faster on Groq. Files.",
            ),
        ],
    }
}

fn google() -> CloudProvider {
    CloudProvider {
        id: "google".to_owned(),
        display_name: "Google (Gemini)".to_owned(),
        protocol: CloudProtocol::Gemini,
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_owned(),
        keys_url: "https://aistudio.google.com/apikey".to_owned(),
        // Gemini authenticates with a `?key=` query parameter, so this header config is unused; it's
        // here only because `CloudAuth` is required. (Bearer is the harmless default.)
        auth: CloudAuth::bearer(),
        streaming: Some(StreamingProtocol::GeminiLive),
        models: vec![
            // ── Live (realtime BidiGenerateContent) models ──
            model(
                "gemini-3.1-flash-live-preview",
                "Gemini 3.1 Flash Live",
                true,
                false,
                "Newest Gemini Live — lowest latency, strong multilingual realtime transcription.",
            )
            .recommended(),
            model(
                "gemini-2.5-flash-native-audio-preview-12-2025",
                "Gemini 2.5 Flash (native audio)",
                true,
                false,
                "Native-audio Live model — high-quality realtime transcription.",
            ),
            // ── File (batch) models ──
            model(
                "gemini-2.5-flash",
                "Gemini 2.5 Flash",
                false,
                true,
                "Fast multimodal model — transcribes audio with strong multilingual (incl. CJK) \
                 accuracy.",
            ),
            model(
                "gemini-2.5-pro",
                "Gemini 2.5 Pro",
                false,
                true,
                "Most accurate Gemini — best for hard audio or nuanced multilingual transcription.",
            ),
            model(
                "gemini-2.5-flash-lite",
                "Gemini 2.5 Flash-Lite",
                false,
                true,
                "Cheapest, fastest 2.5 — quick everyday transcription.",
            ),
            model(
                "gemini-2.0-flash",
                "Gemini 2.0 Flash",
                false,
                true,
                "Faster, lower-cost multimodal model for everyday transcription.",
            ),
            model(
                "gemini-1.5-flash",
                "Gemini 1.5 Flash",
                false,
                true,
                "Earlier fast model, broadly available.",
            ),
        ],
    }
}

fn qwen() -> CloudProvider {
    CloudProvider {
        id: "qwen".to_owned(),
        display_name: "Qwen (DashScope)".to_owned(),
        // OpenAI-compatible chat endpoint carrying the audio as an `input_audio` content part.
        protocol: CloudProtocol::OpenAiChatAudio,
        // International endpoint; China-region keys use `https://dashscope.aliyuncs.com/...`.
        base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".to_owned(),
        keys_url: "https://modelstudio.console.alibabacloud.com/".to_owned(),
        auth: CloudAuth::bearer(),
        streaming: None, // DashScope realtime adapter not wired yet — file-only for now
        models: vec![model(
            "qwen3-asr-flash",
            "Qwen3-ASR Flash",
            false,
            true,
            "Alibaba's fast multilingual speech recognition, strong on Chinese and Cantonese.",
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn catalog_is_non_empty_with_unique_provider_ids() {
        let catalog = cloud_catalog();
        assert!(!catalog.is_empty());
        let ids: HashSet<_> = catalog.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids.len(), catalog.len(), "provider ids are unique");
    }

    #[test]
    fn every_provider_is_well_formed() {
        for provider in cloud_catalog() {
            assert!(
                !provider.base_url.is_empty(),
                "{} has a base_url",
                provider.id
            );
            assert!(
                !provider.auth.header.is_empty(),
                "{} has an auth header",
                provider.id
            );
            assert!(!provider.models.is_empty(), "{} offers models", provider.id);

            let ids: HashSet<_> = provider.models.iter().map(|m| m.id.as_str()).collect();
            assert_eq!(
                ids.len(),
                provider.models.len(),
                "{} model ids unique",
                provider.id
            );

            for m in &provider.models {
                assert!(
                    m.streaming || m.batch,
                    "{}/{} does something",
                    provider.id,
                    m.id
                );
            }
        }
    }

    #[test]
    fn openai_offers_a_streaming_model_and_a_files_only_model() {
        let openai = cloud_catalog()
            .into_iter()
            .find(|p| p.id == "openai")
            .expect("openai is seeded");
        assert!(openai.model("gpt-4o-transcribe").unwrap().streaming);
        let whisper = openai.model("whisper-1").unwrap();
        assert!(
            !whisper.streaming && whisper.batch,
            "whisper-1 is files-only"
        );
    }

    #[test]
    fn openais_recommended_streaming_model_is_the_dedicated_transcriber() {
        // The Live picker pre-selects the recommended streaming model. It must be the gap-free
        // dedicated transcriber (gpt-4o-transcribe) — never a conversational model session, which can
        // drop utterances under rapid speech. Exactly one streaming model is recommended.
        let openai = cloud_catalog()
            .into_iter()
            .find(|p| p.id == "openai")
            .expect("openai is seeded");

        let recommended: Vec<_> = openai
            .models
            .iter()
            .filter(|m| m.streaming && m.recommended)
            .collect();

        assert_eq!(
            recommended.len(),
            1,
            "exactly one recommended streaming model"
        );
        assert_eq!(recommended[0].id, "gpt-4o-transcribe");
    }
}
