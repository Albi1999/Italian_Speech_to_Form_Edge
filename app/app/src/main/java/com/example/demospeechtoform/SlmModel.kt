package com.example.demospeechtoform

import com.google.mediapipe.tasks.genai.llminference.LlmInference.Backend

enum class SlmModel(
    val path: String,
    val url: String,
    val preferredBackend: Backend?,
    val temperature: Float,
    val topK: Int,
    val topP: Float,
    val maxTokens: Int = 1024
) {
    GEMMA_3_1B_IT_GPU(
        path = "/data/local/tmp/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task",
        url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task",
        preferredBackend = Backend.GPU,
        temperature = 0.2f,
        topK = 40,
        topP = 1.0f,
        maxTokens = 4000
    ),
    GEMMA_3_1B_IT_CPU(
        path = "/data/local/tmp/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task",
        url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task",
        preferredBackend = Backend.CPU,
        temperature = 0.1f,
        topK = 10,
        topP = 1.0f,
        maxTokens = 4000
    ),
    PHI_4_MINI_INSTRUCT(
        path = "/data/local/tmp/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv1280.task",
        url = "https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv1280.task",
        preferredBackend = Backend.CPU,
        temperature = 0.2f,
        topK = 40,
        topP = 1.0f,
        maxTokens = 4000
    ),
    QWEN2_5_3B_INSTRUCT(
        path = "/data/local/tmp/Qwen2.5-3B-Instruct_multi-prefill-seq_q8_ekv1280.task",
        url = "https://huggingface.co/litert-community/Qwen2.5-3B-Instruct/resolve/main/Qwen2.5-3B-Instruct_multi-prefill-seq_q8_ekv1280.task",
        preferredBackend = Backend.CPU,
        temperature = 0.2f,
        topK = 40,
        topP = 1.0f,
        maxTokens = 4000
    );
    companion object {
        fun defaultModel(): SlmModel = GEMMA_3_1B_IT_CPU
        fun getAllModels(): List<SlmModel> = entries
    }
}