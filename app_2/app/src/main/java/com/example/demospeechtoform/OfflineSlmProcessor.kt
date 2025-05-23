// File: OfflineSlmProcessor.kt
package com.example.demospeechtoform

import android.content.Context
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class OfflineSlmProcessor private constructor(
    context: Context,
    internal val modelConfig: SlmModel,
    absoluteModelPath: String
) {
    private lateinit var llmInference: LlmInference
    private var llmInferenceSession: LlmInferenceSession? = null

    var isInitialized = false
        private set

    companion object {
        private const val TAG = "OfflineSlmProcessor"
        private var instance: OfflineSlmProcessor? = null

        /**
         * Ottiene un'istanza (singleton) di OfflineSlmProcessor per un dato modello.
         * Gestisce la copia del modello da assets e l'inizializzazione.
         * Questo metodo è ora suspend perché la copia e l'init sono I/O.
         */
        fun getInstance(
            appContext: Context,
            modelToLoad: SlmModel
        ): OfflineSlmProcessor? {
            if (instance != null && instance!!.modelConfig == modelToLoad && instance!!.isInitialized) {
                Log.d(TAG, "Returning existing instance for model: ${modelToLoad.name}")
                return instance
            }
            Log.d(TAG, "Creating new instance for model: ${modelToLoad.name}")
            instance?.close()

            val modelAssetFilename = File(modelToLoad.path).name
            val internalModelFile = File(appContext.filesDir, modelAssetFilename)
            val effectiveModelPath: String

            if (internalModelFile.exists()) {
                Log.i(TAG, "Model '${modelAssetFilename}' found in internal storage: ${internalModelFile.absolutePath}")
                effectiveModelPath = internalModelFile.absolutePath
            } else {
                Log.i(TAG, "Copying model '${modelAssetFilename}' from assets...")
                try {
                    appContext.assets.open(modelAssetFilename).use { inputStream ->
                        FileOutputStream(internalModelFile).use { outputStream ->
                            inputStream.copyTo(outputStream)
                        }
                    }
                    effectiveModelPath = internalModelFile.absolutePath
                    Log.i(TAG, "Model copied to: $effectiveModelPath")
                } catch (e: IOException) {
                    Log.e(TAG, "Error copying model '${modelAssetFilename}': ${e.message}", e)
                    return null
                }
            }
            val newInstance = OfflineSlmProcessor(appContext, modelToLoad, effectiveModelPath)
            if (newInstance.isInitialized) {
                instance = newInstance
                return instance
            } else {
                Log.e(TAG, "Failed to initialize new OfflineSlmProcessor instance for ${modelToLoad.name}")
                return null
            }
        }

        fun resetInstance() {
            instance?.close()
            instance = null
        }
    }

    init {
        try {
            Log.i(TAG, "Initializing LlmInference engine for model: ${modelConfig.name} at $absoluteModelPath")
            val inferenceEngineOptions = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(absoluteModelPath)
                .setMaxTokens(modelConfig.maxTokens)
                //.apply { modelConfig.preferredBackend?.let { setPreferredBackend(it) } }
                .build()
            llmInference = LlmInference.createFromOptions(context, inferenceEngineOptions)
            Log.i(TAG, "LlmInference engine created for ${modelConfig.name}.")

            recreateSession()

            isInitialized = true
            Log.i(TAG, "OfflineSlmProcessor for '${modelConfig.name}' initialized successfully (engine & session).")

        } catch (e: Exception) {
            isInitialized = false
            Log.e(TAG, "Error during OfflineSlmProcessor initialization for '${modelConfig.name}': ${e.message}", e)
        }
    }

    private fun recreateSession() {
        if (!::llmInference.isInitialized) {
            Log.e(TAG, "Cannot create session: LlmInference engine not initialized for ${modelConfig.name}.")
            throw IllegalStateException("LlmInference engine not initialized before creating session.")
        }
        llmInferenceSession?.close()

        Log.i(TAG, "Creating LlmInferenceSession for model: ${modelConfig.name}")
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTemperature(modelConfig.temperature)
            .setTopK(modelConfig.topK)
            .setTopP(modelConfig.topP)
            .build()
        try {
            llmInferenceSession = LlmInferenceSession.createFromOptions(llmInference, sessionOptions)
            Log.i(TAG, "LlmInferenceSession created for ${modelConfig.name}.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create LlmInferenceSession for ${modelConfig.name}: ${e.message}", e)
            throw RuntimeException("Failed to create LlmInferenceSession for ${modelConfig.name}", e)
        }
    }

    /**
     * Genera una risposta in modo sincrono.
     * Per risposte in streaming (asincrone), useresti generateResponseAsync.
     */
    fun generateResponse(fullPrompt: String): String {
        if (!isInitialized || llmInferenceSession == null) {
            val errorMsg = "OfflineSlmProcessor (model: ${modelConfig.name}) not ready or session is null. Init: $isInitialized"
            Log.e(TAG, errorMsg)
            if (!isInitialized) {
                Log.w(TAG, "Attempting to re-initialize in generateResponse (this is a fallback)...")
                if(::llmInference.isInitialized) {
                    try {
                        recreateSession()
                        isInitialized = true
                        Log.i(TAG, "Session recreated on demand.")
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to recreate session on demand.", e)
                        isInitialized = false
                        return "Error: Failed to prepare session on demand for ${modelConfig.name}"
                    }
                } else {
                    return "Error: LLM Engine not available for ${modelConfig.name}"
                }
            }
            if (llmInferenceSession == null) return "Error: LLM Session is null for ${modelConfig.name}"
        }

        return try {
            Log.i(TAG, "Generating response with model '${modelConfig.name}'...")
            if (::llmInference.isInitialized) {
                Log.w(TAG, "Using LlmInference.generateResponse (engine-level). Session-specific options (temp, topK) might not apply here directly.")
                val result = llmInference.generateResponse(fullPrompt)
                Log.d(TAG, "Raw result from MediaPipe LLM Engine for ${modelConfig.name}: '$result'")
                result ?: "Error: LLM Inference (engine) returned null."
            } else {
                "Error: LLM Inference engine not initialized."
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error during MediaPipe LLM inference for model '${modelConfig.name}'", e)
            "Error during LLM inference: ${e.localizedMessage}"
        }
    }


    fun close() {
        if (isInitialized) {
            llmInferenceSession?.close()
            if (::llmInference.isInitialized) llmInference.close()
            isInitialized = false
            Log.i(TAG, "OfflineSlmProcessor for '${modelConfig.name}' closed.")
        }
    }
}