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
                .apply { modelConfig.preferredBackend?.let { setPreferredBackend(it) } }
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
     * Genera una risposta in modo sincrono utilizzando la sessione LLM configurata.
     */
    fun generateResponseSlm(fullPrompt: String): String {
        if (!isInitialized || llmInferenceSession == null) {
            val errorMsg = "OfflineSlmProcessor (model: ${modelConfig.name}) not ready or session is null. Init: $isInitialized, Session: $llmInferenceSession"
            Log.e(TAG, errorMsg)
            if (!isInitialized && ::llmInference.isInitialized) {
                Log.w(TAG, "Attempting to re-initialize session in generateResponseSlm (this is a fallback)...")
                try {
                    recreateSession()
                    if (llmInferenceSession != null) {
                        isInitialized = true
                        Log.i(TAG, "Session recreated on demand.")
                    } else {
                        return "Error: Failed to prepare session on demand for ${modelConfig.name} (session still null)."
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to recreate session on demand.", e)
                    isInitialized = false
                    return "Error: Failed to prepare session on demand for ${modelConfig.name} (exception)."
                }
            } else if (llmInferenceSession == null) {
                return "Error: LLM Session is definitively null for ${modelConfig.name}."
            }
        }

        return try {
            Log.i(TAG, "Adding prompt and generating response with model '${modelConfig.name}' using LlmInferenceSession...")
            llmInferenceSession!!.addQueryChunk(fullPrompt) //
            Log.d(TAG, "Prompt chunk added to session for model ${modelConfig.name}.")
            val result = llmInferenceSession!!.generateResponse()
            Log.d(TAG, "Raw result from MediaPipe LLM Session for ${modelConfig.name}: '$result'")
            result ?: "Error: LLM Inference Session returned null after generateResponse."
        } catch (e: Exception) {
            Log.e(TAG, "Error during MediaPipe LLM session interaction for model '${modelConfig.name}'", e)
            "Error during LLM session interaction: ${e.localizedMessage}"
        }
    }

    fun close() {
        if (isInitialized || llmInferenceSession != null || ::llmInference.isInitialized) {
            llmInferenceSession?.close()
            llmInferenceSession = null
            if (::llmInference.isInitialized) {
                llmInference.close()
            }
            isInitialized = false
            Log.i(TAG, "OfflineSlmProcessor for '${modelConfig.name}' closed.")
        }
    }
}