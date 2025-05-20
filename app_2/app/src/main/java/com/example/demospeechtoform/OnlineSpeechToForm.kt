package com.example.demospeechtoform

// Android and System imports
import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
// Activity and Lifecycle imports
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
// Moshi and Retrofit related imports
import com.squareup.moshi.JsonAdapter
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.launch
import retrofit2.HttpException
import java.io.IOException
import java.io.InputStream

class OnlineSpeechToForm : AppCompatActivity() {

    private lateinit var transcriber: Transcriber
    private lateinit var buttonStart: Button
    private lateinit var buttonResume: Button
    private lateinit var buttonShowResults: Button
    private lateinit var textView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var apiClient: ApiClient

    // Recording State
    private var isRecording = false
    private val confirmedTranscript = StringBuilder()
    private var currentPartialTranscript: String = ""
    private var recordingAnimator: ObjectAnimator? = null

    // Moshi instance for JSON parsing
    private val moshi: Moshi by lazy {
        Moshi.Builder()
            .add(KotlinJsonAdapterFactory())
            .build()
    }

    companion object {
        private const val TAG = "OnlineSpeechToForm"
        private const val REPORT_SCHEMA_FILENAME = "macro_schema.json" // Name of your JSON file in assets
    }

    fun setButtonTint(button: Button, colorResId: Int) {
        var buttonBackground = button.background
        buttonBackground = DrawableCompat.wrap(buttonBackground).mutate()
        DrawableCompat.setTint(buttonBackground, ContextCompat.getColor(button.context, colorResId))
        button.background = buttonBackground
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_online_speech_to_form)

        val mainLayout = findViewById<androidx.constraintlayout.widget.ConstraintLayout>(R.id.OnlineSpeechToForm)
        ViewCompat.setOnApplyWindowInsetsListener(mainLayout) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }
        // Initialize UI elements
        buttonStart = findViewById(R.id.buttonBeginSpeechToText)
        buttonResume = findViewById(R.id.buttonResumeSpeechToText)
        buttonShowResults = findViewById(R.id.buttonShowResult)
        textView = findViewById(R.id.scrollableText)
        progressBar = findViewById(R.id.progressBarOnline)

        // Set initial text for the TextView
        textView.text = getString(R.string.awaiting_transcription)

        initTranscriber()

        transcriber.addListener(object : TextChunkListener {
            override fun onPartialResult(text: String) {
                runOnUiThread {
                    currentPartialTranscript = text
                    updateDisplayedText(isPartial = true)
                }
            }

            override fun onFinalResult(text: String) {
                runOnUiThread {
                    if (text.isNotEmpty()) {
                        confirmedTranscript.append(text).append(" ")
                    }
                    currentPartialTranscript = "" // Clear partial once final is received
                    updateDisplayedText()
                    Log.d(TAG, "Final result UI: $text")
                }
            }
        })

        buttonStart.setOnClickListener {
            handleStartStopRecording()
        }

        buttonResume.setOnClickListener {
            if (!isRecording) {
                confirmedTranscript.clear()
                currentPartialTranscript = ""
                updateDisplayedText()
                Log.d(TAG, "New Session: Starting transcription...")
                transcriber.begin()
            } else {
                Toast.makeText(this, getString(R.string.recording_already_in_progress), Toast.LENGTH_SHORT).show()
            }
        }

        buttonShowResults.setOnClickListener {
            val fullText = confirmedTranscript.toString().trim()
            Log.d(TAG, "Submit button clicked. Full text: $fullText")

            if (fullText.isBlank()) {
                Toast.makeText(this, getString(R.string.no_text_to_process), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (BuildConfig.API_KEY.isBlank() ){
                Toast.makeText(this, getString(R.string.api_key_not_set), Toast.LENGTH_LONG).show()
                Log.e(TAG, "API Key is not set in BuildConfig.")
                textView.text = getString(R.string.api_key_not_set)
                return@setOnClickListener
            }

            callApiToExtractData(fullText)
        }
        updateRecordingUI(false) // Set initial UI state for recording buttons

        val reportSchema = loadReportSchemaFromAssets()
        if (reportSchema == null) {
            progressBar.visibility = View.GONE
            buttonShowResults.isEnabled = true
            return
        }

        apiClient = ApiClient(reportSchema.requestedInformations, reportSchema.topic, reportSchema.language)
    }

    /**
    Loads the report schema (requested_information, topic, language) from the assets JSON file.
     */
    private fun loadReportSchemaFromAssets(): ReportSchemaAsset? {
        return try {
            val inputStream: InputStream = assets.open(REPORT_SCHEMA_FILENAME)
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val adapter: JsonAdapter<ReportSchemaAsset> = moshi.adapter(ReportSchemaAsset::class.java)
            adapter.fromJson(jsonString)
        } catch (e: IOException) {
            Log.e(TAG, "Error reading or parsing $REPORT_SCHEMA_FILENAME from assets", e)
            runOnUiThread { // Ensure UI updates are on the main thread
                Toast.makeText(this, getString(R.string.error_loading_form_schema), Toast.LENGTH_LONG).show()
                textView.text = getString(R.string.error_loading_form_schema)
            }
            null
        } catch (e: Exception) { // Catch other potential parsing errors
            Log.e(TAG, "Unexpected error parsing $REPORT_SCHEMA_FILENAME", e)
            runOnUiThread {
                Toast.makeText(this, getString(R.string.error_loading_form_schema), Toast.LENGTH_LONG).show()
                textView.text = getString(R.string.error_loading_form_schema)
            }
            null
        }
    }

    /**
     * Formats the API response Map (Map<String, Any?>) into a human-readable string.
     */
    private fun formatApiResponseMap(data: Map<String, Any?>): String {
        val builder = StringBuilder(getString(R.string.extracted_information_title))
        try {
            @Suppress("UNCHECKED_CAST")
            val report : Map<String, Any?> = data["report"] as Map<String, Any?>
            for ((key, value) in report) {
                // Format the key: replace underscores with spaces and capitalize words
                val formattedKey = key.replace("_", " ").split(" ")
                    .joinToString(" ") { it.replaceFirstChar(Char::titlecase) }
                builder.append("$formattedKey: ")
                appendValueToBuilder(builder, value, 1)
                builder.append("\n\n") // Add an extra newline for better separation between top-level items
            }
            return builder.toString().trim()
        } catch(_: Exception) {
            return "Errore Inaspettato"
        }

    }

    /**
     * Makes the API call to the configured endpoint to extract form data.
     * Uses ApiClient.apiService.
     */
    @SuppressLint("SetTextI18n")
    private fun callApiToExtractData(transcribedText: String) {
        progressBar.visibility = View.VISIBLE
        buttonShowResults.isEnabled = false
        textView.text = getString(R.string.processing_with_api)



        lifecycleScope.launch {
            try {
                val response = apiClient.callApi(transcribedText)

                if (response.isSuccessful) {
                    val responseBodyMap = response.body() // This is Map<String, Any?>?
                    if (responseBodyMap != null) {
                        Log.i(TAG, "API Raw Response: $responseBodyMap")
                        val formattedResult = formatApiResponseMap(responseBodyMap)
                        textView.text = formattedResult
                    } else {
                        textView.text = getString(R.string.api_response_empty_successful)
                        Log.w(TAG, "API response body is null but successful.")
                    }
                } else {
                    val errorBody = response.errorBody()?.string()
                    Log.e(TAG, "API Call failed: ${response.code()} - $errorBody")
                    var errorMsg = "${getString(R.string.error_from_api_prefix)}: ${response.code()}"
                    if (!errorBody.isNullOrBlank()) {
                        try {
                            val errorAdapter = moshi.adapter(ApiError::class.java)
                            val apiError = errorAdapter.fromJson(errorBody)
                            errorMsg += "\nMessage: ${apiError?.message ?: errorBody}"
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not parse error body as ApiError: $errorBody", e)
                            errorMsg += "\n$errorBody"
                        }
                    }
                    textView.text = errorMsg
                }
            } catch (e: HttpException) {
                Log.e(TAG, "HTTP Exception: ${e.message()}", e)
                textView.text = "${getString(R.string.network_error_http_prefix)}: ${e.message()}"
            } catch (e: IOException) { // Covers network and file IO issues
                Log.e(TAG, "IO Exception (Network or File): ${e.message}", e)
                textView.text = "${getString(R.string.network_or_data_error_prefix)}: ${e.message}"
            } catch (e: Exception) { // Generic catch-all for unexpected errors
                Log.e(TAG, "Generic Exception during API call: ${e.message}", e)
                textView.text = "${getString(R.string.unexpected_error_prefix)}: ${e.message}"
            } finally {
                progressBar.visibility = View.GONE
                buttonShowResults.isEnabled = true
            }
        }
    }



    /**
     * Recursively appends values from the map/list to the StringBuilder for display, with indentation.
     */
    @Suppress("UNCHECKED_CAST")
    private fun appendValueToBuilder(builder: StringBuilder, value: Any?, indentLevel: Int) {
        val indent = "  ".repeat(indentLevel)

        when (value) {
            is Map<*, *> -> { // If the value is a map (nested JSON object)
                builder.append("\n") // Newline before starting the nested object
                (value as? Map<String, Any?>)?.forEach { (k, v) ->
                    val formattedKey = k.replace("_", " ").split(" ")
                        .joinToString(" ") { it.replaceFirstChar(Char::titlecase) }
                    builder.append("$indent$formattedKey: ")
                    appendValueToBuilder(builder, v, indentLevel + 1) // Recursive call for nested value
                    builder.append("\n") // Newline after each key-value pair in the nested object
                }
            }
            is List<*> -> { // If the value is a list (JSON array)
                builder.append("\n") // Newline before starting the list
                value.forEachIndexed { index, item ->
                    builder.append("$indent${getString(R.string.list_item_prefix)} ${index + 1}:\n")
                    appendValueToBuilder(builder, item, indentLevel + 1) // Each item in list could be a map or simple value
                    if (item !is Map<*, *>) builder.append("\n") // Add newline if item isn't a map (which adds its own newlines)
                }
            }
            null -> builder.append(getString(R.string.not_available_shorthand)) // "N/A" for null values
            else -> builder.append(value.toString())
        }
    }

    @SuppressLint("SetTextI18n")
    private fun updateDisplayedText(isPartial: Boolean = false) {
        val confirmedTextToShow = confirmedTranscript.toString()
        val displayText: CharSequence = if (isPartial && currentPartialTranscript.isNotEmpty()) {
            val spannable = SpannableString(confirmedTextToShow + currentPartialTranscript)
            val partialColor = ContextCompat.getColor(this, R.color.teal_200)
            try {
                spannable.setSpan(
                    ForegroundColorSpan(partialColor),
                    confirmedTextToShow.length,
                    spannable.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
                spannable
            } catch (e: IndexOutOfBoundsException) {
                Log.e(TAG, "Error in spannable string for partial transcript: ${e.message}")
                confirmedTextToShow + currentPartialTranscript
            }
        } else {
            confirmedTextToShow.ifEmpty { getString(R.string.awaiting_transcription) }
        }
        textView.text = displayText
    }

    private fun initTranscriber() {
        transcriber = AndroidSpeechRecognizer(this, object : TranscriberStatusListener {
            override fun onReadyForSpeech() {
                runOnUiThread { updateRecordingUI(true) }
            }
            override fun onBeginningOfSpeech() { /* Not used for now */ }
            override fun onEndOfSpeech() { /* Not used for now, UI update handled by stop */ }
            override fun onError(errorDescription: String) {
                runOnUiThread {
                    Log.e(TAG, "STT Error: $errorDescription")
                    Toast.makeText(this@OnlineSpeechToForm, "${getString(R.string.stt_error_prefix)}: $errorDescription", Toast.LENGTH_LONG).show()
                    updateRecordingUI(false)
                }
            }
        })
        Log.d(TAG, "Transcriber initialized for OnlineSpeechToForm.")
    }

    private fun handleStartStopRecording() {
        if (!isRecording) {
            confirmedTranscript.clear()
            currentPartialTranscript = ""
            updateDisplayedText()
            Log.d(TAG, "Starting transcription...")
            transcriber.begin()
        } else {
            Log.d(TAG, "Stopping transcription...")
            transcriber.stop()
            updateRecordingUI(false)
        }
    }

    private fun updateRecordingUI(isCurrentlyRecording: Boolean) {
        this.isRecording = isCurrentlyRecording
        runOnUiThread {
            if (isCurrentlyRecording) {
                buttonStart.text = getString(R.string.stop_transcription_button_text)
                setButtonTint(buttonStart, R.color.button_background_recording_red)
                startRecordingAnimation()
                buttonResume.isEnabled = false // Disable "New Session" while recording
                buttonShowResults.isEnabled = false // Disable "Submit" while recording
            } else {
                buttonStart.text = getString(R.string.start_transcription_button_text)
                setButtonTint(buttonStart, R.color.button_background_default_green)
                stopRecordingAnimation()
                buttonResume.isEnabled = true
                buttonShowResults.isEnabled = true
            }
        }
    }

    private fun startRecordingAnimation() {
        stopRecordingAnimation()
        recordingAnimator = ObjectAnimator.ofFloat(buttonStart, "alpha", 1f, 0.5f, 1f).apply {
            duration = 1000
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
            interpolator = LinearInterpolator()
            start()
        }
    }

    private fun stopRecordingAnimation() {
        recordingAnimator?.cancel()
        buttonStart.alpha = 1.0f
        recordingAnimator = null
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecordingAnimation()
        transcriber.destroy()
        Log.d(TAG, "OnlineSpeechToForm Activity destroyed and transcriber released.")
    }
}