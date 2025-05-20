package com.example.demospeechtoform

import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import androidx.constraintlayout.widget.ConstraintLayout
import android.view.animation.LinearInterpolator
import android.widget.Button
// import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.DrawableCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class OfflineSpeechToForm : AppCompatActivity() {

    private lateinit var transcriber: Transcriber
    private lateinit var buttonStart: Button
    private lateinit var buttonResume: Button
    private lateinit var buttonShowResults: Button
    private lateinit var textView: TextView
    // private lateinit var progressBar: ProgressBar // Al momento non utilizzato in quanto non abbiamo un modello che esegue il text to form

    // Recording state
    private var isRecording = false
    private val confirmedTranscript = StringBuilder()
    private var currentPartialTranscript: String = ""
    private var recordingAnimator: ObjectAnimator? = null

    companion object {
        private const val TAG = "OfflineSpeechToForm"
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

        setContentView(R.layout.activity_offline_speech_to_form)

        val mainLayout = findViewById<ConstraintLayout>(R.id.OfflineSpeechToForm)
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
            Toast.makeText(this, "Testo attuale: $fullText", Toast.LENGTH_LONG).show()
        }
        updateRecordingUI(false)
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
                    Toast.makeText(this@OfflineSpeechToForm, "${getString(R.string.stt_error_prefix)}: $errorDescription", Toast.LENGTH_LONG).show()
                    updateRecordingUI(false)
                }
            }
        })
        Log.d(TAG, "Transcriber initialized for OfflineSpeechToForm.")
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