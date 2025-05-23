package com.example.demospeechtoform

import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.os.Bundle
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import androidx.constraintlayout.widget.ConstraintLayout
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class OfflineSpeechToForm : AppCompatActivity() {

    private val viewModel: SpeechToFormViewModel by viewModels()

    private lateinit var buttonStart: Button
    private lateinit var buttonResume: Button
    private lateinit var buttonShowResults: Button
    private lateinit var textView: TextView
    private lateinit var progressBar: ProgressBar

    private var isRecordingUiState = false
    private var recordingAnimator: ObjectAnimator? = null

    companion object {
        private const val TAG = "OfflineSpeechToForm"
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

        buttonStart = findViewById(R.id.buttonBeginSpeechToText)
        buttonResume = findViewById(R.id.buttonResumeSpeechToText)
        buttonShowResults = findViewById(R.id.buttonShowResult)
        textView = findViewById(R.id.scrollableText)
        progressBar = findViewById(R.id.progressBarOffline)

        // Imposta il modello di default o permetti la selezione
        // viewModel.selectLlmModelForOfflineMode(SlmModel.NOME_ALTRO_MODELLO_CPU) // Esempio per cambiare modello
        viewModel.initTranscriberAndMode(isOnline = false)

        lifecycleScope.launch {
            viewModel.uiState.collect { uiState ->
                Log.d(TAG, "New UI State collected: $uiState")
                updateUi(uiState)
            }
        }

        buttonStart.setOnClickListener {
            viewModel.handleStartStopRecording(isRecordingUiState)
        }

        buttonResume.setOnClickListener {
            val currentState = viewModel.uiState.value
            if (!isRecordingUiState &&
                currentState !is SpeechToFormUiState.ModelLoading &&
                currentState !is SpeechToFormUiState.ProcessingSlm &&
                currentState !is SpeechToFormUiState.Transcribing
            ) {
                Log.d(TAG, "Button Resume clicked.")
                textView.text = ""
                viewModel.startTranscription()
            } else if (isRecordingUiState || currentState is SpeechToFormUiState.Transcribing) {
                Toast.makeText(
                    this,
                    getString(R.string.recording_already_in_progress),
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                Toast.makeText(this, getString(R.string.processing_wait), Toast.LENGTH_SHORT).show()
            }
        }

        buttonShowResults.setOnClickListener {
            val currentUiState = viewModel.uiState.value
            var textForSlm = ""

            if (currentUiState is SpeechToFormUiState.TranscriptionComplete) {
                textForSlm = currentUiState.fullText
                Log.d(TAG, "Show Results button: Testo da trascrizione completa: '$textForSlm'")
            } else if (currentUiState is SpeechToFormUiState.AwaitingTranscription ||
                currentUiState is SpeechToFormUiState.Error ||
                textView.text.toString() == getString(R.string.awaiting_transcription) ||
                (currentUiState is SpeechToFormUiState.Error && textView.text.toString() == currentUiState.message)
            ) {
                Log.d(
                    TAG,
                    "Show Results button: Nessuna trascrizione valida, si attiverà il testo dummy nel ViewModel."
                )
                textForSlm = ""
            } else if (currentUiState is SpeechToFormUiState.SlmResult || currentUiState is SpeechToFormUiState.ProcessingSlm) {
                Toast.makeText(
                    this,
                    getString(R.string.results_already_shown_or_processing),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            } else if (currentUiState is SpeechToFormUiState.ModelLoading) {
                Toast.makeText(
                    this,
                    getString(R.string.model_loading_message, viewModel.selectedLlmModel.name),
                    Toast.LENGTH_LONG
                ).show()
                return@setOnClickListener
            } else {
                Log.d(
                    TAG,
                    "Show Results button: Stato UI non ideale per estrazione diretta, si attiverà il testo dummy."
                )
                textForSlm = ""
            }
            viewModel.processTranscriptionWithSlm(textForSlm)
        }
    }

    @SuppressLint("SetTextI18n")
    private fun updateUi(uiState: SpeechToFormUiState) {
        progressBar.visibility = View.GONE
        val buttonsEnabled = uiState !is SpeechToFormUiState.ModelLoading &&
                uiState !is SpeechToFormUiState.ProcessingSlm &&
                uiState !is SpeechToFormUiState.Transcribing &&
                uiState !is SpeechToFormUiState.ProcessingApi

        buttonStart.isEnabled = buttonsEnabled
        buttonResume.isEnabled = buttonsEnabled
        buttonShowResults.isEnabled = buttonsEnabled


        when (uiState) {
            is SpeechToFormUiState.ModelLoading -> {
                textView.text = uiState.message
                progressBar.visibility = View.VISIBLE
                updateButtonAnimation(false)
                buttonStart.isEnabled = false
                buttonResume.isEnabled = false
                buttonShowResults.isEnabled = false
            }

            SpeechToFormUiState.AwaitingTranscription -> {
                textView.text = getString(R.string.awaiting_transcription)
                updateButtonAnimation(false)
                buttonShowResults.isEnabled = true
            }

            is SpeechToFormUiState.Transcribing -> {
                updateButtonAnimation(true)
                buttonResume.isEnabled = false
                buttonShowResults.isEnabled = false
            }

            is SpeechToFormUiState.TranscriptionComplete -> {
                textView.text =
                    uiState.fullText.ifEmpty { getString(R.string.no_transcription_result_using_dummy) }
                updateButtonAnimation(false)
                buttonShowResults.isEnabled = true
            }

            SpeechToFormUiState.ProcessingSlm -> {
                textView.text = getString(R.string.processing_with_slm)
                progressBar.visibility = View.VISIBLE
                updateButtonAnimation(false)
                buttonStart.isEnabled = false
                buttonResume.isEnabled = false
                buttonShowResults.isEnabled = false
            }

            is SpeechToFormUiState.SlmResult -> {
                textView.text = uiState.formattedData
                updateButtonAnimation(false)
                buttonShowResults.isEnabled = false
            }

            is SpeechToFormUiState.Error -> {
                textView.text = uiState.message
                updateButtonAnimation(false)
                Toast.makeText(this, uiState.message, Toast.LENGTH_LONG).show()
                buttonResume.isEnabled = true
                val isModelSetupError = uiState.message.contains(
                    getString(
                        R.string.slm_model_setup_error_detailed,
                        ""
                    ).substringBefore("%1\$s")
                ) ||
                        uiState.message == getString(R.string.slm_model_not_ready_retry)
                buttonShowResults.isEnabled = !isModelSetupError
            }

            SpeechToFormUiState.ProcessingApi, is SpeechToFormUiState.ApiResult -> {
                Log.w(TAG, "Stato UI Online inatteso in OfflineSpeechToForm: $uiState")
                updateButtonAnimation(false)
                textView.text =
                    getString(R.string.unexpected_error_prefix) + " (Stato Online Inatteso)"
            }
        }
    }

    private fun updateButtonAnimation(isCurrentlyRecording: Boolean) {
        this.isRecordingUiState = isCurrentlyRecording
        if (isCurrentlyRecording) {
            buttonStart.text = getString(R.string.stop_transcription_button_text)
            buttonStart.backgroundTintList = ContextCompat.getColorStateList(this, R.color.button_background_recording_red)
            startRecordingAnimation()
        } else {
            buttonStart.text = getString(R.string.start_transcription_button_text)
            buttonStart.backgroundTintList = ContextCompat.getColorStateList(this, R.color.button_background_default_green)
            stopRecordingAnimation()
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
        Log.d(TAG, "OfflineSpeechToForm Activity destroyed.")
    }
}