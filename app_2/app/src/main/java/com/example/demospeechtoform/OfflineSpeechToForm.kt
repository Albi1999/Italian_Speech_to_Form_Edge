package com.example.demospeechtoform

import android.Manifest
import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.Log
import androidx.constraintlayout.widget.ConstraintLayout
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
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

    private var isRecording = false

    private val confirmedTranscript = StringBuilder()
    private var currentPartialTranscript: String = ""

    private var recordingAnimator: ObjectAnimator? = null

    companion object {
        private const val TAG = "OfflineSpeechToForm"
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
                isGranted: Boolean ->
            if (isGranted) {
                Log.i(TAG, "Permesso RECORD_AUDIO concesso.")
                Toast.makeText(this, "Permesso microfono concesso.", Toast.LENGTH_SHORT).show()
                buttonStart.isEnabled = true
            } else {
                Log.w(TAG, "Permesso RECORD_AUDIO negato.")
                if (!shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_DENIED) {
                    Log.w(TAG, "Permesso negato e rationale non necessaria. Suggerisci impostazioni.")
                    showDialogToOpenAppSettings()
                } else {
                    Toast.makeText(this, "Permesso microfono negato. Impossibile avviare la trascrizione.", Toast.LENGTH_LONG).show()
                }
                buttonStart.isEnabled = true
                updateRecordingUI(false)
            }
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

        buttonStart = findViewById(R.id.buttonBeginSpeechToText)
        buttonResume = findViewById(R.id.buttonResumeSpeechToText)
        buttonShowResults = findViewById(R.id.buttonShowResult)
        textView = findViewById(R.id.scrollableText)

        updateDisplayedText()
        initTranscriber()

        setButtonTint(buttonShowResults, R.color.button_background_action_blue)

        transcriber.addListener(object : TextChunkListener {
            override fun onPartialResult(text: String) {
                runOnUiThread {
                    currentPartialTranscript = text
                    updateDisplayedText(isPartial = true)
                    Log.d(TAG, "Partial result UI: $text")
                }
            }

            override fun onFinalResult(text: String) {
                runOnUiThread {
                    if (text.isNotEmpty()) {
                        confirmedTranscript.append(text).append(" ")
                    }
                    currentPartialTranscript = ""
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
                if (checkAndRequestAudioPermission()) {
                    confirmedTranscript.clear()
                    currentPartialTranscript = ""
                    updateDisplayedText()
                    Log.d(TAG, "Riprendi (nuova sessione): Avvio trascrizione...")
                    transcriber.begin()
                }
            } else {
                Toast.makeText(this, "Registrazione già in corso.", Toast.LENGTH_SHORT).show()
            }
            Log.d(TAG, "Button Resume clicked")
        }


        buttonShowResults.setOnClickListener {
            val fullText = confirmedTranscript.toString().trim() + if (currentPartialTranscript.isNotEmpty()) " ($currentPartialTranscript)" else ""
            Log.d(TAG, "Button Show Results clicked. Testo attuale: $fullText")
            Toast.makeText(this, "Testo attuale: $fullText", Toast.LENGTH_LONG).show()
        }
        updateRecordingUI(false)
    }

    @SuppressLint("SetTextI18n")
    private fun updateDisplayedText(isPartial: Boolean = false) {
        val confirmedTextToShow = confirmedTranscript.toString()

        if (isPartial && currentPartialTranscript.isNotEmpty()) {
            val spannable = SpannableString(confirmedTextToShow + currentPartialTranscript)
            val partialColor = ContextCompat.getColor(this, R.color.teal_200)
            try {
                spannable.setSpan(
                    ForegroundColorSpan(partialColor),
                    confirmedTextToShow.length,
                    confirmedTextToShow.length + currentPartialTranscript.length,
                    Spanned.SPAN_EXCLUSIVE_EXCLUSIVE
                )
            } catch (e: IndexOutOfBoundsException) {
                Log.e(TAG, "Errore nello spannable: ${e.message}")
                textView.text = confirmedTextToShow + currentPartialTranscript
                return
            }
            textView.text = spannable
        } else {
            textView.text = confirmedTextToShow
        }
    }


    private fun initTranscriber() {
        transcriber = AndroidSpeechRecognizer(this, object : TranscriberStatusListener {
            override fun onReadyForSpeech() {
                runOnUiThread {
                    updateRecordingUI(true)
                }
            }

            override fun onBeginningOfSpeech() {
                // runOnUiThread { textView.append("Inizio del parlato...\n") }
            }

            override fun onEndOfSpeech() {
                // runOnUiThread { textView.append("Fine del parlato rilevata. Elaborazione...\n") }
            }

            override fun onError(errorDescription: String) {
                runOnUiThread {
                    confirmedTranscript.append("\n--- Errore: $errorDescription ---\n")
                    updateDisplayedText()
                    Toast.makeText(this@OfflineSpeechToForm, "Errore STT: $errorDescription", Toast.LENGTH_LONG).show()
                    updateRecordingUI(false)
                }
            }
        })
        Log.d(TAG, "Transcriber inizializzato.")
    }

    private fun handleStartStopRecording() {
        if (!isRecording) {
            if (checkAndRequestAudioPermission()) {
                confirmedTranscript.clear()
                currentPartialTranscript = ""
                updateDisplayedText()
                Log.d(TAG, "Avvio trascrizione...")
                transcriber.begin()
            }
        } else {
            Log.d(TAG, "Stop trascrizione...")
            transcriber.stop()
            updateRecordingUI(false)
        }
    }

    private fun checkAndRequestAudioPermission(): Boolean {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.i(TAG, "Permesso RECORD_AUDIO già concesso.")
                return true
            }
            shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> {
                Log.i(TAG, "Mostra spiegazione per permesso RECORD_AUDIO.")
                Toast.makeText(this, "L'app necessita del permesso microfono per trascrivere l'audio.", Toast.LENGTH_LONG).show()
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                return false
            }
            else -> {
                Log.i(TAG, "Richiesta permesso RECORD_AUDIO.")
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                return false
            }
        }
    }


    private fun updateRecordingUI(isCurrentlyRecording: Boolean) {
        isRecording = isCurrentlyRecording
        runOnUiThread {
            if (isCurrentlyRecording) {
                buttonStart.text = getString(R.string.stop_transcription_button_text)
                setButtonTint(buttonStart, R.color.button_background_recording_red)
                startRecordingAnimation()
            } else {
                buttonStart.text = getString(R.string.start_transcription_button_text)
                setButtonTint(buttonStart, R.color.button_background_default_green)
                stopRecordingAnimation()
            }
            buttonStart.isEnabled = true
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

    private fun showDialogToOpenAppSettings() {
        AlertDialog.Builder(this)
            .setTitle("Permesso Necessario")
            .setMessage("L'app necessita del permesso per il microfono per funzionare. Per favore, abilita il permesso nelle impostazioni dell'app.")
            .setPositiveButton("Apri Impostazioni") { dialog, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                val uri = Uri.fromParts("package", packageName, null)
                intent.data = uri
                startActivity(intent)
                dialog.dismiss()
            }
            .setNegativeButton("Annulla") { dialog, _ ->
                dialog.dismiss()
                Toast.makeText(this, "Permesso microfono non concesso.", Toast.LENGTH_SHORT).show()
            }
            .setCancelable(false)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopRecordingAnimation()
        if (::transcriber.isInitialized && transcriber is AndroidSpeechRecognizer) {
            (transcriber as AndroidSpeechRecognizer).destroy()
        }
        Log.d(TAG, "OfflineSpeechToFormActivity distrutta e transcriber rilasciato.")
    }
}