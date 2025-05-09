package com.example.offlinespeechtoform

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : ComponentActivity() {

    private lateinit var registraButton: Button
    private lateinit var trascrizioneTextView: TextView
    private lateinit var outputTextView: TextView
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        registraButton = findViewById(R.id.buttonRegistra)
        trascrizioneTextView = findViewById(R.id.textViewTrascrizione)
        outputTextView = findViewById(R.id.textViewOutput)

        // Inizializza il launcher per la richiesta del permesso
        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                startSpeechRecognition()
            } else {
                trascrizioneTextView.text = "Permesso microfono negato."
            }
        }

        // Inizializza SpeechRecognizer e Intent
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "it-IT") // Imposta la lingua italiana
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, packageName)
        }

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                trascrizioneTextView.text = "Pronto a registrare..."
                Log.d("SpeechRecognizer", "onReadyForSpeech")
            }

            override fun onBeginningOfSpeech() {
                trascrizioneTextView.text = "Inizio registrazione..."
                Log.d("SpeechRecognizer", "onBeginningOfSpeech")
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Puoi visualizzare l'intensità del suono qui
                Log.v("SpeechRecognizer", "onRmsChanged: $rmsdB")
            }

            override fun onEndOfSpeech() {
                trascrizioneTextView.text = "Fine registrazione."
                Log.d("SpeechRecognizer", "onEndOfSpeech")
            }

            override fun onError(error: Int) {
                val errorMessage = getErrorText(error)
                Log.e("SpeechRecognizer", "ERROR $error: $errorMessage")
                trascrizioneTextView.text = "Errore riconoscimento: $errorMessage"
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    val spokenText = matches[0]
                    trascrizioneTextView.text = "Testo riconosciuto: $spokenText"
                    outputTextView.text = "In attesa dell'NER..."
                    Log.d("SpeechRecognizer", "onResults: $spokenText")
                } else {
                    trascrizioneTextView.text = "Nessun risultato di riconoscimento."
                    Log.d("SpeechRecognizer", "onResults: nessun risultato")
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val partialMatches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!partialMatches.isNullOrEmpty()) {
                    trascrizioneTextView.text = "Trascrizione parziale: ${partialMatches[0]}"
                    Log.d("SpeechRecognizer", "onPartialResults: ${partialMatches[0]}")
                }
            }

            override fun onBufferReceived(buffer: ByteArray?) {
                Log.d("SpeechRecognizer", "onBufferReceived: buffer size = ${buffer?.size}")
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                Log.d("SpeechRecognizer", "onEvent: type = $eventType, params = $params")
            }
        })

        registraButton.setOnClickListener {
            // Verifica se il permesso è già concesso
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                startSpeechRecognition()
            } else {
                // Richiedi il permesso
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun startSpeechRecognition() {
        trascrizioneTextView.text = "Avvio riconoscimento..."
        speechRecognizer.startListening(speechRecognizerIntent)
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer.destroy() // Rilascia le risorse quando l'Activity viene distrutta
    }

    fun getErrorText(errorCode: Int): String =
        when (errorCode) {
            SpeechRecognizer.ERROR_AUDIO -> "Errore audio"
            SpeechRecognizer.ERROR_CLIENT -> "Errore lato client"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Permessi insufficienti"
            SpeechRecognizer.ERROR_NETWORK -> "Errore di rete"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Timeout di rete"
            SpeechRecognizer.ERROR_NO_MATCH -> "Nessuna corrispondenza"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Servizio di riconoscimento occupato"
            SpeechRecognizer.ERROR_SERVER -> "Errore dal server"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Timeout di input vocale"
            else -> "Errore sconosciuto"
        }
}