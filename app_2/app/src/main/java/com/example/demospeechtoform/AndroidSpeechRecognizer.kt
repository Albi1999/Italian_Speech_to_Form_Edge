package com.example.demospeechtoform

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.core.content.ContextCompat

class AndroidSpeechRecognizer(
    private val context: Context,
    private val statusListener: TranscriberStatusListener?
) : Transcriber {

    private val speechRecognizerAndroid: SpeechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
    private var textChunkListener: TextChunkListener? = null
    private var isListening = false

    companion object {
        private const val TAG = "AndroidSR"
    }

    init {
        Log.d(TAG, "SpeechRecognizer creato.")

        speechRecognizerAndroid.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Log.i(TAG, "onReadyForSpeech")
                isListening = true
                statusListener?.onReadyForSpeech()
            }

            override fun onBeginningOfSpeech() {
                Log.i(TAG, "onBeginningOfSpeech")
                statusListener?.onBeginningOfSpeech()
            }

            override fun onRmsChanged(rmsdB: Float) {
                // Commenta questa riga se i log V sono troppi:
                // Log.v(TAG, "onRmsChanged: $rmsdB")
            }

            override fun onBufferReceived(buffer: ByteArray?) {
                Log.d(TAG, "onBufferReceived")
            }

            override fun onEndOfSpeech() {
                Log.i(TAG, "onEndOfSpeech")
                statusListener?.onEndOfSpeech()
            }

            override fun onError(error: Int) {
                val errorMessage = getErrorText(error)
                Log.e(TAG, "onError: $errorMessage (code: $error)")
                isListening = false
                statusListener?.onError(errorMessage)
            }

            override fun onResults(results: Bundle?) {
                Log.i(TAG, "onResults")
                isListening = false
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null && matches.isNotEmpty()) {
                    val text = matches[0]
                    Log.d(TAG, "Risultato finale: $text")
                    textChunkListener?.onFinalResult(text)
                } else {
                    Log.d(TAG, "Nessun risultato finale.")
                    textChunkListener?.onFinalResult("")
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                Log.d(TAG, "onPartialResults")
                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null && matches.isNotEmpty()) {
                    val partialText = matches[0]
                    if (partialText.isNotBlank()) {
                        Log.d(TAG, "Risultato parziale: $partialText")
                        textChunkListener?.onPartialResult(partialText)
                    }
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                Log.d(TAG, "onEvent: $eventType")
            }
        })
    }

    override fun addListener(listener: TextChunkListener) {
        this.textChunkListener = listener
        Log.d(TAG, "Listener aggiunto.")
    }

    override fun removeListener(listener: TextChunkListener) {
        if (this.textChunkListener == listener) {
            this.textChunkListener = null
            Log.d(TAG, "Listener rimosso.")
        }
    }

    override fun begin() {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "begin() chiamato senza permesso RECORD_AUDIO.")
            statusListener?.onError("Permesso microfono non concesso.")
            return
        }

        if (isListening) {
            Log.w(TAG, "begin() chiamato mentre si è già in ascolto. Ignorato.")
            return
        }

        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            val errorMessage = "Riconoscimento vocale non disponibile su questo dispositivo."
            Log.e(TAG, errorMessage)
            statusListener?.onError(errorMessage)
            return
        }

        val speechRecognizerIntentAndroid = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "it-IT")
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        try {
            speechRecognizerAndroid.startListening(speechRecognizerIntentAndroid)
            Log.i(TAG, "startListening chiamato.")
        } catch (e: SecurityException) {
            Log.e(TAG, "SecurityException in startListening: ${e.message}", e)
            statusListener?.onError("Errore di sicurezza nell'avvio dell'ascolto.")
        } catch (e: Exception) {
            Log.e(TAG, "Exception in startListening: ${e.message}", e)
            statusListener?.onError("Errore generico nell'avvio dell'ascolto.")
        }
    }

    override fun stop() {
        try {
            speechRecognizerAndroid.stopListening()
            Log.i(TAG, "stopListening chiamato.")
        } catch (e: Exception) {
            Log.e(TAG, "Exception in stopListening: ${e.message}", e)
            statusListener?.onError("Errore durante l'arresto dell'ascolto.")
        }
    }

    override fun destroy() {
        isListening = false
        try {
            speechRecognizerAndroid.destroy()
            Log.i(TAG, "SpeechRecognizer distrutto.")
        } catch (e: Exception) {
            Log.e(TAG, "Exception in destroy: ${e.message}", e)
        }
    }

    private fun getErrorText(errorCode: Int): String =
        when (errorCode) {
            SpeechRecognizer.ERROR_AUDIO -> "Errore audio"
            SpeechRecognizer.ERROR_CLIENT -> "Errore lato client (possibile timeout o problema di configurazione)"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Permessi insufficienti"
            SpeechRecognizer.ERROR_NETWORK -> "Errore di rete (STT Android potrebbe richiederla per alcuni modelli linguistici o download)"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Timeout di rete"
            SpeechRecognizer.ERROR_NO_MATCH -> "Nessuna corrispondenza trovata (nessun parlato riconosciuto)"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Servizio di riconoscimento occupato"
            SpeechRecognizer.ERROR_SERVER -> "Errore dal server (se il riconoscimento è server-based)"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Timeout: nessun input vocale rilevato"
            SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED -> "Lingua non supportata"
            SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "Lingua non disponibile al momento"
            SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> "Disconnesso dal server"
            SpeechRecognizer.ERROR_TOO_MANY_REQUESTS -> "Troppe richieste"
            else -> "Errore sconosciuto di SpeechRecognizer ($errorCode)"
        }
}