package com.example.demospeechtoform

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

class AndroidSpeechRecognizer(context: Context) : Transcriber {
    private val speechRecognizerAndroid: SpeechRecognizer

    init {
        speechRecognizerAndroid = SpeechRecognizer.createSpeechRecognizer(context)

        speechRecognizerAndroid.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                TODO("Not yet implemented")
            }

            override fun onBeginningOfSpeech() {
                TODO("Not yet implemented")
            }

            override fun onRmsChanged(rmsdB: Float) {
                TODO("Not yet implemented")
            }

            override fun onBufferReceived(buffer: ByteArray?) {
                TODO("Not yet implemented")
            }

            override fun onEndOfSpeech() {
                TODO("Not yet implemented")
            }

            override fun onError(error: Int) {
                TODO("Not yet implemented")
            }

            override fun onResults(results: Bundle?) {
                TODO("Not yet implemented")
            }

            override fun onPartialResults(partialResults: Bundle?) {
                TODO("Not yet implemented")
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
                TODO("Not yet implemented")
            }
        })
    }

    override fun addListener(listener: TextChunkListener) {
        TODO("Not yet implemented")
    }

    override fun removeListener(listener: TextChunkListener) {
        TODO("Not yet implemented")
    }

    override fun begin() {
        val speechRecognizerIntentAndroid= Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "it-IT")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        speechRecognizerAndroid.startListening(speechRecognizerIntentAndroid)
    }

    override fun stop() {
        TODO("Not yet implemented")
    }
}