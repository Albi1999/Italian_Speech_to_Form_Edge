package com.example.demospeechtoform

interface TranscriberStatusListener {
    fun onReadyForSpeech()
    fun onBeginningOfSpeech()
    fun onEndOfSpeech()
    fun onError(errorDescription: String)
    // Aggiungere altri stati se necessario, es. onListeningStopped()
}