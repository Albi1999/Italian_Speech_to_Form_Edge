package com.example.demospeechtoform

interface TextChunkListener {
    fun onPartialResult(text: String)
    fun onFinalResult(text: String)
}