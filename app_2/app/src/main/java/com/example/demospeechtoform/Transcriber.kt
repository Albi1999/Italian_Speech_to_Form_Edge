package com.example.demospeechtoform

interface Transcriber {
    fun addListener(listener: TextChunkListener)
    fun removeListener(listener: TextChunkListener)

    fun begin()
    fun stop()
}