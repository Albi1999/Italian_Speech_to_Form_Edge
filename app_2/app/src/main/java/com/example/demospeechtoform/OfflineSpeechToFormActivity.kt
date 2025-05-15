package com.example.demospeechtoform

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.ui.AppBarConfiguration
import com.example.demospeechtoform.databinding.ActivityOfflineSpeechToFormBinding

class OfflineSpeechToFormActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityOfflineSpeechToFormBinding

    private lateinit var transcriber: Transcriber;

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_offline_speech_to_form)
        initTranscriber()

        var buttonStart = findViewById<Button>(R.id.buttonBeginSpeechToText)
        var buttonResume = findViewById<Button>(R.id.buttonResumeSpeechToText)
        var buttonShowResults = findViewById<Button>(R.id.buttonShowResult)
        var textView = findViewById<TextView>(R.id.scrollableText)

        transcriber.addListener(object : TextChunkListener {
            override fun newChunk(text: String) {
                textView.text = textView.text.toString() + " " + text
            }
        })

        buttonStart.setOnClickListener {
            transcriber.begin()
        }
    }

    fun initTranscriber() {
        transcriber  = AndroidSpeechRecognizer(this)
    }
}