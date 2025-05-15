package com.example.demospeechtoform

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.activity.ComponentActivity

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)

        var buttonOnline = findViewById<Button>(R.id.buttonOnline)
        var buttonOffline = findViewById<Button>(R.id.buttonOffline)

        buttonOffline.setOnClickListener {
            val myIntent: Intent = Intent(
                this@MainActivity,
                OnlineSpeechToForm::class.java
            )
            startActivity(myIntent)
        }

        buttonOffline.setOnClickListener {
            val myIntent: Intent = Intent(
                this@MainActivity,
                OfflineSpeechToFormActivity::class.java
            )
            startActivity(myIntent)
        }
    }
}