package com.example.demospeechtoform

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.activity.ComponentActivity

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.layout)

        val buttonOnline = findViewById<Button>(R.id.buttonOnline)
        val buttonOffline = findViewById<Button>(R.id.buttonOffline)

        buttonOnline.setOnClickListener { // Listener per buttonOnline
            val myIntent = Intent(
                this,
                OnlineSpeechToForm::class.java
            )
            startActivity(myIntent)
        }

        buttonOffline.setOnClickListener { // Listener per buttonOffline
            val myIntent = Intent(
                this,
                OfflineSpeechToFormActivity::class.java
            )
            startActivity(myIntent)
        }
    }
}