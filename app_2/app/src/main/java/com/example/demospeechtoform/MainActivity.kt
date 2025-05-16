package com.example.demospeechtoform

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContentView(R.layout.layout)

        val mainLayout = findViewById<ConstraintLayout>(R.id.MainLayout)
        ViewCompat.setOnApplyWindowInsetsListener(mainLayout) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

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
                OfflineSpeechToForm::class.java
            )
            startActivity(myIntent)
        }
    }
}