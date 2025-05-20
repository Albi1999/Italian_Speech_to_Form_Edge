package com.example.demospeechtoform

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class MainActivity : ComponentActivity() {

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var startOnlineSpeechToForm: ActivityResultLauncher<Intent>
    private lateinit var startOfflineSpeechToForm: ActivityResultLauncher<Intent>

    private var pendingActivityIntent: Intent? = null

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

        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                pendingActivityIntent?.let {
                    startActivity(it)
                    pendingActivityIntent = null
                }
                Toast.makeText(this, getString(R.string.microphone_permission_granted), Toast.LENGTH_SHORT).show()
            } else {
                if (!shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) &&
                    ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_DENIED) {
                    showDialogToOpenAppSettings()
                } else {
                    Toast.makeText(this, getString(R.string.cannot_start_transcription_permission_denied), Toast.LENGTH_LONG).show()
                }
                pendingActivityIntent = null
            }
        }

        startOnlineSpeechToForm = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {}
        startOfflineSpeechToForm = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {}

        val buttonOnline = findViewById<Button>(R.id.buttonOnline)
        val buttonOffline = findViewById<Button>(R.id.buttonOffline)

        buttonOnline.setOnClickListener { // Listener per buttonOnline
            val intent = Intent(this, OnlineSpeechToForm::class.java)
            handlePermissionAndStartActivity(intent)
        }

        buttonOffline.setOnClickListener { // Listener per buttonOffline
            val intent = Intent(this, OfflineSpeechToForm::class.java)
            handlePermissionAndStartActivity(intent)
        }
    }

    private fun handlePermissionAndStartActivity(intent: Intent) {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED -> {
                startActivity(intent)
            }
            shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> {
                AlertDialog.Builder(this)
                    .setTitle(getString(R.string.permission_needed_title))
                    .setMessage(getString(R.string.microphone_permission_prompt_message))
                    .setPositiveButton(getString(R.string.button_ok)) { _, _ ->
                        pendingActivityIntent = intent
                        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                    .setNegativeButton(getString(R.string.button_cancel)) { dialog, _ ->
                        dialog.dismiss()
                        Toast.makeText(this, getString(R.string.microphone_permission_denied), Toast.LENGTH_SHORT).show()
                    }
                    .show()
            }
            else -> {
                pendingActivityIntent = intent
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun showDialogToOpenAppSettings() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.permission_needed_title))
            .setMessage(getString(R.string.microphone_permission_settings_message))
            .setPositiveButton(getString(R.string.button_open_settings)) { dialog, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                val uri = Uri.fromParts("package", packageName, null)
                intent.data = uri
                startActivity(intent)
                dialog.dismiss()
            }
            .setNegativeButton(getString(R.string.button_cancel)) { dialog, _ ->
                dialog.dismiss()
                Toast.makeText(this, getString(R.string.microphone_permission_denied), Toast.LENGTH_SHORT).show()
            }
            .setCancelable(false)
            .show()
    }
}