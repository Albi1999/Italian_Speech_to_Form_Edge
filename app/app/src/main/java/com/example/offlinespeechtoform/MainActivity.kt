package com.example.offlinespeechtoform

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.io.IOException
import android.content.res.AssetManager
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream

class MainActivity : ComponentActivity() {

    private lateinit var registraButtonAndroid: Button
    private lateinit var registraButtonVosk: Button
    private lateinit var trascrizioneTextView: TextView
    private lateinit var outputTextView: TextView // Per il futuro output NER

    // Android Speech Recognition
    private lateinit var speechRecognizerAndroid: SpeechRecognizer
    private lateinit var speechRecognizerIntentAndroid: Intent
    private var isRecordingAndroid = false

    // Vosk Speech Recognition
    private var voskModel: Model? = null
    private var voskRecognizer: Recognizer? = null
    private var isRecordingVosk = false
    private var audioRecordVosk: AudioRecord? = null
    private lateinit var voskRecordingThread: Thread
    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormatEncoding = AudioFormat.ENCODING_PCM_16BIT
    private var bufferSizeVosk = 0 // Will be calculated based on AudioRecord.getMinBufferSize

    private lateinit var requestPermissionLauncher: ActivityResultLauncher<String>

    private val VOSK_MODEL_NAME = "vosk-model-small-it-0.4"
    private val VOSK_MODEL_ASSET_PATH = "vosk-model-small-it-0.4"

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        registraButtonAndroid = findViewById(R.id.buttonRegistra)
        registraButtonVosk = findViewById(R.id.buttonVosk)
        trascrizioneTextView = findViewById(R.id.textViewTrascrizione)
        outputTextView = findViewById(R.id.textViewOutput)

        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormatEncoding)
        if (minBufferSize > 0) {
            bufferSizeVosk = minBufferSize
            Log.d(TAG, "bufferSizeVosk calcolato: $bufferSizeVosk")
        } else {
            Log.e(TAG, "Errore nel calcolare bufferSizeVosk, codice errore: $minBufferSize. Impossibile usare Vosk.")
            registraButtonVosk.isEnabled = false
            trascrizioneTextView.text = "Errore inizializzazione audio Vosk (buffer size)."
        }


        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                trascrizioneTextView.text = "Permesso microfono concesso. Riprova ad avviare la registrazione."
            } else {
                trascrizioneTextView.text = "Permesso microfono negato."
                isRecordingAndroid = false
                isRecordingVosk = false
                updateButtonUIAndroid(false)
                updateButtonUIVosk(false)
            }
        }

        setupAndroidSpeechRecognizer()
        if (bufferSizeVosk > 0) {
            initVosk()
        }

        registraButtonAndroid.setOnClickListener {
            handleAndroidRecording()
        }

        registraButtonVosk.setOnClickListener {
            if (bufferSizeVosk > 0) {
                handleVoskRecording()
            } else {
                Log.e(TAG, "Tentativo di avviare Vosk ma bufferSize non è valido.")
                trascrizioneTextView.text = "Errore inizializzazione audio Vosk."
            }
        }

        updateButtonUIAndroid(false)
        updateButtonUIVosk(false)
        registraButtonVosk.isEnabled = false
    }

    private fun setupAndroidSpeechRecognizer() {
        speechRecognizerAndroid = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntentAndroid = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "it-IT")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true) // Per risultati parziali
        }

        speechRecognizerAndroid.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Log.d(TAG, "Android STT: Ready for speech")
                trascrizioneTextView.text = "In ascolto (Android)..."
            }

            override fun onBeginningOfSpeech() {
                Log.d(TAG, "Android STT: Beginning of speech")
            }

            override fun onRmsChanged(rmsdB: Float) {}

            override fun onBufferReceived(buffer: ByteArray?) {
                Log.d(TAG, "Android STT: onBufferReceived")
            }

            override fun onEndOfSpeech() {
                Log.d(TAG, "Android STT: End of speech")
            }

            override fun onError(error: Int) {
                Log.e(TAG, "Android STT Error: ${getErrorText(error)}")
                trascrizioneTextView.text = "Errore Android STT: ${getErrorText(error)}"
                isRecordingAndroid = false
                updateButtonUIAndroid(false)
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null && matches.isNotEmpty()) {
                    val text = matches[0]
                    trascrizioneTextView.text = "Android: $text"
                    Log.d(TAG, "Android STT Result: $text")
                    // Here you would pass 'text' to your NER module
                    processWithNER(text)
                }
                isRecordingAndroid = false
                updateButtonUIAndroid(false)
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (matches != null && matches.isNotEmpty()) {
                    val partialText = matches[0]
                    trascrizioneTextView.text = "Android (parziale): $partialText"
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    private fun handleAndroidRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            if (!isRecordingAndroid) {
                startSpeechRecognitionAndroid()
            } else {
                stopSpeechRecognitionAndroid()
            }
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startSpeechRecognitionAndroid() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            trascrizioneTextView.text = "Riconoscimento vocale Android non disponibile."
            Log.e(TAG, "Android STT: Not available on this device.")
            return
        }
        trascrizioneTextView.text = "Avvio riconoscimento (Android)..."
        Log.d(TAG, "startListening (Android) chiamato.")
        isRecordingAndroid = true
        updateButtonUIAndroid(true)
        speechRecognizerAndroid.startListening(speechRecognizerIntentAndroid)
    }

    private fun stopSpeechRecognitionAndroid() {
        speechRecognizerAndroid.stopListening()
    }



    // VOSK MODEL -------------------------------------------------------------------------------------------

    @Throws(IOException::class)
    private fun copyAssetFolder(assetManager: AssetManager, fromAssetPath: String, toFolderPath: String) {
        Log.d(TAG, "Inizio copia da assets: '$fromAssetPath' a '$toFolderPath'")
        val files = assetManager.list(fromAssetPath)
            ?: throw IOException("Asset path '$fromAssetPath' non trovato o non è una directory.")

        val targetDir = File(toFolderPath)
        if (!targetDir.exists()) {
            if (!targetDir.mkdirs()) {
                throw IOException("Impossibile creare la directory di destinazione: $toFolderPath")
            }
        }

        for (filename in files) {
            val sourceAssetPath = if (fromAssetPath.isEmpty()) filename else "$fromAssetPath/$filename"
            val destFilePath = "$toFolderPath/$filename"
            var inputStream: InputStream? = null
            var outputStream: OutputStream? = null

            try {
                val subFiles = assetManager.list(sourceAssetPath)
                if (subFiles != null && subFiles.isNotEmpty()) {
                    Log.d(TAG, "Creazione sottodirectory e ricorsione per: $sourceAssetPath")
                    copyAssetFolder(assetManager, sourceAssetPath, destFilePath)
                } else {
                    inputStream = assetManager.open(sourceAssetPath)
                    outputStream = FileOutputStream(destFilePath)
                    inputStream.copyTo(outputStream)
                    Log.d(TAG, "Copiato file asset: $sourceAssetPath -> $destFilePath")
                }
            } catch (e: FileNotFoundException) {
                Log.w(TAG, "Asset non trovato o è una directory (gestito da logica subFiles): $sourceAssetPath")
            } finally {
                inputStream?.close()
                outputStream?.close()
            }
        }
        Log.d(TAG, "Fine copia da assets: '$fromAssetPath'")
    }


    private fun initVosk() {
        LibVosk.setLogLevel(LogLevel.INFO)
        val modelDir = File(getExternalFilesDir(null), VOSK_MODEL_NAME)

        if (!modelDir.exists() || modelDir.list()?.isEmpty() == true) {
            trascrizioneTextView.text = "Modello Vosk non trovato, copia da assets..."
            registraButtonVosk.isEnabled = false
            Log.i(TAG, "Modello Vosk non presente o vuoto in ${modelDir.absolutePath}. Copia da assets: '$VOSK_MODEL_ASSET_PATH'")

            Thread {
                try {
                    copyAssetFolder(assets, VOSK_MODEL_ASSET_PATH, modelDir.absolutePath)
                    Log.i(TAG, "Copia del modello da assets completata con successo.")

                    loadVoskModelInternal(modelDir.absolutePath)

                } catch (e: IOException) {
                    Log.e(TAG, "Errore durante la copia del modello Vosk da assets: ${e.message}", e)
                    runOnUiThread {
                        trascrizioneTextView.text = "Errore copia modello Vosk."
                        registraButtonVosk.isEnabled = false
                    }
                }
            }.start()
        } else {
            Log.i(TAG, "Modello Vosk già presente in ${modelDir.absolutePath}. Caricamento...")
            loadVoskModelInternal(modelDir.absolutePath)
        }
    }

    private fun loadVoskModelInternal(modelPath: String) {
        try {
            voskModel = Model(modelPath)
            voskRecognizer = Recognizer(voskModel, sampleRate.toFloat())
            Log.i(TAG, "Modello Vosk caricato correttamente da $modelPath.")
            runOnUiThread {
                trascrizioneTextView.text = "Modello Vosk pronto."
                registraButtonVosk.isEnabled = true
            }
        } catch (e: IOException) {
            Log.e(TAG, "Errore nel caricare il modello Vosk da $modelPath: ${e.localizedMessage}", e)
            runOnUiThread {
                trascrizioneTextView.text = "Errore caricamento modello Vosk (IOException)."
                registraButtonVosk.isEnabled = false
            }
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Errore UnsatisfiedLinkError (librerie native Vosk/JNA?): ${e.message}", e)
            runOnUiThread {
                trascrizioneTextView.text = "Errore librerie native Vosk."
                registraButtonVosk.isEnabled = false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Errore generico durante l'inizializzazione del modello Vosk: ${e.message}", e)
            runOnUiThread {
                trascrizioneTextView.text = "Errore iniz. modello Vosk (generico)."
                registraButtonVosk.isEnabled = false
            }
        }
    }


    private fun handleVoskRecording() {
        if (voskModel == null || voskRecognizer == null) {
            trascrizioneTextView.text = "Modello Vosk non ancora pronto."
            Log.w(TAG, "Tentativo di registrazione Vosk ma il modello/riconoscitore non è inizializzato.")
            initVosk()
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            if (!isRecordingVosk) {
                startVoskRecognition()
            } else {
                stopVoskRecognition()
            }
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startVoskRecognition() {
        if (bufferSizeVosk <= 0) {
            Log.e(TAG, "startVoskRecognition chiamato ma bufferSizeVosk non è valido: $bufferSizeVosk")
            trascrizioneTextView.text = "Errore Audio Vosk (buffer)."
            isRecordingVosk = false
            updateButtonUIVosk(false)
            return
        }

        if (voskRecognizer == null) {
            Log.e(TAG, "Riconoscitore Vosk non inizializzato.")
            trascrizioneTextView.text = "Errore: Riconoscitore Vosk non pronto."
            return
        }
        trascrizioneTextView.text = "Avvio riconoscimento (Vosk)..."
        isRecordingVosk = true
        updateButtonUIVosk(true)

        try {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Vosk: Permesso RECORD_AUDIO non concesso prima di creare AudioRecord.")
                isRecordingVosk = false
                updateButtonUIVosk(false)
                trascrizioneTextView.text = "Vosk: Permesso microfono mancante."
                return
            }

            audioRecordVosk = AudioRecord(
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                sampleRate,
                channelConfig,
                audioFormatEncoding,
                bufferSizeVosk
            )
        } catch (e: SecurityException) {
            Log.e(TAG, "Vosk: SecurityException durante la creazione di AudioRecord. ${e.message}", e)
            isRecordingVosk = false
            updateButtonUIVosk(false)
            trascrizioneTextView.text = "Vosk: Errore sicurezza AudioRecord."
            return
        }


        if (audioRecordVosk?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "Impossibile inizializzare AudioRecord per Vosk.")
            trascrizioneTextView.text = "Errore AudioRecord Vosk."
            isRecordingVosk = false
            updateButtonUIVosk(false)
            audioRecordVosk?.release()
            audioRecordVosk = null
            return
        }

        audioRecordVosk?.startRecording()
        Log.d(TAG, "AudioRecord Vosk avviato.")

        voskRecordingThread = Thread {
            val buffer = ByteArray(bufferSizeVosk)
            var lastPartialResultTime = System.currentTimeMillis()

            while (isRecordingVosk && audioRecordVosk != null && audioRecordVosk?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val bytesRead = audioRecordVosk?.read(buffer, 0, buffer.size) ?: 0
                if (bytesRead > 0) {
                    if (voskRecognizer?.acceptWaveForm(buffer, bytesRead) == true) {
                        val resultJson = voskRecognizer?.result
                        if (!resultJson.isNullOrEmpty()) {
                            val mainResultText = parseVoskResult(resultJson)
                            Log.d(TAG, "Vosk Result: $mainResultText (da result)")
                            runOnUiThread {
                                trascrizioneTextView.text = "Vosk: $mainResultText"
                            }
                        }
                    } else {
                        if (System.currentTimeMillis() - lastPartialResultTime > 300) { // Ogni 300ms
                            val partialResultJson = voskRecognizer?.partialResult
                            if (!partialResultJson.isNullOrEmpty()) {
                                val partialText = parseVoskPartialResult(partialResultJson)
                                if (partialText.isNotBlank()) {
                                    Log.d(TAG, "Vosk Partial: $partialText")
                                    runOnUiThread {
                                        trascrizioneTextView.append("\n(Vosk parziale: $partialText)")
                                    }
                                }
                                lastPartialResultTime = System.currentTimeMillis()
                            }
                        }
                    }
                } else if (bytesRead < 0) {
                    Log.e(TAG, "Errore durante la lettura da AudioRecord: $bytesRead")
                }
                // Breve pausa per non stressare la CPU inutilmente se non ci sono dati disponibili
                // if (bytesRead == 0) Thread.sleep(10)
            }

            try {
                audioRecordVosk?.stop()
                audioRecordVosk?.release()
                audioRecordVosk = null
                Log.d(TAG, "AudioRecord Vosk stoppato e rilasciato.")

                val finalResultJson = voskRecognizer?.finalResult
                if (!finalResultJson.isNullOrEmpty()) {
                    val finalText = parseVoskResult(finalResultJson)
                    Log.i(TAG, "Vosk Final Result: $finalText")
                    runOnUiThread {
                        trascrizioneTextView.text = "Vosk (finale): $finalText"
                        processWithNER(finalText)
                    }
                }
                voskRecognizer?.reset()
            } catch (e: Exception) {
                Log.e(TAG, "Errore durante lo stop/rilascio di AudioRecord o finalResult Vosk: ${e.message}", e)
            }
            Log.d(TAG, "Thread di registrazione Vosk terminato.")
        }
        voskRecordingThread.name = "VoskRecordingThread"
        voskRecordingThread.start()
    }

    private fun stopVoskRecognition() {
        isRecordingVosk = false
        updateButtonUIVosk(false)
        trascrizioneTextView.append("\nRegistrazione Vosk in arresto...")

        Log.i(TAG, "stopVoskRecognition chiamato, isRecordingVosk impostato su false.")
    }

    private fun parseVoskResult(jsonResult: String): String {
        return try {
            val jsonObject = JSONObject(jsonResult)
            jsonObject.optString("text", "")
        } catch (e: Exception) {
            Log.e(TAG, "Errore parsing JSON Vosk (result/finalResult): $jsonResult", e)
            jsonResult
        }
    }
    private fun parseVoskPartialResult(jsonPartialResult: String): String {
        return try {
            val jsonObject = JSONObject(jsonPartialResult)
            jsonObject.optString("partial", "")
        } catch (e: Exception) {
            Log.e(TAG, "Errore parsing JSON Vosk (partialResult): $jsonPartialResult", e)
            ""
        }
    }


    private fun updateButtonUIAndroid(isRecording: Boolean) {
        if (isRecording) {
            registraButtonAndroid.text = "Stoppa (Android)"
            registraButtonAndroid.setBackgroundColor(Color.RED)
            registraButtonVosk.isEnabled = false
        } else {
            registraButtonAndroid.text = "Avvia (Android)"
            registraButtonAndroid.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_blue_light))

            registraButtonVosk.isEnabled = (voskModel != null && voskRecognizer != null)
        }
    }

    private fun updateButtonUIVosk(isRecording: Boolean) {
        if (isRecording) {
            registraButtonVosk.text = "Stoppa (Vosk)"
            registraButtonVosk.setBackgroundColor(Color.RED)
            registraButtonAndroid.isEnabled = false
        } else {
            registraButtonVosk.text = "Avvia (Vosk)"
            registraButtonVosk.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
            registraButtonAndroid.isEnabled = true
        }
        if (!isRecording && (voskModel == null || voskRecognizer == null || bufferSizeVosk <= 0)) {
            registraButtonVosk.isEnabled = false
        } else if (!isRecording) {

            registraButtonVosk.isEnabled = true
        }

    }

    // Funzione placeholder per il tuo modulo NER
    private fun processWithNER(text: String) {
        Log.d(TAG, "Testo da inviare al NER: $text")
        outputTextView.text = "NER Input: $text \n(Implementare logica NER)"
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizerAndroid.destroy()

        if (::voskRecordingThread.isInitialized && voskRecordingThread.isAlive) {
            isRecordingVosk = false
            try {
                voskRecordingThread.join(500)
            } catch (e: InterruptedException) {
                Thread.currentThread().interrupt()
                Log.e(TAG, "Interruzione durante l'attesa del thread Vosk.", e)
            }
        }

        audioRecordVosk?.release()
        audioRecordVosk = null

        voskRecognizer?.close()
        voskModel?.close()
        voskRecognizer = null
        voskModel = null
        Log.i(TAG, "Risorse Vosk e Android STT rilasciate in onDestroy.")
    }

    fun getErrorText(errorCode: Int): String =
        when (errorCode) {
            SpeechRecognizer.ERROR_AUDIO -> "Errore audio"
            SpeechRecognizer.ERROR_CLIENT -> "Errore lato client"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Permessi insufficienti"
            SpeechRecognizer.ERROR_NETWORK -> "Errore di rete (STT Android potrebbe richiederla per alcuni modelli)"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Timeout di rete"
            SpeechRecognizer.ERROR_NO_MATCH -> "Nessuna corrispondenza"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Servizio di riconoscimento occupato"
            SpeechRecognizer.ERROR_SERVER -> "Errore dal server"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Nessun input vocale rilevato (timeout)"
            SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED -> "Lingua non supportata"
            SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "Lingua non disponibile al momento"
            SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> "Disconnesso dal server"
            SpeechRecognizer.ERROR_TOO_MANY_REQUESTS -> "Troppe richieste"
            else -> "Errore sconosciuto ($errorCode)"
        }
}