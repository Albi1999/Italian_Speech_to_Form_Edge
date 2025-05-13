package com.example.offlinespeechtoform

import android.Manifest
import android.annotation.SuppressLint
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

@SuppressLint("SetTextI18n")
class MainActivity : ComponentActivity() {

    private lateinit var registraButtonAndroid: Button
    private lateinit var registraButtonVosk: Button
    private lateinit var trascrizioneTextView: TextView
    private lateinit var outputTextView: TextView

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
    private var bufferSizeVosk = 0
    private var isVoskModelReady = false

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

        updateButtonUIAndroid(false)
        updateButtonUIVosk(false)
        registraButtonVosk.isEnabled = false

        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormatEncoding)
        if (minBufferSize > 0) {
            bufferSizeVosk = minBufferSize
            Log.d(TAG, "bufferSizeVosk calcolato: $bufferSizeVosk")
        } else {
            Log.e(TAG, "Errore nel calcolare bufferSizeVosk, codice errore: $minBufferSize. Impossibile usare Vosk.")
            trascrizioneTextView.text = "Errore inizializzazione audio Vosk (buffer size)."
            outputTextView.text = "Vosk non è disponibile a causa di un errore nella configurazione dell'audio."
        }

        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                trascrizioneTextView.text = "Permesso microfono concesso. Puoi avviare la registrazione."
                if (isVoskModelReady && bufferSizeVosk > 0) {
                    registraButtonVosk.isEnabled = true
                }
            } else {
                trascrizioneTextView.text = "Permesso microfono negato. Impossibile registrare."
                isRecordingAndroid = false
                isRecordingVosk = false
                updateButtonUIAndroid(false)
                updateButtonUIVosk(false)
            }
        }

        setupAndroidSpeechRecognizer()
        if (bufferSizeVosk > 0) {
            initVosk()
        } else {
            Log.w(TAG, "Vosk non inizializzato a causa di bufferSize non valido.")
        }

        registraButtonAndroid.setOnClickListener {
            handleAndroidRecording()
        }

        registraButtonVosk.setOnClickListener {
            if (!isVoskModelReady) {
                trascrizioneTextView.text = "Modello Vosk non ancora pronto o errore."
                Log.w(TAG, "Pulsante Vosk premuto ma modello non pronto.")
                return@setOnClickListener
            }
            if (bufferSizeVosk > 0) {
                handleVoskRecording()
            } else {
                Log.e(TAG, "Tentativo di avviare Vosk ma bufferSize non è valido.")
                trascrizioneTextView.text = "Errore inizializzazione audio Vosk (buffer)."
            }
        }
    }


    // ANDROID SPEECH TO TEXT EMBEDDED -------------------------------------------------------------

    private fun setupAndroidSpeechRecognizer() {
        speechRecognizerAndroid = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntentAndroid = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "it-IT")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
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
        trascrizioneTextView.text = "Trascrizione in corso (Android)..."
        Log.d(TAG, "startListening (Android) chiamato.")
        isRecordingAndroid = true
        updateButtonUIAndroid(true)
        speechRecognizerAndroid.startListening(speechRecognizerIntentAndroid)
    }

    private fun stopSpeechRecognitionAndroid() {
        speechRecognizerAndroid.stopListening()
    }



    // VOSK MODEL ----------------------------------------------------------------------------------

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

        trascrizioneTextView.text = "Controllo modello Vosk..."
        registraButtonVosk.isEnabled = false

        Thread {
            var modelPathToLoad: String? = null
            if (!modelDir.exists() || modelDir.list()?.isEmpty() == true) {
                runOnUiThread {
                    trascrizioneTextView.text = "Modello Vosk non trovato, avvio copia da assets ($VOSK_MODEL_NAME)... Questo potrebbe richiedere alcuni istanti."
                    outputTextView.text = "Attendere il completamento della copia del modello."
                }
                Log.i(TAG, "Modello Vosk non presente o vuoto in ${modelDir.absolutePath}. Copia da assets: '$VOSK_MODEL_ASSET_PATH'")
                try {
                    copyAssetFolder(assets, VOSK_MODEL_ASSET_PATH, modelDir.absolutePath)
                    Log.i(TAG, "Copia del modello da assets completata con successo.")
                    modelPathToLoad = modelDir.absolutePath
                } catch (e: IOException) {
                    Log.e(TAG, "Errore durante la copia del modello Vosk da assets: ${e.message}", e)
                    runOnUiThread {
                        trascrizioneTextView.text = "ERRORE: Impossibile copiare il modello Vosk."
                        outputTextView.text = "Dettagli errore: ${e.message}"
                        isVoskModelReady = false
                        updateButtonUIVosk(false)
                    }
                }
            } else {
                Log.i(TAG, "Modello Vosk già presente in ${modelDir.absolutePath}. Caricamento...")
                modelPathToLoad = modelDir.absolutePath
            }

            if (modelPathToLoad != null) {
                loadVoskModelInternal(modelPathToLoad)
            }
        }.start()
    }

    private fun loadVoskModelInternal(modelPath: String) {
        runOnUiThread {
            trascrizioneTextView.text = "Caricamento modello Vosk in corso..."
        }
        try {
            voskModel = Model(modelPath)
            voskRecognizer = Recognizer(voskModel, sampleRate.toFloat())
            Log.i(TAG, "Modello Vosk caricato correttamente da $modelPath.")
            isVoskModelReady = true
            runOnUiThread {
                trascrizioneTextView.text = "Modello Speech To Text caricato e pronto alla trascrizione"
                if (bufferSizeVosk > 0 && ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    registraButtonVosk.isEnabled = true
                } else if (bufferSizeVosk <= 0) {
                    trascrizioneTextView.append("\\nErrore: Buffer audio non valido per Vosk.")
                    registraButtonVosk.isEnabled = false
                } else {
                    trascrizioneTextView.append("\\nConcedere il permesso microfono per usare Vosk.")
                    registraButtonVosk.isEnabled = false
                }
                updateButtonUIVosk(false)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Errore nel caricare il modello Vosk da $modelPath: ${e.localizedMessage}", e)
            isVoskModelReady = false
            runOnUiThread {
                val errorMsg: String
                when (e) {
                    is IOException -> errorMsg = "Errore I/O caricamento modello Vosk."
                    else -> errorMsg = "Errore sconosciuto caricamento modello Vosk."
                }
                trascrizioneTextView.text = errorMsg
                outputTextView.text = "Dettagli: ${e.message}. Vosk non sarà disponibile."
                registraButtonVosk.isEnabled = false
                updateButtonUIVosk(false)
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
        trascrizioneTextView.text = "Trascrizione in corso (Vosk)..."
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
            var lastRecognizedPartialText = ""
            var lastPartialDisplayTime = 0L

            val accumulatedSegmentText = StringBuilder()

            Log.d(TAG, "Vosk Recording Thread avviato (stop manuale).")

            while (isRecordingVosk && audioRecordVosk != null && audioRecordVosk?.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                val bytesRead = audioRecordVosk?.read(buffer, 0, buffer.size) ?: 0
                if (bytesRead > 0) {
                    if (voskRecognizer?.acceptWaveForm(buffer, bytesRead) == true) {
                        val resultJson = voskRecognizer?.result
                        if (!resultJson.isNullOrEmpty()) {
                            val mainResultText = parseVoskResult(resultJson)
                            if (mainResultText.isNotBlank()) {
                                Log.d(TAG, "Vosk Result (segmento): $mainResultText")
                                if (accumulatedSegmentText.isNotEmpty()) accumulatedSegmentText.append(" ")
                                accumulatedSegmentText.append(mainResultText)
                                lastRecognizedPartialText = ""
                                runOnUiThread {
                                    trascrizioneTextView.text = "Vosk: ${accumulatedSegmentText.toString()}"
                                    outputTextView.text = "Processando..."
                                }
                            } else {
                                Log.d(TAG, "Vosk Result (segmento vuoto)")
                            }
                        }
                    } else {
                        if (System.currentTimeMillis() - lastPartialDisplayTime > 300) { // Ogni 300ms
                            val currentPartialJson = voskRecognizer?.partialResult
                            val currentPartialText = if (currentPartialJson.isNullOrEmpty()) "" else parseVoskPartialResult(currentPartialJson)

                            if (currentPartialText.isNotBlank() && currentPartialText != lastRecognizedPartialText) {
                                Log.d(TAG, "Vosk Partial (cambiato): $currentPartialText")
                                lastRecognizedPartialText = currentPartialText
                                runOnUiThread {
                                    val textToShow = if (accumulatedSegmentText.isNotEmpty()) {
                                        "${accumulatedSegmentText.toString()} $currentPartialText"
                                    } else {
                                        currentPartialText
                                    }
                                    trascrizioneTextView.text = "Vosk (parziale): $textToShow"
                                }
                            } else if (currentPartialText.isBlank() && lastRecognizedPartialText.isNotBlank()){
                                lastRecognizedPartialText = ""
                            }
                            lastPartialDisplayTime = System.currentTimeMillis()
                        }
                    }
                } else if (bytesRead < 0) {
                    Log.e(TAG, "Errore durante la lettura da AudioRecord: $bytesRead")
                } else {
                    Thread.sleep(50)
                }
            }

            Log.d(TAG, "Vosk Recording Thread: Uscita dal ciclo di registrazione. isRecordingVosk = $isRecordingVosk")

            try {
                audioRecordVosk?.stop()
                audioRecordVosk?.release()
                audioRecordVosk = null
                Log.d(TAG, "AudioRecord Vosk stoppato e rilasciato.")

                var textToProcessForNER = ""
                val finalResultJson = voskRecognizer?.finalResult
                if (!finalResultJson.isNullOrEmpty()) {
                    val finalText = parseVoskResult(finalResultJson)
                    if (finalText.isNotBlank()) {
                        Log.i(TAG, "Vosk Final Result: $finalText")
                        accumulatedSegmentText.clear().append(finalText)
                        textToProcessForNER = finalText
                    }
                }

                if (textToProcessForNER.isBlank() && accumulatedSegmentText.isNotEmpty()) {
                    Log.i(TAG, "Vosk Final Result era vuoto/blank, uso testo accumulato dai segmenti: ${accumulatedSegmentText.toString()}")
                    textToProcessForNER = accumulatedSegmentText.toString()
                }

                if (textToProcessForNER.isNotBlank()) {
                    val displayText = "Vosk (finale): $textToProcessForNER"
                    Log.i(TAG, "Testo finale per UI e NER: $textToProcessForNER")
                    runOnUiThread {
                        trascrizioneTextView.text = displayText
                        outputTextView.text = "Elaborazione NER in corso..."
                        processWithNER(textToProcessForNER)
                    }
                } else {
                    Log.i(TAG, "Nessun testo significativo finale da Vosk (né da finalResult né da segmenti).")
                    runOnUiThread {
                        if (!trascrizioneTextView.text.contains("(finale):") && !trascrizioneTextView.text.contains("Nessun risultato")) {
                            if (accumulatedSegmentText.isEmpty() && lastRecognizedPartialText.isNotEmpty() && trascrizioneTextView.text.contains(lastRecognizedPartialText)){
                                outputTextView.text = "Nessun risultato completo per NER."
                            } else if (accumulatedSegmentText.isNotEmpty()){
                                trascrizioneTextView.text = "Vosk: ${accumulatedSegmentText.toString()}"
                                outputTextView.text = "Nessun risultato finale conclusivo per NER."
                            }
                            else {
                                trascrizioneTextView.append("\nNessun risultato finale completo da Vosk.")
                                outputTextView.text = ""
                            }
                        }
                    }
                }
                voskRecognizer?.reset()
            } catch (e: Exception) {
                Log.e(TAG, "Errore durante lo stop/rilascio di AudioRecord o finalResult Vosk: ${e.message}", e)
                runOnUiThread{
                    trascrizioneTextView.append("\nErrore finalizzazione Vosk: ${e.message}")
                }
            } finally {
                Log.d(TAG, "Vosk Recording Thread terminato.")
                runOnUiThread {
                    updateButtonUIVosk(false)
                }
            }
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
            registraButtonAndroid.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_blue_dark))
            if (isVoskModelReady && bufferSizeVosk > 0) {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    registraButtonVosk.isEnabled = true
                } else {
                    registraButtonVosk.isEnabled = false
                }
            } else {
                registraButtonVosk.isEnabled = false
            }
        }
    }

    private fun updateButtonUIVosk(isRecording: Boolean) {
        if (isRecording) {
            registraButtonVosk.text = "Stoppa (Vosk)"
            registraButtonVosk.setBackgroundColor(Color.RED)
            registraButtonAndroid.isEnabled = false
        } else {
            registraButtonVosk.text = "Avvia (Vosk)"
            registraButtonVosk.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_dark))
            registraButtonAndroid.isEnabled = true
            if (!isVoskModelReady || bufferSizeVosk <= 0) {
                registraButtonVosk.isEnabled = false
            } else {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    registraButtonVosk.isEnabled = true
                } else {
                    registraButtonVosk.isEnabled = false
                }
            }
        }
    }

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