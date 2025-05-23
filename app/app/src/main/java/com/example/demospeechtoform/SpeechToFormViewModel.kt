package com.example.demospeechtoform

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.squareup.moshi.JsonAdapter
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import retrofit2.HttpException
import java.io.IOException
import java.io.InputStream
import java.lang.StringBuilder

sealed class SpeechToFormUiState {
    object AwaitingTranscription : SpeechToFormUiState()
    data class Transcribing(val confirmedText: String, val partialText: String) : SpeechToFormUiState()
    data class TranscriptionComplete(val fullText: String) : SpeechToFormUiState()

    // Stati specifici per Online LLM
    object ProcessingApi : SpeechToFormUiState()
    data class ApiResult(val formattedData: String) : SpeechToFormUiState()

    // Stati specifici per Offline SLM
    data class ModelLoading(val message: String) : SpeechToFormUiState()
    object ProcessingSlm : SpeechToFormUiState()
    data class SlmResult(val formattedData: String) : SpeechToFormUiState()

    data class Error(val message: String) : SpeechToFormUiState()
}

class SpeechToFormViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<SpeechToFormUiState>(SpeechToFormUiState.AwaitingTranscription)
    val uiState: StateFlow<SpeechToFormUiState> = _uiState.asStateFlow()

    // --- Trascrizione ---
    private lateinit var transcriber: Transcriber
    private var transcriberInitialized = false
    private val confirmedTranscript = StringBuilder()
    private var currentPartialTranscript: String = ""

    // --- Modalità e Processori ---
    private var isOnlineMode: Boolean = true
    private var apiClient: ApiClient? = null

    // --- SLM OFFLINE ---
    private var offlineSlmProcessorInstance: OfflineSlmProcessor? = null
    internal var selectedLlmModel: SlmModel = SlmModel.defaultModel()

    // --- Utilities ---
    private val moshi: Moshi by lazy {
        Moshi.Builder()
            .add(KotlinJsonAdapterFactory())
            .build()
    }

    companion object {
        private const val TAG = "SpeechToFormViewModel"
        private const val REPORT_SCHEMA_FILENAME = "macro_schema.json"
    }

    fun initTranscriberAndMode(isOnline: Boolean, modelToUseForOffline: SlmModel = selectedLlmModel) {
        this.isOnlineMode = isOnline
        val previousModel = this.selectedLlmModel
        this.selectedLlmModel = modelToUseForOffline

        Log.i(TAG, "Attempting to initialize ViewModel. Online mode: $isOnline. Offline model: ${this.selectedLlmModel.name}")

        if (isOnline) {
            if (offlineSlmProcessorInstance != null) {
                Log.d(TAG, "Mode changed to Online. Closing existing OfflineSlmProcessor.")
                offlineSlmProcessorInstance?.close()
                offlineSlmProcessorInstance = null
            }
        } else {
            apiClient = null
            if (previousModel != this.selectedLlmModel ||
                offlineSlmProcessorInstance == null ||
                offlineSlmProcessorInstance?.modelConfig != this.selectedLlmModel ||
                offlineSlmProcessorInstance?.isInitialized == false) {

                Log.d(TAG, "Offline SLM processor needs (re)initialization. Current model: ${this.selectedLlmModel.name}")
                offlineSlmProcessorInstance?.close()
                offlineSlmProcessorInstance = null
                setupOfflineSlmProcessor()
            } else if (offlineSlmProcessorInstance != null && offlineSlmProcessorInstance!!.isInitialized) {
                Log.i(TAG, "OfflineSlmProcessor for ${this.selectedLlmModel.name} already available and initialized.")
                _uiState.value = SpeechToFormUiState.AwaitingTranscription
            }
        }

        if (!transcriberInitialized) {
            transcriber = AndroidSpeechRecognizer(getApplication(), object : TranscriberStatusListener {
                override fun onReadyForSpeech() {
                    Log.d(TAG, "TranscriberStatusListener: onReadyForSpeech")
                }
                override fun onBeginningOfSpeech() { /* Log.d(TAG, "TranscriberStatusListener: onBeginningOfSpeech") */ }
                override fun onEndOfSpeech() { /* Log.d(TAG, "TranscriberStatusListener: onEndOfSpeech") */ }
                override fun onError(errorDescription: String) {
                    Log.e(TAG, "TranscriberStatusListener STT Error: $errorDescription")
                    _uiState.value = SpeechToFormUiState.Error("${getApplication<Application>().getString(R.string.stt_error_prefix)}: $errorDescription")
                }
            })
            transcriber.addListener(object : TextChunkListener {
                override fun onPartialResult(text: String) {
                    currentPartialTranscript = text
                    _uiState.value = SpeechToFormUiState.Transcribing(confirmedTranscript.toString(), currentPartialTranscript)
                }
                override fun onFinalResult(text: String) {
                    if (text.isNotEmpty()) {
                        confirmedTranscript.append(text).append(" ")
                    }
                    currentPartialTranscript = ""
                    val finalTextResult = confirmedTranscript.toString().trim()
                    _uiState.value = SpeechToFormUiState.TranscriptionComplete(finalTextResult)
                    Log.d(TAG, "TranscriberListener Final result: $finalTextResult")
                }
            })
            transcriberInitialized = true
            Log.d(TAG, "Transcriber core initialized.")
        }

        if (isOnline) {
            val reportSchema = loadReportSchemaFromAssets()
            if (reportSchema == null) {
                _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.error_loading_form_schema))
                return
            }
            apiClient = ApiClient(reportSchema.requestedInformations, reportSchema.topic, reportSchema.language)
            Log.i(TAG, "ViewModel configured for ONLINE mode.")
            resetTranscriptionState()
            _uiState.value = SpeechToFormUiState.AwaitingTranscription
        } else {
            if (offlineSlmProcessorInstance?.isInitialized == true && _uiState.value !is SpeechToFormUiState.Error) {
                resetTranscriptionState()
                _uiState.value = SpeechToFormUiState.AwaitingTranscription
            } else if (_uiState.value !is SpeechToFormUiState.ModelLoading && _uiState.value !is SpeechToFormUiState.Error) {
                setupOfflineSlmProcessor()
            }
        }
    }

    private fun setupOfflineSlmProcessor() {
        if (_uiState.value is SpeechToFormUiState.ModelLoading && offlineSlmProcessorInstance == null) {
            Log.d(TAG, "Model setup for ${selectedLlmModel.name} is already in progress.")
            return
        }

        Log.d(TAG, "Starting SLM processor setup for: ${selectedLlmModel.name}")
        _uiState.value = SpeechToFormUiState.ModelLoading(
            getApplication<Application>().getString(R.string.model_loading_message, selectedLlmModel.name)
        )

        viewModelScope.launch {
            val wrapper = withContext(Dispatchers.IO) {
                Log.d(TAG, "Calling OfflineSlmProcessor.getInstance for ${selectedLlmModel.name} on IO thread.")
                OfflineSlmProcessor.getInstance(getApplication(), selectedLlmModel)
            }

            if (wrapper != null && wrapper.isInitialized) {
                offlineSlmProcessorInstance = wrapper
                Log.i(TAG, "OfflineSlmProcessor for ${selectedLlmModel.name} is ready.")
                _uiState.value = SpeechToFormUiState.AwaitingTranscription
            } else {
                Log.e(TAG, "Failed to get/initialize OfflineSlmProcessor for ${selectedLlmModel.name}.")
                _uiState.value = SpeechToFormUiState.Error(
                    getApplication<Application>().getString(R.string.slm_model_setup_error_detailed, selectedLlmModel.name)
                )
            }
        }
    }

    private fun resetTranscriptionState() {
        confirmedTranscript.clear()
        currentPartialTranscript = ""
        val currentState = _uiState.value
        if (currentState !is SpeechToFormUiState.Error &&
            currentState !is SpeechToFormUiState.ModelLoading &&
            currentState !is SpeechToFormUiState.ProcessingSlm &&
            currentState !is SpeechToFormUiState.ProcessingApi) {
            _uiState.value = SpeechToFormUiState.AwaitingTranscription
        }
    }

    fun startTranscription() {
        if (!transcriberInitialized) {
            Log.e(TAG, "Transcriber not initialized. Cannot start.")
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.transcriber_not_initialized_error))
            return
        }
        if (!isOnlineMode && (offlineSlmProcessorInstance == null || offlineSlmProcessorInstance?.isInitialized == false)) {
            Log.w(TAG, "Offline SLM model not ready. Cannot start transcription.")
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.slm_model_not_ready_retry))
            setupOfflineSlmProcessor()
            return
        }

        resetTranscriptionState()
        transcriber.begin()
        _uiState.value = SpeechToFormUiState.Transcribing("", "")
    }

    fun stopTranscription() {
        if (transcriberInitialized && (_uiState.value is SpeechToFormUiState.Transcribing || _uiState.value is SpeechToFormUiState.AwaitingTranscription)) {
            transcriber.stop()
            val completeText = confirmedTranscript.toString().trim()
            if (_uiState.value !is SpeechToFormUiState.TranscriptionComplete && _uiState.value !is SpeechToFormUiState.Error) {
                _uiState.value = SpeechToFormUiState.TranscriptionComplete(completeText)
            }
        } else {
            Log.d(TAG, "Stop transcription called but not in a relevant state or transcriber not initialized.")
        }
    }

    fun handleStartStopRecording(isCurrentlyRecording: Boolean) {
        if (!isCurrentlyRecording) {
            startTranscription()
        } else {
            stopTranscription()
        }
    }

    fun callApiToExtractData(transcribedText: String) {
        if (!isOnlineMode) {
            Log.w(TAG, "callApiToExtractData called in offline mode. Ignored.")
            return
        }
        if (apiClient == null) {
            Log.e(TAG, "ApiClient not initialized for Online mode.")
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.api_client_not_initialized_error))
            return
        }
        if (transcribedText.isBlank()) {
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.no_text_to_process))
            return
        }
        if (BuildConfig.API_KEY.isBlank()) {
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.api_key_not_set))
            Log.e(TAG, "API Key is not set in BuildConfig.")
            return
        }

        _uiState.value = SpeechToFormUiState.ProcessingApi
        viewModelScope.launch {
            try {
                val response = apiClient!!.callApi(transcribedText)
                if (response.isSuccessful) {
                    val responseBodyMap = response.body()
                    if (responseBodyMap != null) {
                        val formattedResult = formatApiResponseMap(responseBodyMap)
                        _uiState.value = SpeechToFormUiState.ApiResult(formattedResult)
                    } else {
                        _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.api_response_empty_successful))
                    }
                } else {
                    val errorBody = response.errorBody()?.string()
                    var errorMsg = "${getApplication<Application>().getString(R.string.error_from_api_prefix)}: ${response.code()}"
                    if (!errorBody.isNullOrBlank()) {
                        errorMsg += "\n$errorBody"
                    }
                    _uiState.value = SpeechToFormUiState.Error(errorMsg)
                }
            } catch (e: HttpException) {
                Log.e(TAG, "API HTTP Exception: ${e.message()}", e)
                _uiState.value = SpeechToFormUiState.Error("${getApplication<Application>().getString(R.string.network_error_http_prefix)}: ${e.code()} ${e.message()}")
            } catch (e: IOException) {
                Log.e(TAG, "API IO Exception: ${e.message}", e)
                _uiState.value = SpeechToFormUiState.Error("${getApplication<Application>().getString(R.string.network_or_data_error_prefix)}: ${e.message}")
            } catch (e: Exception) {
                Log.e(TAG, "API Generic Exception: ${e.message}", e)
                _uiState.value = SpeechToFormUiState.Error("${getApplication<Application>().getString(R.string.unexpected_error_prefix)}: ${e.message}")
            }
        }
    }

    fun processTranscriptionWithSlm(fullText: String) {
        val textToProcess: String
        val isDummyTextUsed: Boolean

        if (fullText.isBlank()) {
            Log.d(TAG, "Testo trascritto vuoto. Utilizzo del testo dummy per il processamento SLM.")
            textToProcess = "Verbale redatto in data 12/04/2025 alle ore 19:58 per la violazione riscontrata al veicolo motoveicolo, marca Honda CBR650R, targato EF522BG (tipo ufficiale) proveniente da italia. In particolare, in via della Libertà, civico 21, è stato accertato parcheggio in doppia fila, in violazione dell'articolo 6, comma 4 del codice penale. Data l'assenza del conducente, la contestazione non è stata immediata.Si prevede la decurtazione di 6 punti dalla patente. Richiedo la stampa in italiano via wifi, non stampare anche la comunicazione."
            isDummyTextUsed = true
        } else {
            textToProcess = fullText
            isDummyTextUsed = false
        }

        if (offlineSlmProcessorInstance == null || !offlineSlmProcessorInstance!!.isInitialized) {
            Log.e(TAG, "OfflineSlmProcessor not ready for SLM. Model: ${selectedLlmModel.name}, Initialized: ${offlineSlmProcessorInstance?.isInitialized}")
            _uiState.value = SpeechToFormUiState.Error(getApplication<Application>().getString(R.string.slm_model_not_ready_retry))
            return
        }

        _uiState.value = SpeechToFormUiState.ProcessingSlm
        viewModelScope.launch(Dispatchers.IO) {
            val jsonStructureExample: String = loadReportSchemaFromAssets()?.let { schema ->
                try {
                    val inputStream: InputStream = getApplication<Application>().assets.open(REPORT_SCHEMA_FILENAME)
                    val schemaJsonString = inputStream.bufferedReader().use { it.readText() }
                    val jsonObject = moshi.adapter<Map<String, Any?>>(
                        com.squareup.moshi.Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java)
                    ).fromJson(schemaJsonString)
                    moshi.adapter<Map<String, Any?>>(
                        com.squareup.moshi.Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java)
                    ).indent("  ").toJson(jsonObject) ?: """{"error":"JSON pretty print failed"}"""
                } catch (e: Exception) {
                    Log.e(TAG, "Could not load/parse $REPORT_SCHEMA_FILENAME for SLM prompt.", e)
                    """{"error":"fallback schema used for SLM due to loading error"}"""
                }
            } ?: """{"error":"report schema not loaded for SLM"}"""

            val prompt = """
            CONTESTO: Il tuo compito è estrarre informazioni specifiche da un testo fornito da un utente e usarle per compilare un oggetto JSON. Devi seguire scrupolosamente lo schema JSON fornito.
            
            ISTRUZIONI PRINCIPALI:
            1. Analizza il TESTO UTENTE fornito qui sotto.
            2. Identifica e estrai le informazioni che corrispondono ai campi definiti nello SCHEMA JSON DA COMPILARE.
            3. Popola lo SCHEMA JSON DA COMPILARE utilizzando ESCLUSIVAMENTE le informazioni trovate nel TESTO UTENTE.
            4. Se un'informazione per un campo specifico non è presente nel TESTO UTENTE o non corrisponde ai valori accettati (se specificati nello schema), il valore di quel campo nel JSON finale deve essere `null` (il valore JSON null, non la stringa "null"). Non inventare dati.
            5. La tua risposta DEVE ESSERE ESCLUSIVAMENTE l'oggetto JSON risultante, completo e valido. Non includere NESSUN testo, commento, o spiegazione al di fuori dell'oggetto JSON stesso. Deve iniziare con `{` e finire con `}`.
            6. Per i campi con "accepted_values" nello schema, il valore estratto DEVE essere uno di quelli elencati.
            7. Per i campi numerici (es. "punti", "articolo", "comma", "Civico_1"), il valore nel JSON deve essere un numero, non una stringa.
            
            TESTO UTENTE:
            "$fullText"
            
            SCHEMA JSON DA COMPILARE (Usa questa esatta struttura e questi nomi di campo. Popola i valori basandoti sul TESTO UTENTE):
            $jsonStructureExample
            
            JSON COMPILATO:
            """.trimIndent()

            Log.d(TAG, "Processing with SLM model: ${selectedLlmModel.name}.  Dummy text in uso: $isDummyTextUsed")
            if(isDummyTextUsed) Log.d(TAG, "Testo Dummy inviato: $textToProcess")

            val result = offlineSlmProcessorInstance!!.generateResponseSlm(prompt)

            if (result.startsWith("Error:")) {
                _uiState.value = SpeechToFormUiState.Error(result)
            } else {
                var formattedSlmOutput = result
                Log.d(TAG, "Risultato grezzo SLM per ${selectedLlmModel.name}: $result")
                try {
                    val slmResultMapAdapter = moshi.adapter<Map<String, Any?>>(
                        com.squareup.moshi.Types.newParameterizedType(Map::class.java, String::class.java, Any::class.java)
                    )
                    val cleanResult = result.removePrefix("```json").removeSuffix("```").trim()
                    val slmDataMap = slmResultMapAdapter.fromJson(cleanResult)
                    if (slmDataMap != null) {
                        val wrappedSlmDataMap = mapOf("report" to slmDataMap)
                        formattedSlmOutput = formatApiResponseMap(wrappedSlmDataMap)
                    } else {
                        Log.w(TAG, "Risultato SLM per ${selectedLlmModel.name} non era una mappa JSON valida dopo la pulizia. Uso output grezzo.")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Impossibile parsare o formattare l'output JSON di SLM per ${selectedLlmModel.name}: ${e.message}. Uso output grezzo.")
                }
                _uiState.value = SpeechToFormUiState.SlmResult(formattedSlmOutput)
            }
        }
    }

    private fun loadReportSchemaFromAssets(): ReportSchemaAsset? {
        try {
            val inputStream: InputStream = getApplication<Application>().assets.open(REPORT_SCHEMA_FILENAME)
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            val adapter: JsonAdapter<ReportSchemaAsset> = moshi.adapter(ReportSchemaAsset::class.java)
            return adapter.fromJson(jsonString)
        } catch (e: IOException) {
            Log.e(TAG, "Error reading $REPORT_SCHEMA_FILENAME: ${e.message}", e)
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing $REPORT_SCHEMA_FILENAME: ${e.message}", e)
        }
        return null
    }

    @Suppress("UNCHECKED_CAST")
    private fun formatApiResponseMap(data: Map<String, Any?>): String {
        val builder = StringBuilder(getApplication<Application>().getString(R.string.extracted_information_title))
        return try {
            val report: Map<String, Any?> = data["report"] as? Map<String, Any?> ?: data
            for ((key, value) in report) {
                val formattedKey = key.replace("_", " ").split(" ")
                    .joinToString(" ") { it.replaceFirstChar(Char::titlecaseChar) }
                builder.append("$formattedKey: ")
                appendValueToBuilder(builder, value, 1)
                builder.append("\n\n")
            }
            if (builder.endsWith("\n\n")) builder.setLength(builder.length - 2)
            builder.toString().trim()
        } catch (e: Exception) {
            Log.e(TAG, "Error formatting API/SLM response map: ${e.message}", e)
            getApplication<Application>().getString(R.string.error_formatting_response_data)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun appendValueToBuilder(builder: StringBuilder, value: Any?, indentLevel: Int) {
        val indent = "  ".repeat(indentLevel)
        when (value) {
            is Map<*, *> -> {
                builder.append("\n")
                (value as? Map<String, Any?>)?.forEach { (k, v) ->
                    val formattedKey = k.replace("_", " ").split(" ")
                        .joinToString(" ") { it.replaceFirstChar(Char::titlecaseChar) }
                    builder.append("$indent$formattedKey: ")
                    appendValueToBuilder(builder, v, indentLevel + 1)
                    builder.append("\n")
                }
                if (builder.endsWith("\n")) builder.setLength(builder.length -1)
            }
            is List<*> -> {
                builder.append("\n")
                value.forEachIndexed { index, item ->
                    builder.append("$indent${getApplication<Application>().getString(R.string.list_item_prefix)} ${index + 1}:\n")
                    appendValueToBuilder(builder, item, indentLevel + 1)
                    if (item !is Map<*, *> && item !is List<*>) builder.append("\n")
                }
                if (builder.endsWith("\n")) builder.setLength(builder.length -1)
            }
            null -> builder.append(getApplication<Application>().getString(R.string.not_available_shorthand))
            else -> builder.append(value.toString())
        }
    }

    override fun onCleared() {
        super.onCleared()
        Log.d(TAG, "ViewModel onCleared called.")
        if (transcriberInitialized) {
            transcriber.destroy()
            transcriberInitialized = false
            Log.d(TAG, "Transcriber destroyed.")
        }
        offlineSlmProcessorInstance?.close()
        OfflineSlmProcessor.resetInstance()
        Log.d(TAG, "Offline SLM processor instance closed/reset.")
    }
}