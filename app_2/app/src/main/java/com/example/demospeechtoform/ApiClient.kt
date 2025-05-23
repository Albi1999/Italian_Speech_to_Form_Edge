package com.example.demospeechtoform

import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Response
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit

class ApiClient(
    val requestedInformations: List<RequestedInformationItem>,
    val topic: String,
    val language: String
) {
    private val moshi: Moshi = Moshi.Builder()
        .add(KotlinJsonAdapterFactory())
        .build()
    private val loggingInterceptor : HttpLoggingInterceptor = HttpLoggingInterceptor().apply {
        level = if (BuildConfig.DEBUG) { // Show logs only in debug builds
            HttpLoggingInterceptor.Level.BODY
        } else {
            HttpLoggingInterceptor.Level.NONE
        }
    }
    private val okHttpClient : OkHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor) // Add the logging interceptor
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()
    private val apiService: ApiService = Retrofit.Builder()
        .baseUrl(BuildConfig.BASE_URL) // Use the URL from BuildConfig
        .client(okHttpClient)
        .addConverterFactory(MoshiConverterFactory.create(moshi))
        .build()
        .create(ApiService::class.java)

    suspend fun callApi(text: String): Response<Map<String, Any?>> {
        val result = apiService.getSummaryWithReport(
            BuildConfig.CLIENT_APPLICATION,
            BuildConfig.CLIENT_TENANT,
            "Bearer " + BuildConfig.API_KEY,
            ApiRequest(text, requestedInformations, topic, language))
        return result
    }
}
    