package com.example.demospeechtoform

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

interface ApiService {

    /**
     * Defines the POST request to the report endpoint.
     *
     * @param clientApplication The 'X-Client-Application' header value.
     * @param clientTenant The 'X-Client-Tenant' header value.
     * @param authorization The 'Authorization' header value (API Key).
     * @param requestBody The request payload, containing the document and schema.
     * @return A Retrofit Response object. The body, if successful, will be a Map<String, Any?>
     * representing the dynamically structured JSON response.
     */
    @POST("document-analysis-api/report") // Endpoint path relative to base URL
    suspend fun getSummaryWithReport(
        @Header("X-Client-Application") clientApplication: String,
        @Header("X-Client-Tenant") clientTenant: String,
        @Header("Authorization") authorization: String,
        @Body requestBody: ApiRequest
    ): Response<Map<String, Any?>> // Expecting a dynamic map as response
}
